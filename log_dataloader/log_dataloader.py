
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1bd909f048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1bd909f048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1c44c481e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1c44c481e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1c62f8fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1c62f8fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bf1f77620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bf1f77620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bf1f77620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▉       | 2850816/9912422 [00:00<00:00, 28399108.29it/s]9920512it [00:00, 32623602.38it/s]                             
0it [00:00, ?it/s]32768it [00:00, 567774.71it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470095.34it/s]1654784it [00:00, 11754021.05it/s]                         
0it [00:00, ?it/s]8192it [00:00, 176475.29it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bef19a5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bef360c88>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1bf1f77268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1bf1f77268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1bf1f77268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:35,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:34,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:34,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:33,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:33,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:32,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:04,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:04,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:03,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:03,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:02,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:02,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:02,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:01,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:01,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:43,  3.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:43,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:42,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:42,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:41,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:41,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:41,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:40,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:40,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:40,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:28,  4.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:28,  4.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:27,  4.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:27,  4.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:26,  4.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:12,  9.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:12,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:12,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:12,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 16.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 16.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 16.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 23.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 23.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 23.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 23.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 23.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 23.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 24.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 24.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:03, 24.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:03, 24.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:02, 25.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:02, 26.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:02, 26.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:02, 26.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:02, 26.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:02, 26.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:02, 26.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:02, 26.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:02, 26.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:02, 26.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:02, 26.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:02, 26.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:02, 26.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:02, 26.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:02, 26.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:02, 26.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:02, 27.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:02, 27.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:02, 27.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:02, 27.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:02, 27.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 27.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 27.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 27.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 27.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 27.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 27.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 27.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 27.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 27.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 27.61 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:01, 27.61 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:01, 27.61 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:01, 27.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:01, 27.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:01, 27.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:01, 27.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:01, 27.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:01, 27.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:01, 27.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:01, 27.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:01, 27.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:01, 27.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:01, 27.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:01, 27.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:01, 27.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:01, 27.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:01, 27.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:01, 27.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:01, 27.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:01, 27.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:01, 27.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:01, 27.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:01, 27.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:01, 27.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:01, 27.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:01, 27.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 27.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 27.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 27.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 27.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 27.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 27.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 27.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:00, 27.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:00, 27.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:00, 27.82 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 27.82 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 27.82 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 27.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 27.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 27.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 27.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 27.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 27.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 27.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:04<00:00, 27.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 27.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 27.93 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 27.93 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 27.93 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 27.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 27.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 27.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 27.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 27.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 27.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 27.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 27.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 27.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 27.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 27.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 27.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 27.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 27.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.81s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.81s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 27.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.81s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 27.91 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.76s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.81s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 27.91 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.76s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.76s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 23.95 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.77s/ url]
0 examples [00:00, ? examples/s]2020-07-03 00:08:20.772954: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-03 00:08:20.786120: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-03 00:08:20.786297: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a09c0e0f70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-03 00:08:20.786314: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
42 examples [00:00, 419.05 examples/s]162 examples [00:00, 520.52 examples/s]279 examples [00:00, 623.94 examples/s]390 examples [00:00, 717.42 examples/s]510 examples [00:00, 815.59 examples/s]623 examples [00:00, 888.05 examples/s]739 examples [00:00, 953.59 examples/s]848 examples [00:00, 989.89 examples/s]954 examples [00:00, 1009.53 examples/s]1067 examples [00:01, 1042.20 examples/s]1179 examples [00:01, 1064.25 examples/s]1296 examples [00:01, 1092.41 examples/s]1410 examples [00:01, 1104.26 examples/s]1523 examples [00:01, 1110.17 examples/s]1640 examples [00:01, 1126.07 examples/s]1757 examples [00:01, 1136.54 examples/s]1872 examples [00:01, 1118.33 examples/s]1985 examples [00:01, 1084.68 examples/s]2099 examples [00:01, 1098.52 examples/s]2210 examples [00:02, 1101.30 examples/s]2324 examples [00:02, 1109.89 examples/s]2441 examples [00:02, 1125.05 examples/s]2561 examples [00:02, 1145.64 examples/s]2683 examples [00:02, 1164.81 examples/s]2806 examples [00:02, 1181.31 examples/s]2925 examples [00:02, 1180.39 examples/s]3044 examples [00:02, 1123.85 examples/s]3160 examples [00:02, 1132.44 examples/s]3275 examples [00:02, 1135.02 examples/s]3389 examples [00:03, 1134.63 examples/s]3503 examples [00:03, 1132.10 examples/s]3624 examples [00:03, 1151.87 examples/s]3745 examples [00:03, 1167.60 examples/s]3870 examples [00:03, 1188.91 examples/s]3993 examples [00:03, 1200.47 examples/s]4114 examples [00:03, 1192.40 examples/s]4234 examples [00:03, 1186.37 examples/s]4353 examples [00:03, 1186.66 examples/s]4472 examples [00:03, 1136.94 examples/s]4587 examples [00:04, 1122.89 examples/s]4702 examples [00:04, 1130.21 examples/s]4816 examples [00:04, 1118.08 examples/s]4929 examples [00:04, 1118.89 examples/s]5042 examples [00:04, 1114.47 examples/s]5156 examples [00:04, 1118.78 examples/s]5268 examples [00:04, 1111.29 examples/s]5380 examples [00:04, 1092.78 examples/s]5497 examples [00:04, 1112.48 examples/s]5616 examples [00:04, 1132.55 examples/s]5735 examples [00:05, 1146.43 examples/s]5853 examples [00:05, 1154.38 examples/s]5970 examples [00:05, 1158.97 examples/s]6088 examples [00:05, 1163.28 examples/s]6207 examples [00:05, 1169.20 examples/s]6326 examples [00:05, 1173.20 examples/s]6444 examples [00:05, 1171.95 examples/s]6562 examples [00:05, 1172.70 examples/s]6680 examples [00:05, 1172.19 examples/s]6798 examples [00:06, 1170.24 examples/s]6916 examples [00:06, 1170.88 examples/s]7034 examples [00:06, 1169.93 examples/s]7159 examples [00:06, 1191.72 examples/s]7279 examples [00:06, 1191.43 examples/s]7399 examples [00:06, 1180.90 examples/s]7518 examples [00:06, 1177.02 examples/s]7636 examples [00:06, 1170.56 examples/s]7754 examples [00:06, 1171.64 examples/s]7872 examples [00:06, 1172.37 examples/s]7990 examples [00:07, 1135.26 examples/s]8108 examples [00:07, 1148.03 examples/s]8226 examples [00:07, 1154.63 examples/s]8343 examples [00:07, 1157.99 examples/s]8462 examples [00:07, 1166.71 examples/s]8579 examples [00:07, 1154.99 examples/s]8695 examples [00:07, 1152.64 examples/s]8814 examples [00:07, 1161.64 examples/s]8931 examples [00:07, 1148.47 examples/s]9046 examples [00:07, 1145.89 examples/s]9163 examples [00:08, 1151.71 examples/s]9281 examples [00:08, 1159.96 examples/s]9398 examples [00:08, 1136.00 examples/s]9512 examples [00:08, 1079.03 examples/s]9621 examples [00:08, 1052.86 examples/s]9739 examples [00:08, 1086.36 examples/s]9855 examples [00:08, 1105.50 examples/s]9969 examples [00:08, 1115.24 examples/s]10081 examples [00:08, 1073.43 examples/s]10199 examples [00:08, 1102.97 examples/s]10310 examples [00:09, 1091.50 examples/s]10422 examples [00:09, 1099.47 examples/s]10537 examples [00:09, 1111.59 examples/s]10649 examples [00:09, 1106.86 examples/s]10766 examples [00:09, 1124.27 examples/s]10884 examples [00:09, 1139.26 examples/s]10999 examples [00:09, 1114.07 examples/s]11111 examples [00:09, 1078.32 examples/s]11220 examples [00:09, 1068.89 examples/s]11329 examples [00:10, 1074.46 examples/s]11437 examples [00:10, 1058.33 examples/s]11559 examples [00:10, 1100.34 examples/s]11680 examples [00:10, 1130.27 examples/s]11800 examples [00:10, 1149.98 examples/s]11916 examples [00:10, 1150.75 examples/s]12035 examples [00:10, 1160.83 examples/s]12158 examples [00:10, 1178.15 examples/s]12277 examples [00:10, 1162.92 examples/s]12394 examples [00:10, 1154.96 examples/s]12510 examples [00:11, 1146.27 examples/s]12629 examples [00:11, 1158.07 examples/s]12747 examples [00:11, 1163.53 examples/s]12864 examples [00:11, 1162.14 examples/s]12982 examples [00:11, 1165.78 examples/s]13099 examples [00:11, 1158.17 examples/s]13217 examples [00:11, 1162.52 examples/s]13337 examples [00:11, 1171.67 examples/s]13455 examples [00:11, 1164.41 examples/s]13573 examples [00:11, 1167.45 examples/s]13690 examples [00:12, 1166.19 examples/s]13811 examples [00:12, 1176.22 examples/s]13929 examples [00:12, 1176.07 examples/s]14047 examples [00:12, 1150.03 examples/s]14163 examples [00:12, 1150.91 examples/s]14279 examples [00:12, 1120.65 examples/s]14392 examples [00:12, 1050.69 examples/s]14506 examples [00:12, 1075.88 examples/s]14622 examples [00:12, 1098.30 examples/s]14739 examples [00:12, 1117.73 examples/s]14857 examples [00:13, 1131.54 examples/s]14971 examples [00:13, 1111.50 examples/s]15084 examples [00:13, 1115.75 examples/s]15196 examples [00:13, 1102.84 examples/s]15307 examples [00:13, 1007.31 examples/s]15417 examples [00:13, 1032.36 examples/s]15524 examples [00:13, 1041.34 examples/s]15631 examples [00:13, 1046.81 examples/s]15743 examples [00:13, 1066.20 examples/s]15851 examples [00:14, 1064.45 examples/s]15966 examples [00:14, 1087.61 examples/s]16081 examples [00:14, 1104.07 examples/s]16192 examples [00:14, 1102.03 examples/s]16308 examples [00:14, 1118.25 examples/s]16425 examples [00:14, 1131.17 examples/s]16543 examples [00:14, 1142.80 examples/s]16659 examples [00:14, 1146.35 examples/s]16774 examples [00:14, 1140.80 examples/s]16889 examples [00:14, 1141.70 examples/s]17005 examples [00:15, 1145.44 examples/s]17120 examples [00:15, 1133.46 examples/s]17239 examples [00:15, 1147.43 examples/s]17354 examples [00:15, 1145.85 examples/s]17470 examples [00:15, 1147.32 examples/s]17585 examples [00:15, 1135.57 examples/s]17703 examples [00:15, 1148.22 examples/s]17818 examples [00:15, 1097.70 examples/s]17931 examples [00:15, 1104.85 examples/s]18046 examples [00:15, 1114.17 examples/s]18169 examples [00:16, 1145.98 examples/s]18285 examples [00:16, 1145.71 examples/s]18400 examples [00:16, 1118.91 examples/s]18513 examples [00:16, 1090.37 examples/s]18627 examples [00:16, 1102.62 examples/s]18745 examples [00:16, 1122.22 examples/s]18858 examples [00:16, 1106.08 examples/s]18976 examples [00:16, 1125.84 examples/s]19089 examples [00:16, 1110.13 examples/s]19203 examples [00:17, 1118.76 examples/s]19320 examples [00:17, 1132.62 examples/s]19438 examples [00:17, 1143.89 examples/s]19555 examples [00:17, 1149.10 examples/s]19671 examples [00:17, 1118.72 examples/s]19784 examples [00:17, 1119.48 examples/s]19902 examples [00:17, 1136.08 examples/s]20016 examples [00:17, 1090.35 examples/s]20143 examples [00:17, 1137.34 examples/s]20260 examples [00:17, 1145.42 examples/s]20376 examples [00:18, 1141.08 examples/s]20496 examples [00:18, 1157.40 examples/s]20614 examples [00:18, 1161.80 examples/s]20736 examples [00:18, 1177.38 examples/s]20854 examples [00:18, 1177.22 examples/s]20973 examples [00:18, 1180.06 examples/s]21092 examples [00:18, 1156.76 examples/s]21208 examples [00:18, 1150.24 examples/s]21325 examples [00:18, 1153.31 examples/s]21441 examples [00:18, 1154.52 examples/s]21557 examples [00:19, 1119.14 examples/s]21674 examples [00:19, 1131.70 examples/s]21788 examples [00:19, 1104.59 examples/s]21904 examples [00:19, 1119.51 examples/s]22019 examples [00:19, 1126.52 examples/s]22133 examples [00:19, 1129.96 examples/s]22248 examples [00:19, 1135.75 examples/s]22365 examples [00:19, 1145.61 examples/s]22482 examples [00:19, 1151.15 examples/s]22598 examples [00:19, 1133.88 examples/s]22715 examples [00:20, 1141.60 examples/s]22830 examples [00:20, 1139.86 examples/s]22948 examples [00:20, 1148.35 examples/s]23067 examples [00:20, 1158.23 examples/s]23183 examples [00:20, 1155.78 examples/s]23302 examples [00:20, 1165.38 examples/s]23419 examples [00:20, 1148.59 examples/s]23537 examples [00:20, 1157.60 examples/s]23654 examples [00:20, 1159.95 examples/s]23771 examples [00:20, 1157.40 examples/s]23887 examples [00:21, 1152.95 examples/s]24004 examples [00:21, 1157.14 examples/s]24120 examples [00:21, 1129.76 examples/s]24234 examples [00:21, 1115.86 examples/s]24346 examples [00:21, 1095.50 examples/s]24461 examples [00:21, 1110.01 examples/s]24576 examples [00:21, 1119.04 examples/s]24695 examples [00:21, 1136.86 examples/s]24814 examples [00:21, 1150.10 examples/s]24930 examples [00:22, 1150.40 examples/s]25047 examples [00:22, 1155.20 examples/s]25163 examples [00:22, 1128.53 examples/s]25277 examples [00:22, 1090.16 examples/s]25392 examples [00:22, 1106.60 examples/s]25508 examples [00:22, 1121.57 examples/s]25621 examples [00:22, 1083.31 examples/s]25735 examples [00:22, 1099.06 examples/s]25854 examples [00:22, 1123.40 examples/s]25970 examples [00:22, 1133.29 examples/s]26085 examples [00:23, 1136.94 examples/s]26199 examples [00:23, 1131.01 examples/s]26317 examples [00:23, 1143.80 examples/s]26432 examples [00:23, 1124.69 examples/s]26548 examples [00:23, 1134.16 examples/s]26662 examples [00:23, 1134.16 examples/s]26780 examples [00:23, 1145.80 examples/s]26895 examples [00:23, 1138.21 examples/s]27009 examples [00:23, 1130.56 examples/s]27123 examples [00:23, 1117.52 examples/s]27239 examples [00:24, 1128.55 examples/s]27355 examples [00:24, 1137.45 examples/s]27470 examples [00:24, 1140.31 examples/s]27586 examples [00:24, 1143.73 examples/s]27703 examples [00:24, 1148.60 examples/s]27818 examples [00:24, 1145.36 examples/s]27933 examples [00:24, 1142.83 examples/s]28049 examples [00:24, 1146.35 examples/s]28164 examples [00:24, 1144.78 examples/s]28282 examples [00:24, 1152.44 examples/s]28398 examples [00:25, 1144.13 examples/s]28513 examples [00:25, 1136.74 examples/s]28631 examples [00:25, 1147.83 examples/s]28746 examples [00:25, 1108.10 examples/s]28858 examples [00:25, 1087.93 examples/s]28971 examples [00:25, 1098.74 examples/s]29087 examples [00:25, 1116.20 examples/s]29205 examples [00:25, 1133.89 examples/s]29324 examples [00:25, 1147.51 examples/s]29439 examples [00:26, 1141.08 examples/s]29554 examples [00:26, 1118.31 examples/s]29667 examples [00:26, 1097.03 examples/s]29780 examples [00:26, 1105.95 examples/s]29895 examples [00:26, 1116.57 examples/s]30007 examples [00:27, 222.19 examples/s] 30119 examples [00:27, 292.38 examples/s]30228 examples [00:28, 374.43 examples/s]30335 examples [00:28, 464.86 examples/s]30450 examples [00:28, 565.58 examples/s]30554 examples [00:28, 653.04 examples/s]30658 examples [00:28, 731.74 examples/s]30761 examples [00:28, 781.05 examples/s]30861 examples [00:28, 827.49 examples/s]30978 examples [00:28, 906.19 examples/s]31094 examples [00:28, 969.34 examples/s]31209 examples [00:28, 1016.59 examples/s]31323 examples [00:29, 1049.33 examples/s]31434 examples [00:29, 1061.21 examples/s]31549 examples [00:29, 1084.17 examples/s]31666 examples [00:29, 1108.10 examples/s]31780 examples [00:29, 1111.29 examples/s]31893 examples [00:29, 1110.46 examples/s]32006 examples [00:29, 1107.26 examples/s]32121 examples [00:29, 1118.83 examples/s]32237 examples [00:29, 1129.14 examples/s]32353 examples [00:30, 1136.36 examples/s]32470 examples [00:30, 1145.75 examples/s]32585 examples [00:30, 1128.70 examples/s]32699 examples [00:30, 1120.62 examples/s]32815 examples [00:30, 1132.03 examples/s]32929 examples [00:30, 1117.95 examples/s]33045 examples [00:30, 1127.61 examples/s]33158 examples [00:30, 1102.30 examples/s]33269 examples [00:30, 1069.39 examples/s]33383 examples [00:30, 1089.59 examples/s]33498 examples [00:31, 1106.40 examples/s]33615 examples [00:31, 1124.69 examples/s]33730 examples [00:31, 1131.62 examples/s]33846 examples [00:31, 1138.67 examples/s]33963 examples [00:31, 1145.49 examples/s]34078 examples [00:31, 1121.90 examples/s]34200 examples [00:31, 1147.51 examples/s]34316 examples [00:31, 1143.15 examples/s]34432 examples [00:31, 1148.05 examples/s]34548 examples [00:31, 1149.01 examples/s]34664 examples [00:32, 1141.23 examples/s]34779 examples [00:32, 1124.99 examples/s]34892 examples [00:32, 1122.79 examples/s]35009 examples [00:32, 1133.70 examples/s]35126 examples [00:32, 1142.01 examples/s]35241 examples [00:32, 1134.51 examples/s]35355 examples [00:32, 1135.13 examples/s]35469 examples [00:32, 1123.91 examples/s]35588 examples [00:32, 1142.27 examples/s]35703 examples [00:32, 1140.83 examples/s]35818 examples [00:33, 1141.21 examples/s]35933 examples [00:33, 1112.09 examples/s]36045 examples [00:33, 1107.16 examples/s]36163 examples [00:33, 1127.79 examples/s]36281 examples [00:33, 1141.74 examples/s]36396 examples [00:33, 1142.58 examples/s]36514 examples [00:33, 1152.36 examples/s]36630 examples [00:33, 1130.55 examples/s]36744 examples [00:33, 1116.12 examples/s]36856 examples [00:34, 1116.20 examples/s]36968 examples [00:34, 1102.72 examples/s]37083 examples [00:34, 1114.28 examples/s]37195 examples [00:34, 1104.59 examples/s]37311 examples [00:34, 1119.42 examples/s]37426 examples [00:34, 1127.25 examples/s]37540 examples [00:34, 1129.28 examples/s]37653 examples [00:34, 1122.98 examples/s]37766 examples [00:34, 1122.55 examples/s]37879 examples [00:34, 1120.83 examples/s]37995 examples [00:35, 1130.77 examples/s]38109 examples [00:35, 1070.59 examples/s]38226 examples [00:35, 1096.18 examples/s]38337 examples [00:35, 1097.57 examples/s]38451 examples [00:35, 1109.18 examples/s]38568 examples [00:35, 1124.64 examples/s]38683 examples [00:35, 1131.10 examples/s]38797 examples [00:35, 1115.50 examples/s]38909 examples [00:35, 1074.78 examples/s]39028 examples [00:35, 1105.14 examples/s]39148 examples [00:36, 1131.04 examples/s]39266 examples [00:36, 1143.24 examples/s]39384 examples [00:36, 1152.36 examples/s]39500 examples [00:36, 1148.30 examples/s]39616 examples [00:36, 1148.41 examples/s]39733 examples [00:36, 1153.99 examples/s]39850 examples [00:36, 1158.23 examples/s]39969 examples [00:36, 1165.07 examples/s]40086 examples [00:36, 1102.52 examples/s]40203 examples [00:36, 1120.26 examples/s]40321 examples [00:37, 1136.72 examples/s]40436 examples [00:37, 1140.02 examples/s]40554 examples [00:37, 1150.30 examples/s]40670 examples [00:37, 1146.03 examples/s]40786 examples [00:37, 1147.53 examples/s]40903 examples [00:37, 1152.72 examples/s]41019 examples [00:37, 1152.35 examples/s]41135 examples [00:37, 1152.71 examples/s]41251 examples [00:37, 1138.82 examples/s]41369 examples [00:37, 1147.98 examples/s]41484 examples [00:38, 1146.68 examples/s]41599 examples [00:38, 1106.16 examples/s]41715 examples [00:38, 1119.79 examples/s]41830 examples [00:38, 1127.22 examples/s]41948 examples [00:38, 1140.11 examples/s]42065 examples [00:38, 1147.15 examples/s]42180 examples [00:38, 1145.22 examples/s]42298 examples [00:38, 1152.78 examples/s]42415 examples [00:38, 1157.34 examples/s]42531 examples [00:39, 1157.85 examples/s]42647 examples [00:39, 1131.57 examples/s]42761 examples [00:39, 1131.87 examples/s]42879 examples [00:39, 1143.99 examples/s]42994 examples [00:39, 1143.48 examples/s]43109 examples [00:39, 1130.15 examples/s]43227 examples [00:39, 1142.78 examples/s]43342 examples [00:39, 1119.11 examples/s]43459 examples [00:39, 1132.46 examples/s]43573 examples [00:39, 1121.60 examples/s]43691 examples [00:40, 1136.02 examples/s]43805 examples [00:40, 1135.88 examples/s]43926 examples [00:40, 1155.38 examples/s]44042 examples [00:40, 1139.69 examples/s]44158 examples [00:40, 1145.34 examples/s]44274 examples [00:40, 1148.47 examples/s]44389 examples [00:40, 1115.37 examples/s]44507 examples [00:40, 1131.00 examples/s]44622 examples [00:40, 1136.60 examples/s]44736 examples [00:40, 1136.01 examples/s]44851 examples [00:41, 1138.79 examples/s]44965 examples [00:41, 1136.39 examples/s]45079 examples [00:41, 1035.05 examples/s]45185 examples [00:41, 1012.85 examples/s]45288 examples [00:41, 982.53 examples/s] 45390 examples [00:41, 992.78 examples/s]45491 examples [00:41, 950.52 examples/s]45588 examples [00:41, 902.49 examples/s]45687 examples [00:41, 925.32 examples/s]45790 examples [00:42, 954.38 examples/s]45887 examples [00:42, 955.75 examples/s]45997 examples [00:42, 994.18 examples/s]46110 examples [00:42, 1030.79 examples/s]46225 examples [00:42, 1062.45 examples/s]46343 examples [00:42, 1094.52 examples/s]46460 examples [00:42, 1114.63 examples/s]46573 examples [00:42, 1099.56 examples/s]46691 examples [00:42, 1120.13 examples/s]46804 examples [00:42, 1101.72 examples/s]46923 examples [00:43, 1125.81 examples/s]47040 examples [00:43, 1137.27 examples/s]47156 examples [00:43, 1142.61 examples/s]47271 examples [00:43, 1135.99 examples/s]47385 examples [00:43, 1112.49 examples/s]47502 examples [00:43, 1127.05 examples/s]47620 examples [00:43, 1142.01 examples/s]47736 examples [00:43, 1144.94 examples/s]47851 examples [00:43, 1143.47 examples/s]47966 examples [00:43, 1115.77 examples/s]48078 examples [00:44, 1108.09 examples/s]48189 examples [00:44, 1100.22 examples/s]48300 examples [00:44, 1072.53 examples/s]48409 examples [00:44, 1075.65 examples/s]48518 examples [00:44, 1077.20 examples/s]48634 examples [00:44, 1100.74 examples/s]48751 examples [00:44, 1119.03 examples/s]48869 examples [00:44, 1134.68 examples/s]48986 examples [00:44, 1144.18 examples/s]49101 examples [00:45, 1136.60 examples/s]49219 examples [00:45, 1147.22 examples/s]49338 examples [00:45, 1159.67 examples/s]49455 examples [00:45, 1158.97 examples/s]49574 examples [00:45, 1164.40 examples/s]49691 examples [00:45, 1144.25 examples/s]49806 examples [00:45, 1125.05 examples/s]49923 examples [00:45, 1135.42 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 19%|█▉        | 9573/50000 [00:00<00:00, 95723.70 examples/s] 48%|████▊     | 24048/50000 [00:00<00:00, 106548.65 examples/s] 78%|███████▊  | 38863/50000 [00:00<00:00, 116349.17 examples/s]                                                                0 examples [00:00, ? examples/s]98 examples [00:00, 974.68 examples/s]223 examples [00:00, 1041.72 examples/s]335 examples [00:00, 1063.99 examples/s]454 examples [00:00, 1097.25 examples/s]574 examples [00:00, 1125.08 examples/s]693 examples [00:00, 1141.27 examples/s]819 examples [00:00, 1171.01 examples/s]932 examples [00:00, 1155.51 examples/s]1051 examples [00:00, 1163.63 examples/s]1167 examples [00:01, 1161.11 examples/s]1282 examples [00:01, 1156.48 examples/s]1396 examples [00:01, 1146.40 examples/s]1510 examples [00:01, 1130.43 examples/s]1627 examples [00:01, 1139.48 examples/s]1749 examples [00:01, 1160.96 examples/s]1867 examples [00:01, 1166.41 examples/s]1988 examples [00:01, 1178.24 examples/s]2106 examples [00:01, 1175.73 examples/s]2224 examples [00:01, 1168.07 examples/s]2347 examples [00:02, 1183.65 examples/s]2466 examples [00:02, 1179.87 examples/s]2586 examples [00:02, 1185.10 examples/s]2705 examples [00:02, 1174.69 examples/s]2823 examples [00:02, 1171.42 examples/s]2941 examples [00:02, 1171.31 examples/s]3059 examples [00:02, 1120.67 examples/s]3173 examples [00:02, 1124.23 examples/s]3286 examples [00:02, 1108.04 examples/s]3398 examples [00:02, 1101.72 examples/s]3517 examples [00:03, 1124.93 examples/s]3630 examples [00:03, 1111.75 examples/s]3746 examples [00:03, 1124.55 examples/s]3859 examples [00:03, 1125.85 examples/s]3978 examples [00:03, 1142.53 examples/s]4097 examples [00:03, 1154.76 examples/s]4213 examples [00:03, 1143.11 examples/s]4329 examples [00:03, 1148.04 examples/s]4445 examples [00:03, 1149.80 examples/s]4565 examples [00:03, 1161.69 examples/s]4689 examples [00:04, 1181.73 examples/s]4808 examples [00:04, 1159.29 examples/s]4925 examples [00:04, 1089.51 examples/s]5035 examples [00:04, 1073.34 examples/s]5153 examples [00:04, 1103.13 examples/s]5273 examples [00:04, 1128.37 examples/s]5392 examples [00:04, 1146.00 examples/s]5509 examples [00:04, 1150.72 examples/s]5629 examples [00:04, 1163.29 examples/s]5748 examples [00:05, 1168.24 examples/s]5867 examples [00:05, 1173.72 examples/s]5985 examples [00:05, 1166.97 examples/s]6102 examples [00:05, 1123.76 examples/s]6215 examples [00:05, 1116.11 examples/s]6334 examples [00:05, 1135.90 examples/s]6453 examples [00:05, 1149.05 examples/s]6569 examples [00:05, 1131.89 examples/s]6685 examples [00:05, 1138.22 examples/s]6804 examples [00:05, 1152.70 examples/s]6923 examples [00:06, 1159.85 examples/s]7040 examples [00:06, 1160.09 examples/s]7159 examples [00:06, 1166.97 examples/s]7277 examples [00:06, 1169.32 examples/s]7397 examples [00:06, 1177.63 examples/s]7515 examples [00:06, 1172.23 examples/s]7638 examples [00:06, 1187.88 examples/s]7759 examples [00:06, 1191.59 examples/s]7879 examples [00:06, 1185.45 examples/s]7998 examples [00:06, 1186.21 examples/s]8117 examples [00:07, 1175.73 examples/s]8236 examples [00:07, 1178.00 examples/s]8356 examples [00:07, 1183.02 examples/s]8475 examples [00:07, 1161.39 examples/s]8592 examples [00:07, 1141.12 examples/s]8707 examples [00:07, 1138.21 examples/s]8826 examples [00:07, 1152.81 examples/s]8943 examples [00:07, 1156.83 examples/s]9059 examples [00:07, 1151.46 examples/s]9179 examples [00:07, 1163.60 examples/s]9296 examples [00:08, 1165.37 examples/s]9416 examples [00:08, 1173.66 examples/s]9535 examples [00:08, 1176.70 examples/s]9654 examples [00:08, 1179.96 examples/s]9773 examples [00:08, 1180.25 examples/s]9892 examples [00:08, 1157.37 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1VUFL4/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1VUFL4/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bf1f77620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bf1f77620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bf1f77620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b8c1b1da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b8c1d3f98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bf1f77620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bf1f77620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bf1f77620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b8c1d3b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bef360358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1b699e7378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1b699e7378> 

  function with postional parmater data_info <function split_train_valid at 0x7f1b699e7378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1b699e7488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1b699e7488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1b699e7488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=68096d24501ded77a1750d8382725d9e3791c43ca1a178639090c626ca962eaf
  Stored in directory: /tmp/pip-ephem-wheel-cache-04d21uio/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:17:35, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:08:12, 15.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:39:00, 22.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:27:48, 32.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.10M/862M [00:01<5:12:52, 45.8kB/s].vector_cache/glove.6B.zip:   1%|          | 6.39M/862M [00:01<3:38:17, 65.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.6M/862M [00:01<2:31:57, 93.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.9M/862M [00:01<1:46:05, 133kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.1M/862M [00:01<1:13:53, 190kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:01<51:38, 271kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.8M/862M [00:01<36:00, 386kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.9M/862M [00:01<25:14, 548kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.3M/862M [00:02<17:37, 780kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.4M/862M [00:02<12:26, 1.10MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.2M/862M [00:02<08:44, 1.56MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.9M/862M [00:02<06:12, 2.18MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<04:42, 2.86MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<05:12, 2.58MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<07:29, 1.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<06:14, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:04<04:37, 2.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<06:31, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<06:01, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:06<04:34, 2.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<06:10, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<07:01, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<05:29, 2.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:08<04:01, 3.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<08:39, 1.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<07:27, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:10<05:33, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<06:55, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<06:15, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<04:40, 2.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:12<03:26, 3.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<1:40:49, 130kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<1:11:43, 182kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<50:44, 258kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:14<35:39, 366kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<30:09, 432kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<22:51, 569kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:16<16:22, 794kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:16<11:39, 1.11MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<13:59, 925kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<11:28, 1.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<08:21, 1.55MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.0M/862M [00:18<06:04, 2.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:18<05:00, 2.57MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<15:56, 808kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<13:33, 950kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<10:04, 1.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:20<07:11, 1.78MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<13:58, 917kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<12:11, 1.05MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<09:01, 1.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<06:34, 1.94MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<08:22, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<08:16, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<06:22, 2.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<04:38, 2.74MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:21, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:55, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:49, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<04:55, 2.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<14:08, 892kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:16, 1.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<09:10, 1.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<06:31, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<56:52, 221kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<42:09, 297kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<29:58, 418kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<21:12, 590kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<18:31, 673kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<15:17, 815kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<11:13, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<08:09, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:28, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:13, 1.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:59, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<05:46, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:02, 1.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:01, 1.37MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:00, 1.76MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:15, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:11, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:40, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:12, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:38<04:31, 2.70MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<08:46, 1.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:15, 1.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:14, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:40<05:14, 2.32MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:02, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:00, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:38, 1.59MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:31, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:53, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:27, 1.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:33, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:52, 2.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:07, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:32, 1.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:03, 1.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:24, 2.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:58, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:10, 1.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:02, 1.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:16, 2.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:10, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:21, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:54, 2.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:16, 2.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:11, 1.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:28, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:37, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:51, 2.42MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:43, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:27, 1.57MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:47, 2.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:19, 2.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:51, 1.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:51, 1.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:26, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:56, 2.94MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<09:01, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<08:54, 1.30MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:49, 1.69MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:55, 2.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:58, 1.44MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<08:25, 1.37MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:37, 1.73MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:48, 2.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:23, 1.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:41, 1.32MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<07:34, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:37, 2.03MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:16, 1.81MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:55, 1.64MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:29, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:00, 2.82MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<10:48, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<10:20, 1.09MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:49, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<05:38, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<07:54, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:10, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:15, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:35, 2.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:11, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:49, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:20, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:53, 2.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:50, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:29, 1.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:51, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:15, 2.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<08:45, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<08:43, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:40, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:14<04:49, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<10:34, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<09:55, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:29, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<05:21, 2.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<09:28, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<09:25, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:16, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<05:14, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<09:19, 1.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<09:01, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:49, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:54, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:15, 1.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<08:25, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:27, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:38, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:50, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:13, 1.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:20, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:34, 2.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:29, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<07:59, 1.33MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:17, 1.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:31, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<08:24, 1.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<08:21, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:21, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:36, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:48, 1.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<07:05, 1.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:30, 1.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<04:00, 2.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:28, 1.61MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:49, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:16, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:50, 2.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:14, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:44, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:15, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<03:49, 2.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:21, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:42, 1.53MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:15, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<03:48, 2.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<09:37, 1.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<08:58, 1.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:50, 1.48MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<04:54, 2.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<10:21, 977kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<09:28, 1.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<07:05, 1.42MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:06, 1.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:07, 1.41MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:34, 1.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:54, 1.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<04:17, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:15, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:11, 1.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<06:14, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<04:30, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:50, 1.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:11, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:32, 1.78MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:00, 2.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:55, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:55, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<05:17, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:51, 2.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:55, 1.65MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:38, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:15, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:47, 2.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<07:22, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:30, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:45, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:12, 2.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:32, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:06, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:44, 2.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<03:26, 2.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<06:47, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<06:41, 1.43MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<05:09, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:42, 2.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<16:42, 568kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<13:52, 683kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<10:15, 924kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<07:17, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<10:53, 864kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<09:48, 961kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<07:20, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<05:17, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:23, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:38, 1.41MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:05, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:42, 2.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:38, 1.64MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:07, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:46, 1.94MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<03:28, 2.66MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<07:41, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<07:38, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:53, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<04:15, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<08:40, 1.05MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<08:19, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:18, 1.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<04:30, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<07:26, 1.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<07:24, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:39, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:07, 2.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:22, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:15, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:04, 2.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<02:59, 2.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<24:05, 371kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<19:01, 470kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<13:49, 645kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<09:45, 909kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<13:12, 672kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<11:16, 786kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<08:18, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<05:55, 1.49MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<08:05, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<07:24, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:42, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<04:06, 2.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:00, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:07, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:41, 1.86MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<03:24, 2.55MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:43, 1.51MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<06:00, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:38, 1.86MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<03:21, 2.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:57, 1.44MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:08, 1.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:43, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:24, 2.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:49, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:46, 1.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:14, 1.62MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<03:46, 2.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<09:21, 903kB/s] .vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<08:30, 993kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<06:25, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<04:35, 1.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<09:51, 849kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<08:32, 980kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<06:23, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<04:33, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<20:08, 413kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<16:01, 519kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<11:41, 710kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<08:15, 999kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<12:21, 667kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<10:14, 804kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<07:33, 1.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<05:22, 1.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<25:01, 326kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<19:24, 421kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<14:02, 581kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<09:53, 821kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<13:19, 608kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<11:06, 729kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<08:12, 985kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<05:49, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<11:52, 676kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<10:07, 793kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<07:26, 1.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<05:17, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<07:51, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:15, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:27, 1.46MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<03:53, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<07:31, 1.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:43, 1.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:04, 1.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<03:38, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<37:03, 211kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<27:50, 281kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<19:51, 393kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<13:56, 557kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<12:56, 599kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<10:31, 736kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<07:42, 1.00MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:47<05:27, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<33:32, 229kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<25:14, 304kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<18:05, 424kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<12:41, 601kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<14:57, 509kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<11:50, 643kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<08:36, 882kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<06:05, 1.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<58:27, 129kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:53<42:19, 178kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<29:52, 252kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<20:53, 358kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<19:52, 376kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<15:39, 478kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<11:22, 656kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<08:00, 927kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<11:30, 644kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<09:22, 790kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<06:53, 1.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<04:52, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<1:03:58, 115kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<46:05, 159kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<32:30, 225kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<22:42, 320kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<43:56, 166kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<32:25, 224kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<23:01, 315kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<16:07, 448kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<15:27, 466kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<12:28, 577kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<09:04, 792kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<06:24, 1.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<07:42, 925kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<06:53, 1.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:12, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<03:43, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<09:57, 709kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<08:14, 857kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<06:04, 1.16MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<04:19, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<1:00:18, 116kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<43:48, 160kB/s]  .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<31:01, 225kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<21:39, 320kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<20:39, 335kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<16:03, 431kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<11:37, 595kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<08:10, 841kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<11:05, 618kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<09:16, 739kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:51, 998kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<04:51, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<09:47, 693kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<08:04, 841kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<05:56, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<04:13, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<57:59, 116kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<42:08, 159kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<29:49, 225kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<20:49, 320kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<19:51, 335kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:18<15:22, 432kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<11:03, 600kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<07:48, 846kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<07:46, 846kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<06:58, 943kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:11, 1.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<03:42, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:06, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<04:45, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:36, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<02:35, 2.48MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<3:06:26, 34.5kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<2:11:35, 48.9kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<1:32:11, 69.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<1:04:07, 99.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<1:30:37, 70.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<1:04:52, 98.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<45:39, 139kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<31:51, 198kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<25:00, 252kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<18:37, 338kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<13:17, 473kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<10:17, 606kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<08:34, 727kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<06:17, 987kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<04:26, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<08:06, 760kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<07:01, 876kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<05:12, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:42, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<05:37, 1.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<05:17, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:01, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:52, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<08:10, 737kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<06:46, 889kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<04:57, 1.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<03:31, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<06:34, 906kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<07:01, 849kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:25, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:56, 1.50MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:04, 1.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:09, 1.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:13, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<02:19, 2.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<07:47, 747kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<06:44, 863kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:58, 1.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<03:32, 1.63MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:17, 1.09MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:39, 1.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:29, 1.64MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:24, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:54, 1.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:05, 1.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<03:01, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:13, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:26, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:39, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<01:56, 2.88MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:31, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:23, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:35, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:45, 1.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:08, 1.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:27, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<01:46, 3.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:18, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:51, 1.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:52, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<02:03, 2.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<17:18, 308kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<13:15, 402kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<09:32, 558kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<06:40, 789kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<20:33, 256kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<15:12, 346kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<10:48, 485kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<08:26, 616kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<08:00, 649kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<06:08, 845kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<04:24, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:11, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:41, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:45, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:50, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:58, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:19, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:25, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:42, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:07, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:17, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:33, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:01, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:11, 2.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:30, 1.39MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:56, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:10, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:35, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:11, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:33, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:51, 2.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:11, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:29, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:43, 1.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:58, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:51, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:14, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:35, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:52, 2.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:13, 1.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:29, 1.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:42, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:57, 2.31MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:17, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:27, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:43, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:57, 2.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:15, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:24, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:37, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:53, 2.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:13, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:22, 1.29MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:38, 1.65MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:53, 2.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:15, 1.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:23, 1.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:37, 1.63MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:53, 2.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:18, 1.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:23, 1.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:39, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:54, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:07, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:11, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:26, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:45, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:44, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:57, 1.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:19, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:40, 2.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:58, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:06, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:25, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:44, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<03:09, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<03:10, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:25, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:44, 2.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:46, 1.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:53, 1.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:15, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:37, 2.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:01, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<03:08, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:26, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:45, 2.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:50, 1.32MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:53, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:12, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:35, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:45, 1.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:52, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:11, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:34, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:17, 1.58MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:34, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<02:02, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:27, 2.42MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:29, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:33, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:57, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:24, 2.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:21, 1.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:29, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:56, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:50<01:23, 2.44MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:38, 1.29MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:42, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:03, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:29, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:59, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:12, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:43, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:15, 2.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:47, 1.82MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:02, 1.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:35, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:09, 2.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:26, 1.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:34, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:58, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:26, 2.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:46, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:00, 1.56MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:33, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:06, 2.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<02:12, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<02:17, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:46, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:16, 2.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:24, 1.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:22, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:48, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:17, 2.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:18, 1.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:17, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:44, 1.66MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:15, 2.29MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:47, 1.58MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:56, 1.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:31, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:05, 2.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:11, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:14, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:44, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:14, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:05, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:06, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:36, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:09, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:43, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:50, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:26, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:02, 2.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:02, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:01, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:33, 1.63MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:06, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<02:01, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:59, 1.25MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:32, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:05, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:03, 1.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:59, 1.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:30, 1.60MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:05, 2.19MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:00, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:56, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:28, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:02, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:52, 1.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:49, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:22, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:58, 2.28MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:29, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:31, 1.45MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:10, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:50, 2.59MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:28, 1.46MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:34, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:14, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:52, 2.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:33, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:35, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:12, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:51, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:16, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:24, 1.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:06, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:47, 2.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:27, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:25, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:04, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<00:46, 2.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:19, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:19, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:01, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:43, 2.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:49, 987kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:39, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:14, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<00:52, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<02:05, 829kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:54, 910kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:25, 1.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<01:00, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:31, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:27, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:05, 1.51MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:41<00:46, 2.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:02, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:01, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:47, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:33, 2.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<03:18, 462kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:40, 569kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<01:57, 774kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<01:21, 1.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:49, 800kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:32, 942kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:07, 1.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:46, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:41, 822kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:31, 912kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:07, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:47, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:07, 1.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:05, 1.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:49, 1.57MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:51<00:34, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:20, 928kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:12, 1.03MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:54, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:37, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:21, 867kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:14, 952kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:54, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:38, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:50, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:50, 1.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:39, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:27, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:05, 949kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:59, 1.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:44, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:30, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:41, 1.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:41, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:31, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:01<00:21, 2.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:00, 891kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:55, 972kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:41, 1.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<00:28, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:54, 909kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:49, 999kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:36, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:25, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:39, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:37, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:28, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:19, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:44, 931kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:40, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:29, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:20, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:34, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:32, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:24, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:16, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:36, 904kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:33, 995kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:24, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:16, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:32, 893kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:29, 975kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:21, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:14, 1.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:19, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:19, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:14, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:09, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:15, 1.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:15, 1.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:07, 2.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:12, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:12, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:05, 2.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:09, 869kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:07, 1.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:02, 1.89MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:13, 306kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:09, 404kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:05, 564kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 798kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<23:30:18,  4.73it/s]  0%|          | 953/400000 [00:00<16:25:04,  6.75it/s]  0%|          | 1914/400000 [00:00<11:28:06,  9.64it/s]  1%|          | 2872/400000 [00:00<8:00:43, 13.77it/s]   1%|          | 3853/400000 [00:00<5:35:52, 19.66it/s]  1%|          | 4817/400000 [00:00<3:54:44, 28.06it/s]  1%|▏         | 5791/400000 [00:00<2:44:07, 40.03it/s]  2%|▏         | 6649/400000 [00:00<1:54:51, 57.08it/s]  2%|▏         | 7672/400000 [00:01<1:20:23, 81.34it/s]  2%|▏         | 8634/400000 [00:01<56:20, 115.78it/s]   2%|▏         | 9589/400000 [00:01<39:32, 164.55it/s]  3%|▎         | 10522/400000 [00:01<27:49, 233.27it/s]  3%|▎         | 11449/400000 [00:01<19:38, 329.59it/s]  3%|▎         | 12467/400000 [00:01<13:54, 464.39it/s]  3%|▎         | 13466/400000 [00:01<09:54, 650.45it/s]  4%|▎         | 14451/400000 [00:01<07:06, 903.63it/s]  4%|▍         | 15442/400000 [00:01<05:09, 1241.21it/s]  4%|▍         | 16427/400000 [00:01<03:48, 1682.22it/s]  4%|▍         | 17441/400000 [00:02<02:50, 2243.55it/s]  5%|▍         | 18427/400000 [00:02<02:10, 2916.40it/s]  5%|▍         | 19409/400000 [00:02<01:43, 3667.16it/s]  5%|▌         | 20396/400000 [00:02<01:24, 4519.08it/s]  5%|▌         | 21366/400000 [00:02<01:10, 5377.46it/s]  6%|▌         | 22361/400000 [00:02<01:00, 6237.36it/s]  6%|▌         | 23338/400000 [00:02<00:54, 6925.02it/s]  6%|▌         | 24339/400000 [00:02<00:49, 7629.86it/s]  6%|▋         | 25315/400000 [00:02<00:45, 8162.53it/s]  7%|▋         | 26290/400000 [00:02<00:43, 8552.82it/s]  7%|▋         | 27273/400000 [00:03<00:41, 8898.43it/s]  7%|▋         | 28249/400000 [00:03<00:40, 9138.63it/s]  7%|▋         | 29265/400000 [00:03<00:39, 9422.34it/s]  8%|▊         | 30253/400000 [00:03<00:39, 9360.83it/s]  8%|▊         | 31221/400000 [00:03<00:40, 9217.70it/s]  8%|▊         | 32166/400000 [00:03<00:39, 9248.38it/s]  8%|▊         | 33170/400000 [00:03<00:38, 9471.20it/s]  9%|▊         | 34145/400000 [00:03<00:38, 9553.00it/s]  9%|▉         | 35110/400000 [00:03<00:38, 9527.34it/s]  9%|▉         | 36069/400000 [00:03<00:39, 9299.85it/s]  9%|▉         | 37029/400000 [00:04<00:38, 9387.83it/s]  9%|▉         | 37972/400000 [00:04<00:39, 9126.26it/s] 10%|▉         | 39000/400000 [00:04<00:38, 9444.05it/s] 10%|▉         | 39992/400000 [00:04<00:37, 9581.04it/s] 10%|█         | 40958/400000 [00:04<00:37, 9604.53it/s] 10%|█         | 41973/400000 [00:04<00:36, 9759.50it/s] 11%|█         | 43014/400000 [00:04<00:35, 9943.49it/s] 11%|█         | 44028/400000 [00:04<00:35, 9998.92it/s] 11%|█▏        | 45033/400000 [00:04<00:35, 10012.68it/s] 12%|█▏        | 46036/400000 [00:04<00:36, 9804.62it/s]  12%|█▏        | 47038/400000 [00:05<00:35, 9867.58it/s] 12%|█▏        | 48043/400000 [00:05<00:35, 9920.78it/s] 12%|█▏        | 49037/400000 [00:05<00:35, 9801.39it/s] 13%|█▎        | 50019/400000 [00:05<00:35, 9794.33it/s] 13%|█▎        | 51000/400000 [00:05<00:35, 9759.50it/s] 13%|█▎        | 51977/400000 [00:05<00:36, 9530.36it/s] 13%|█▎        | 52953/400000 [00:05<00:36, 9596.53it/s] 13%|█▎        | 53931/400000 [00:05<00:35, 9650.84it/s] 14%|█▎        | 54900/400000 [00:05<00:35, 9660.85it/s] 14%|█▍        | 55867/400000 [00:06<00:36, 9507.09it/s] 14%|█▍        | 56819/400000 [00:06<00:36, 9463.44it/s] 14%|█▍        | 57772/400000 [00:06<00:36, 9482.13it/s] 15%|█▍        | 58802/400000 [00:06<00:35, 9712.63it/s] 15%|█▍        | 59776/400000 [00:06<00:35, 9573.45it/s] 15%|█▌        | 60736/400000 [00:06<00:36, 9195.28it/s] 15%|█▌        | 61691/400000 [00:06<00:36, 9296.90it/s] 16%|█▌        | 62649/400000 [00:06<00:35, 9377.03it/s] 16%|█▌        | 63654/400000 [00:06<00:35, 9568.73it/s] 16%|█▌        | 64647/400000 [00:06<00:34, 9671.95it/s] 16%|█▋        | 65622/400000 [00:07<00:34, 9693.61it/s] 17%|█▋        | 66608/400000 [00:07<00:34, 9742.33it/s] 17%|█▋        | 67584/400000 [00:07<00:34, 9616.03it/s] 17%|█▋        | 68614/400000 [00:07<00:33, 9809.37it/s] 17%|█▋        | 69608/400000 [00:07<00:33, 9847.15it/s] 18%|█▊        | 70594/400000 [00:07<00:33, 9779.91it/s] 18%|█▊        | 71650/400000 [00:07<00:32, 9999.95it/s] 18%|█▊        | 72652/400000 [00:07<00:33, 9862.91it/s] 18%|█▊        | 73642/400000 [00:07<00:33, 9873.81it/s] 19%|█▊        | 74648/400000 [00:07<00:32, 9928.52it/s] 19%|█▉        | 75649/400000 [00:08<00:32, 9950.72it/s] 19%|█▉        | 76673/400000 [00:08<00:32, 10034.88it/s] 19%|█▉        | 77678/400000 [00:08<00:32, 10023.88it/s] 20%|█▉        | 78704/400000 [00:08<00:31, 10092.03it/s] 20%|█▉        | 79714/400000 [00:08<00:31, 10064.28it/s] 20%|██        | 80721/400000 [00:08<00:31, 10023.85it/s] 20%|██        | 81738/400000 [00:08<00:31, 10065.12it/s] 21%|██        | 82745/400000 [00:08<00:31, 9989.58it/s]  21%|██        | 83745/400000 [00:08<00:32, 9683.13it/s] 21%|██        | 84733/400000 [00:08<00:32, 9739.61it/s] 21%|██▏       | 85709/400000 [00:09<00:32, 9732.33it/s] 22%|██▏       | 86684/400000 [00:09<00:33, 9485.08it/s] 22%|██▏       | 87642/400000 [00:09<00:32, 9513.14it/s] 22%|██▏       | 88601/400000 [00:09<00:32, 9535.49it/s] 22%|██▏       | 89608/400000 [00:09<00:32, 9689.18it/s] 23%|██▎       | 90579/400000 [00:09<00:32, 9625.68it/s] 23%|██▎       | 91571/400000 [00:09<00:31, 9710.69it/s] 23%|██▎       | 92543/400000 [00:09<00:32, 9582.51it/s] 23%|██▎       | 93544/400000 [00:09<00:31, 9706.01it/s] 24%|██▎       | 94544/400000 [00:09<00:31, 9791.61it/s] 24%|██▍       | 95525/400000 [00:10<00:31, 9644.48it/s] 24%|██▍       | 96491/400000 [00:10<00:32, 9388.60it/s] 24%|██▍       | 97433/400000 [00:10<00:32, 9283.86it/s] 25%|██▍       | 98382/400000 [00:10<00:32, 9344.01it/s] 25%|██▍       | 99318/400000 [00:10<00:32, 9300.12it/s] 25%|██▌       | 100259/400000 [00:10<00:32, 9332.35it/s] 25%|██▌       | 101249/400000 [00:10<00:31, 9494.01it/s] 26%|██▌       | 102259/400000 [00:10<00:30, 9665.56it/s] 26%|██▌       | 103236/400000 [00:10<00:30, 9695.48it/s] 26%|██▌       | 104250/400000 [00:10<00:30, 9821.38it/s] 26%|██▋       | 105234/400000 [00:11<00:30, 9823.36it/s] 27%|██▋       | 106224/400000 [00:11<00:29, 9843.96it/s] 27%|██▋       | 107242/400000 [00:11<00:29, 9939.99it/s] 27%|██▋       | 108265/400000 [00:11<00:29, 10022.45it/s] 27%|██▋       | 109300/400000 [00:11<00:28, 10115.50it/s] 28%|██▊       | 110313/400000 [00:11<00:28, 10021.64it/s] 28%|██▊       | 111316/400000 [00:11<00:29, 9936.95it/s]  28%|██▊       | 112311/400000 [00:11<00:29, 9862.72it/s] 28%|██▊       | 113298/400000 [00:11<00:29, 9850.85it/s] 29%|██▊       | 114284/400000 [00:12<00:29, 9733.68it/s] 29%|██▉       | 115258/400000 [00:12<00:29, 9728.94it/s] 29%|██▉       | 116265/400000 [00:12<00:28, 9826.17it/s] 29%|██▉       | 117256/400000 [00:12<00:28, 9849.79it/s] 30%|██▉       | 118242/400000 [00:12<00:28, 9786.60it/s] 30%|██▉       | 119222/400000 [00:12<00:29, 9642.07it/s] 30%|███       | 120187/400000 [00:12<00:29, 9606.88it/s] 30%|███       | 121217/400000 [00:12<00:28, 9803.54it/s] 31%|███       | 122199/400000 [00:12<00:28, 9732.71it/s] 31%|███       | 123185/400000 [00:12<00:28, 9768.10it/s] 31%|███       | 124222/400000 [00:13<00:27, 9940.32it/s] 31%|███▏      | 125231/400000 [00:13<00:27, 9982.54it/s] 32%|███▏      | 126231/400000 [00:13<00:27, 9873.81it/s] 32%|███▏      | 127220/400000 [00:13<00:27, 9849.92it/s] 32%|███▏      | 128206/400000 [00:13<00:28, 9466.59it/s] 32%|███▏      | 129157/400000 [00:13<00:30, 8896.30it/s] 33%|███▎      | 130057/400000 [00:13<00:30, 8880.70it/s] 33%|███▎      | 130960/400000 [00:13<00:30, 8922.48it/s] 33%|███▎      | 131910/400000 [00:13<00:29, 9086.64it/s] 33%|███▎      | 132938/400000 [00:13<00:28, 9414.01it/s] 33%|███▎      | 133957/400000 [00:14<00:27, 9632.77it/s] 34%|███▎      | 134926/400000 [00:14<00:27, 9618.18it/s] 34%|███▍      | 135906/400000 [00:14<00:27, 9670.11it/s] 34%|███▍      | 136902/400000 [00:14<00:26, 9753.16it/s] 34%|███▍      | 137880/400000 [00:14<00:27, 9583.55it/s] 35%|███▍      | 138883/400000 [00:14<00:26, 9713.25it/s] 35%|███▍      | 139857/400000 [00:14<00:27, 9610.66it/s] 35%|███▌      | 140847/400000 [00:14<00:26, 9694.14it/s] 35%|███▌      | 141819/400000 [00:14<00:26, 9699.78it/s] 36%|███▌      | 142790/400000 [00:14<00:26, 9624.26it/s] 36%|███▌      | 143778/400000 [00:15<00:26, 9699.35it/s] 36%|███▌      | 144749/400000 [00:15<00:26, 9616.06it/s] 36%|███▋      | 145712/400000 [00:15<00:26, 9472.64it/s] 37%|███▋      | 146684/400000 [00:15<00:26, 9542.40it/s] 37%|███▋      | 147681/400000 [00:15<00:26, 9666.49it/s] 37%|███▋      | 148701/400000 [00:15<00:25, 9818.27it/s] 37%|███▋      | 149685/400000 [00:15<00:25, 9718.26it/s] 38%|███▊      | 150658/400000 [00:15<00:25, 9626.95it/s] 38%|███▊      | 151622/400000 [00:15<00:26, 9355.39it/s] 38%|███▊      | 152567/400000 [00:15<00:26, 9382.03it/s] 38%|███▊      | 153558/400000 [00:16<00:25, 9534.12it/s] 39%|███▊      | 154559/400000 [00:16<00:25, 9670.69it/s] 39%|███▉      | 155538/400000 [00:16<00:25, 9703.42it/s] 39%|███▉      | 156567/400000 [00:16<00:24, 9870.67it/s] 39%|███▉      | 157674/400000 [00:16<00:23, 10201.06it/s] 40%|███▉      | 158698/400000 [00:16<00:23, 10105.91it/s] 40%|███▉      | 159712/400000 [00:16<00:24, 9851.63it/s]  40%|████      | 160701/400000 [00:16<00:24, 9661.99it/s] 40%|████      | 161671/400000 [00:16<00:25, 9481.80it/s] 41%|████      | 162623/400000 [00:17<00:25, 9419.27it/s] 41%|████      | 163570/400000 [00:17<00:25, 9433.71it/s] 41%|████      | 164540/400000 [00:17<00:24, 9509.75it/s] 41%|████▏     | 165525/400000 [00:17<00:24, 9607.39it/s] 42%|████▏     | 166549/400000 [00:17<00:23, 9787.48it/s] 42%|████▏     | 167574/400000 [00:17<00:23, 9919.28it/s] 42%|████▏     | 168568/400000 [00:17<00:23, 9920.03it/s] 42%|████▏     | 169562/400000 [00:17<00:23, 9917.85it/s] 43%|████▎     | 170555/400000 [00:17<00:23, 9858.13it/s] 43%|████▎     | 171577/400000 [00:17<00:22, 9962.96it/s] 43%|████▎     | 172578/400000 [00:18<00:22, 9974.84it/s] 43%|████▎     | 173576/400000 [00:18<00:23, 9740.13it/s] 44%|████▎     | 174552/400000 [00:18<00:23, 9704.72it/s] 44%|████▍     | 175562/400000 [00:18<00:22, 9818.04it/s] 44%|████▍     | 176545/400000 [00:18<00:23, 9527.91it/s] 44%|████▍     | 177567/400000 [00:18<00:22, 9725.10it/s] 45%|████▍     | 178543/400000 [00:18<00:22, 9632.88it/s] 45%|████▍     | 179523/400000 [00:18<00:22, 9680.23it/s] 45%|████▌     | 180516/400000 [00:18<00:22, 9751.17it/s] 45%|████▌     | 181517/400000 [00:18<00:22, 9826.51it/s] 46%|████▌     | 182501/400000 [00:19<00:22, 9789.68it/s] 46%|████▌     | 183481/400000 [00:19<00:22, 9472.09it/s] 46%|████▌     | 184441/400000 [00:19<00:22, 9507.91it/s] 46%|████▋     | 185411/400000 [00:19<00:22, 9563.03it/s] 47%|████▋     | 186409/400000 [00:19<00:22, 9682.36it/s] 47%|████▋     | 187415/400000 [00:19<00:21, 9790.66it/s] 47%|████▋     | 188396/400000 [00:19<00:21, 9672.97it/s] 47%|████▋     | 189365/400000 [00:19<00:21, 9633.39it/s] 48%|████▊     | 190342/400000 [00:19<00:21, 9672.81it/s] 48%|████▊     | 191310/400000 [00:19<00:21, 9536.74it/s] 48%|████▊     | 192265/400000 [00:20<00:22, 9257.39it/s] 48%|████▊     | 193256/400000 [00:20<00:21, 9442.89it/s] 49%|████▊     | 194225/400000 [00:20<00:21, 9513.63it/s] 49%|████▉     | 195208/400000 [00:20<00:21, 9604.74it/s] 49%|████▉     | 196193/400000 [00:20<00:21, 9675.12it/s] 49%|████▉     | 197195/400000 [00:20<00:20, 9775.12it/s] 50%|████▉     | 198174/400000 [00:20<00:20, 9768.51it/s] 50%|████▉     | 199171/400000 [00:20<00:20, 9826.33it/s] 50%|█████     | 200155/400000 [00:20<00:20, 9828.98it/s] 50%|█████     | 201162/400000 [00:20<00:20, 9897.70it/s] 51%|█████     | 202168/400000 [00:21<00:19, 9943.85it/s] 51%|█████     | 203181/400000 [00:21<00:19, 9997.26it/s] 51%|█████     | 204183/400000 [00:21<00:19, 10002.30it/s] 51%|█████▏    | 205184/400000 [00:21<00:19, 9789.44it/s]  52%|█████▏    | 206165/400000 [00:21<00:20, 9487.13it/s] 52%|█████▏    | 207147/400000 [00:21<00:20, 9583.27it/s] 52%|█████▏    | 208108/400000 [00:21<00:20, 9467.62it/s] 52%|█████▏    | 209097/400000 [00:21<00:19, 9588.96it/s] 53%|█████▎    | 210089/400000 [00:21<00:19, 9683.85it/s] 53%|█████▎    | 211079/400000 [00:22<00:19, 9746.69it/s] 53%|█████▎    | 212055/400000 [00:22<00:19, 9459.88it/s] 53%|█████▎    | 213024/400000 [00:22<00:19, 9524.81it/s] 54%|█████▎    | 214002/400000 [00:22<00:19, 9599.43it/s] 54%|█████▍    | 215012/400000 [00:22<00:18, 9742.75it/s] 54%|█████▍    | 215988/400000 [00:22<00:18, 9696.60it/s] 54%|█████▍    | 216959/400000 [00:22<00:18, 9676.94it/s] 54%|█████▍    | 217928/400000 [00:22<00:19, 9499.60it/s] 55%|█████▍    | 218913/400000 [00:22<00:18, 9599.86it/s] 55%|█████▍    | 219904/400000 [00:22<00:18, 9688.76it/s] 55%|█████▌    | 220913/400000 [00:23<00:18, 9803.76it/s] 55%|█████▌    | 221895/400000 [00:23<00:18, 9784.52it/s] 56%|█████▌    | 222875/400000 [00:23<00:18, 9683.72it/s] 56%|█████▌    | 223845/400000 [00:23<00:18, 9494.90it/s] 56%|█████▌    | 224796/400000 [00:23<00:18, 9493.77it/s] 56%|█████▋    | 225788/400000 [00:23<00:18, 9616.83it/s] 57%|█████▋    | 226772/400000 [00:23<00:17, 9681.81it/s] 57%|█████▋    | 227757/400000 [00:23<00:17, 9731.07it/s] 57%|█████▋    | 228753/400000 [00:23<00:17, 9798.59it/s] 57%|█████▋    | 229749/400000 [00:23<00:17, 9845.86it/s] 58%|█████▊    | 230755/400000 [00:24<00:17, 9906.45it/s] 58%|█████▊    | 231747/400000 [00:24<00:17, 9589.35it/s] 58%|█████▊    | 232709/400000 [00:24<00:17, 9582.14it/s] 58%|█████▊    | 233669/400000 [00:24<00:17, 9550.14it/s] 59%|█████▊    | 234654/400000 [00:24<00:17, 9637.35it/s] 59%|█████▉    | 235682/400000 [00:24<00:16, 9818.88it/s] 59%|█████▉    | 236666/400000 [00:24<00:16, 9773.34it/s] 59%|█████▉    | 237645/400000 [00:24<00:16, 9766.37it/s] 60%|█████▉    | 238623/400000 [00:24<00:16, 9615.79it/s] 60%|█████▉    | 239608/400000 [00:24<00:16, 9684.21it/s] 60%|██████    | 240590/400000 [00:25<00:16, 9722.88it/s] 60%|██████    | 241563/400000 [00:25<00:16, 9499.06it/s] 61%|██████    | 242515/400000 [00:25<00:17, 9122.51it/s] 61%|██████    | 243503/400000 [00:25<00:16, 9337.05it/s] 61%|██████    | 244533/400000 [00:25<00:16, 9604.81it/s] 61%|██████▏   | 245557/400000 [00:25<00:15, 9785.76it/s] 62%|██████▏   | 246576/400000 [00:25<00:15, 9902.36it/s] 62%|██████▏   | 247570/400000 [00:25<00:15, 9775.75it/s] 62%|██████▏   | 248551/400000 [00:25<00:15, 9745.57it/s] 62%|██████▏   | 249528/400000 [00:26<00:16, 9325.09it/s] 63%|██████▎   | 250466/400000 [00:26<00:16, 9224.21it/s] 63%|██████▎   | 251393/400000 [00:26<00:16, 9185.25it/s] 63%|██████▎   | 252322/400000 [00:26<00:16, 9215.25it/s] 63%|██████▎   | 253283/400000 [00:26<00:15, 9330.01it/s] 64%|██████▎   | 254235/400000 [00:26<00:15, 9384.33it/s] 64%|██████▍   | 255231/400000 [00:26<00:15, 9546.97it/s] 64%|██████▍   | 256229/400000 [00:26<00:14, 9671.00it/s] 64%|██████▍   | 257198/400000 [00:26<00:14, 9548.01it/s] 65%|██████▍   | 258180/400000 [00:26<00:14, 9625.37it/s] 65%|██████▍   | 259217/400000 [00:27<00:14, 9836.54it/s] 65%|██████▌   | 260220/400000 [00:27<00:14, 9891.75it/s] 65%|██████▌   | 261216/400000 [00:27<00:14, 9909.90it/s] 66%|██████▌   | 262208/400000 [00:27<00:14, 9784.76it/s] 66%|██████▌   | 263188/400000 [00:27<00:14, 9600.21it/s] 66%|██████▌   | 264150/400000 [00:27<00:14, 9426.03it/s] 66%|██████▋   | 265141/400000 [00:27<00:14, 9565.60it/s] 67%|██████▋   | 266100/400000 [00:27<00:15, 8841.91it/s] 67%|██████▋   | 267002/400000 [00:27<00:14, 8893.50it/s] 67%|██████▋   | 267940/400000 [00:27<00:14, 9033.61it/s] 67%|██████▋   | 268855/400000 [00:28<00:14, 9067.26it/s] 67%|██████▋   | 269787/400000 [00:28<00:14, 9140.36it/s] 68%|██████▊   | 270742/400000 [00:28<00:13, 9258.26it/s] 68%|██████▊   | 271671/400000 [00:28<00:14, 9089.04it/s] 68%|██████▊   | 272623/400000 [00:28<00:13, 9213.93it/s] 68%|██████▊   | 273603/400000 [00:28<00:13, 9381.59it/s] 69%|██████▊   | 274597/400000 [00:28<00:13, 9540.32it/s] 69%|██████▉   | 275583/400000 [00:28<00:12, 9631.89it/s] 69%|██████▉   | 276548/400000 [00:28<00:12, 9504.34it/s] 69%|██████▉   | 277538/400000 [00:28<00:12, 9619.39it/s] 70%|██████▉   | 278521/400000 [00:29<00:12, 9680.68it/s] 70%|██████▉   | 279491/400000 [00:29<00:12, 9487.27it/s] 70%|███████   | 280442/400000 [00:29<00:12, 9474.47it/s] 70%|███████   | 281391/400000 [00:29<00:12, 9477.53it/s] 71%|███████   | 282381/400000 [00:29<00:12, 9599.78it/s] 71%|███████   | 283378/400000 [00:29<00:12, 9707.03it/s] 71%|███████   | 284350/400000 [00:29<00:12, 9631.64it/s] 71%|███████▏  | 285314/400000 [00:29<00:12, 9439.47it/s] 72%|███████▏  | 286306/400000 [00:29<00:11, 9575.86it/s] 72%|███████▏  | 287300/400000 [00:29<00:11, 9680.71it/s] 72%|███████▏  | 288311/400000 [00:30<00:11, 9805.47it/s] 72%|███████▏  | 289331/400000 [00:30<00:11, 9918.47it/s] 73%|███████▎  | 290325/400000 [00:30<00:11, 9896.48it/s] 73%|███████▎  | 291316/400000 [00:30<00:11, 9839.73it/s] 73%|███████▎  | 292301/400000 [00:30<00:10, 9805.16it/s] 73%|███████▎  | 293285/400000 [00:30<00:10, 9815.09it/s] 74%|███████▎  | 294267/400000 [00:30<00:10, 9785.44it/s] 74%|███████▍  | 295246/400000 [00:30<00:10, 9588.39it/s] 74%|███████▍  | 296206/400000 [00:30<00:11, 9214.32it/s] 74%|███████▍  | 297225/400000 [00:31<00:10, 9486.46it/s] 75%|███████▍  | 298203/400000 [00:31<00:10, 9572.34it/s] 75%|███████▍  | 299164/400000 [00:31<00:10, 9552.39it/s] 75%|███████▌  | 300122/400000 [00:31<00:10, 9438.51it/s] 75%|███████▌  | 301068/400000 [00:31<00:10, 9420.22it/s] 76%|███████▌  | 302160/400000 [00:31<00:09, 9824.77it/s] 76%|███████▌  | 303182/400000 [00:31<00:09, 9939.45it/s] 76%|███████▌  | 304205/400000 [00:31<00:09, 10024.25it/s] 76%|███████▋  | 305211/400000 [00:31<00:09, 9975.37it/s]  77%|███████▋  | 306222/400000 [00:31<00:09, 10014.65it/s] 77%|███████▋  | 307231/400000 [00:32<00:09, 10036.69it/s] 77%|███████▋  | 308244/400000 [00:32<00:09, 10063.74it/s] 77%|███████▋  | 309266/400000 [00:32<00:08, 10108.15it/s] 78%|███████▊  | 310278/400000 [00:32<00:08, 10060.23it/s] 78%|███████▊  | 311285/400000 [00:32<00:09, 9841.51it/s]  78%|███████▊  | 312289/400000 [00:32<00:08, 9899.50it/s] 78%|███████▊  | 313298/400000 [00:32<00:08, 9955.39it/s] 79%|███████▊  | 314313/400000 [00:32<00:08, 10011.91it/s] 79%|███████▉  | 315315/400000 [00:32<00:08, 9649.91it/s]  79%|███████▉  | 316289/400000 [00:32<00:08, 9671.95it/s] 79%|███████▉  | 317259/400000 [00:33<00:08, 9502.27it/s] 80%|███████▉  | 318212/400000 [00:33<00:08, 9453.91it/s] 80%|███████▉  | 319231/400000 [00:33<00:08, 9660.53it/s] 80%|████████  | 320217/400000 [00:33<00:08, 9714.68it/s] 80%|████████  | 321191/400000 [00:33<00:08, 9656.73it/s] 81%|████████  | 322200/400000 [00:33<00:07, 9781.67it/s] 81%|████████  | 323203/400000 [00:33<00:07, 9851.78it/s] 81%|████████  | 324190/400000 [00:33<00:07, 9808.18it/s] 81%|████████▏ | 325172/400000 [00:33<00:07, 9742.96it/s] 82%|████████▏ | 326147/400000 [00:33<00:07, 9657.85it/s] 82%|████████▏ | 327159/400000 [00:34<00:07, 9789.67it/s] 82%|████████▏ | 328141/400000 [00:34<00:07, 9797.08it/s] 82%|████████▏ | 329193/400000 [00:34<00:07, 10001.10it/s] 83%|████████▎ | 330211/400000 [00:34<00:06, 10051.62it/s] 83%|████████▎ | 331218/400000 [00:34<00:07, 9812.07it/s]  83%|████████▎ | 332202/400000 [00:34<00:06, 9691.07it/s] 83%|████████▎ | 333196/400000 [00:34<00:06, 9764.29it/s] 84%|████████▎ | 334174/400000 [00:34<00:06, 9749.70it/s] 84%|████████▍ | 335150/400000 [00:34<00:06, 9658.52it/s] 84%|████████▍ | 336129/400000 [00:34<00:06, 9696.02it/s] 84%|████████▍ | 337126/400000 [00:35<00:06, 9774.51it/s] 85%|████████▍ | 338130/400000 [00:35<00:06, 9852.42it/s] 85%|████████▍ | 339157/400000 [00:35<00:06, 9972.25it/s] 85%|████████▌ | 340155/400000 [00:35<00:06, 9903.74it/s] 85%|████████▌ | 341147/400000 [00:35<00:06, 9623.14it/s] 86%|████████▌ | 342133/400000 [00:35<00:05, 9691.71it/s] 86%|████████▌ | 343124/400000 [00:35<00:05, 9753.69it/s] 86%|████████▌ | 344117/400000 [00:35<00:05, 9803.47it/s] 86%|████████▋ | 345100/400000 [00:35<00:05, 9810.01it/s] 87%|████████▋ | 346082/400000 [00:36<00:05, 9457.99it/s] 87%|████████▋ | 347083/400000 [00:36<00:05, 9615.69it/s] 87%|████████▋ | 348067/400000 [00:36<00:05, 9680.52it/s] 87%|████████▋ | 349038/400000 [00:36<00:05, 9634.72it/s] 88%|████████▊ | 350003/400000 [00:36<00:05, 9590.99it/s] 88%|████████▊ | 350964/400000 [00:36<00:05, 9465.38it/s] 88%|████████▊ | 351993/400000 [00:36<00:04, 9697.15it/s] 88%|████████▊ | 352968/400000 [00:36<00:04, 9711.14it/s] 88%|████████▊ | 353941/400000 [00:36<00:04, 9686.15it/s] 89%|████████▊ | 354916/400000 [00:36<00:04, 9705.04it/s] 89%|████████▉ | 355902/400000 [00:37<00:04, 9750.02it/s] 89%|████████▉ | 356915/400000 [00:37<00:04, 9859.16it/s] 89%|████████▉ | 357905/400000 [00:37<00:04, 9870.29it/s] 90%|████████▉ | 358896/400000 [00:37<00:04, 9879.73it/s] 90%|████████▉ | 359892/400000 [00:37<00:04, 9900.50it/s] 90%|█████████ | 360883/400000 [00:37<00:04, 9703.47it/s] 90%|█████████ | 361855/400000 [00:37<00:03, 9579.48it/s] 91%|█████████ | 362815/400000 [00:37<00:03, 9460.31it/s] 91%|█████████ | 363763/400000 [00:37<00:03, 9339.64it/s] 91%|█████████ | 364702/400000 [00:37<00:03, 9352.51it/s] 91%|█████████▏| 365675/400000 [00:38<00:03, 9461.98it/s] 92%|█████████▏| 366637/400000 [00:38<00:03, 9506.40it/s] 92%|█████████▏| 367636/400000 [00:38<00:03, 9645.39it/s] 92%|█████████▏| 368625/400000 [00:38<00:03, 9716.75it/s] 92%|█████████▏| 369608/400000 [00:38<00:03, 9749.73it/s] 93%|█████████▎| 370584/400000 [00:38<00:03, 9735.69it/s] 93%|█████████▎| 371587/400000 [00:38<00:02, 9820.52it/s] 93%|█████████▎| 372576/400000 [00:38<00:02, 9840.77it/s] 93%|█████████▎| 373567/400000 [00:38<00:02, 9860.69it/s] 94%|█████████▎| 374567/400000 [00:38<00:02, 9901.61it/s] 94%|█████████▍| 375558/400000 [00:39<00:02, 9775.08it/s] 94%|█████████▍| 376537/400000 [00:39<00:02, 9597.91it/s] 94%|█████████▍| 377517/400000 [00:39<00:02, 9656.33it/s] 95%|█████████▍| 378544/400000 [00:39<00:02, 9830.55it/s] 95%|█████████▍| 379529/400000 [00:39<00:02, 9764.35it/s] 95%|█████████▌| 380507/400000 [00:39<00:02, 9594.63it/s] 95%|█████████▌| 381504/400000 [00:39<00:01, 9703.10it/s] 96%|█████████▌| 382476/400000 [00:39<00:01, 9694.92it/s] 96%|█████████▌| 383447/400000 [00:39<00:01, 9672.44it/s] 96%|█████████▌| 384448/400000 [00:39<00:01, 9769.98it/s] 96%|█████████▋| 385430/400000 [00:40<00:01, 9783.11it/s] 97%|█████████▋| 386409/400000 [00:40<00:01, 9727.23it/s] 97%|█████████▋| 387383/400000 [00:40<00:01, 9568.99it/s] 97%|█████████▋| 388341/400000 [00:40<00:01, 9512.76it/s] 97%|█████████▋| 389293/400000 [00:40<00:01, 9395.56it/s] 98%|█████████▊| 390237/400000 [00:40<00:01, 9406.97it/s] 98%|█████████▊| 391179/400000 [00:40<00:00, 9376.36it/s] 98%|█████████▊| 392118/400000 [00:40<00:00, 9364.36it/s] 98%|█████████▊| 393083/400000 [00:40<00:00, 9446.97it/s] 99%|█████████▊| 394091/400000 [00:40<00:00, 9626.16it/s] 99%|█████████▉| 395076/400000 [00:41<00:00, 9692.05it/s] 99%|█████████▉| 396047/400000 [00:41<00:00, 9461.10it/s] 99%|█████████▉| 397050/400000 [00:41<00:00, 9622.45it/s]100%|█████████▉| 398044/400000 [00:41<00:00, 9714.31it/s]100%|█████████▉| 399040/400000 [00:41<00:00, 9784.40it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9619.78it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1b69a01198>, <torchtext.data.dataset.TabularDataset object at 0x7f1b69a012e8>, <torchtext.vocab.Vocab object at 0x7f1b78152668>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.98 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.98 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.98 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.67 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.67 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.10 file/s]2020-07-03 00:17:16.275955: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-03 00:17:16.279703: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-03 00:17:16.279876: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5650e4286100 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-03 00:17:16.279891: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3629056/9912422 [00:00<00:00, 35977252.92it/s]9920512it [00:00, 33139963.32it/s]                             
0it [00:00, ?it/s]32768it [00:00, 591094.61it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 442104.06it/s]1654784it [00:00, 11113994.20it/s]                         
0it [00:00, ?it/s]8192it [00:00, 171314.73it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
