
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f6271f3eea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f6271f3eea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f62dd1ff1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f62dd1ff1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f62fb546e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f62fb546e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f628a526488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f628a526488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f628a526488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:33, 297223.57it/s]  2%|▏         | 212992/9912422 [00:00<00:25, 385590.12it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 533313.38it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 753367.08it/s] 75%|███████▍  | 7397376/9912422 [00:00<00:02, 1064458.18it/s]9920512it [00:00, 9934265.96it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 139124.02it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 297026.59it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 385145.58it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 532660.11it/s]1654784it [00:00, 2680238.21it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 51799.25it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6271651390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f62716475f8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f628a5260d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f628a5260d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f628a5260d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:01<02:54,  1.08s/ MiB][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:01<02:54,  1.08s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:01<02:53,  1.08s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:01<02:51,  1.08s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:01<02:01,  1.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<02:01,  1.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<02:00,  1.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<01:59,  1.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<01:59,  1.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<01:58,  1.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<01:23,  1.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<01:23,  1.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<01:22,  1.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<01:22,  1.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<01:21,  1.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<01:21,  1.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<01:20,  1.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:56,  2.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:56,  2.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:56,  2.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:56,  2.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:55,  2.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:55,  2.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:39,  3.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:39,  3.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:39,  3.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:38,  3.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:38,  3.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:38,  3.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:27,  4.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:27,  4.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:27,  4.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:27,  4.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:26,  4.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:26,  4.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:26,  4.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:19,  6.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:19,  6.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:19,  6.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:18,  6.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:18,  6.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:18,  6.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:18,  6.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:13,  9.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:13,  9.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:13,  9.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:13,  9.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  9.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:13,  9.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:02<00:07, 15.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:07, 15.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:07, 15.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:07, 15.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:07, 15.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:02<00:05, 20.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:05, 20.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:05, 20.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:05, 20.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:05, 20.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:05, 20.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:02<00:04, 24.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 28.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 28.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 28.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:03, 28.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 28.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:03, 28.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 32.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 32.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 32.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 32.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 32.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 32.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 32.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 36.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 36.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 36.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 36.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 36.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 36.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:02, 36.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 40.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 40.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 40.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 40.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 40.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 40.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 40.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 43.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 43.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 43.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 43.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 43.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 43.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 43.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 45.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 45.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:01, 45.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 45.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 45.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 45.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 45.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 46.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 47.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 47.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 47.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 47.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 47.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 47.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 47.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 46.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 46.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 46.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 48.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 48.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 48.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 48.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 48.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 48.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 48.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 49.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 49.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 49.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 49.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 49.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 49.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 49.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 49.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 49.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 49.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 49.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 49.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 49.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 49.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 49.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 50.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 50.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 50.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 50.15 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 50.15 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 50.15 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 50.15 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 49.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:04<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 50.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 50.50 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 50.50 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 50.50 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 50.50 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 50.50 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 50.50 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 49.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 49.49 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.91s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 49.49 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.91s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.91s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 23.43 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.91s/ url]
0 examples [00:00, ? examples/s]2020-06-26 00:10:03.936869: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-26 00:10:03.949258: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394445000 Hz
2020-06-26 00:10:03.949464: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560b4f657600 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-26 00:10:03.949484: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
18 examples [00:00, 179.39 examples/s]109 examples [00:00, 236.26 examples/s]203 examples [00:00, 304.61 examples/s]296 examples [00:00, 381.42 examples/s]381 examples [00:00, 456.57 examples/s]476 examples [00:00, 540.37 examples/s]566 examples [00:00, 612.62 examples/s]649 examples [00:00, 664.15 examples/s]741 examples [00:00, 724.31 examples/s]834 examples [00:01, 774.52 examples/s]930 examples [00:01, 821.20 examples/s]1020 examples [00:01, 832.75 examples/s]1111 examples [00:01, 852.55 examples/s]1207 examples [00:01, 880.35 examples/s]1298 examples [00:01, 885.56 examples/s]1389 examples [00:01, 845.01 examples/s]1476 examples [00:01, 851.88 examples/s]1563 examples [00:01, 830.80 examples/s]1648 examples [00:01, 811.52 examples/s]1730 examples [00:02, 810.57 examples/s]1812 examples [00:02, 808.11 examples/s]1908 examples [00:02, 846.45 examples/s]2006 examples [00:02, 880.36 examples/s]2100 examples [00:02, 896.25 examples/s]2191 examples [00:02, 897.00 examples/s]2282 examples [00:02, 876.76 examples/s]2371 examples [00:02, 869.34 examples/s]2459 examples [00:02, 854.94 examples/s]2545 examples [00:02, 855.76 examples/s]2631 examples [00:03, 838.66 examples/s]2716 examples [00:03, 832.22 examples/s]2800 examples [00:03, 810.92 examples/s]2882 examples [00:03, 806.85 examples/s]2971 examples [00:03, 827.96 examples/s]3065 examples [00:03, 856.47 examples/s]3161 examples [00:03, 882.31 examples/s]3257 examples [00:03, 902.29 examples/s]3353 examples [00:03, 917.92 examples/s]3447 examples [00:04, 923.51 examples/s]3540 examples [00:04, 853.19 examples/s]3627 examples [00:04, 844.61 examples/s]3713 examples [00:04, 842.12 examples/s]3815 examples [00:04, 887.27 examples/s]3913 examples [00:04, 911.90 examples/s]4006 examples [00:04, 898.36 examples/s]4097 examples [00:04, 876.72 examples/s]4186 examples [00:04, 845.56 examples/s]4272 examples [00:04, 804.76 examples/s]4354 examples [00:05, 794.62 examples/s]4436 examples [00:05, 801.93 examples/s]4534 examples [00:05, 847.19 examples/s]4629 examples [00:05, 873.37 examples/s]4722 examples [00:05, 886.63 examples/s]4812 examples [00:05, 839.14 examples/s]4900 examples [00:05, 850.49 examples/s]4995 examples [00:05, 876.50 examples/s]5094 examples [00:05, 905.92 examples/s]5186 examples [00:06, 909.14 examples/s]5278 examples [00:06, 884.61 examples/s]5367 examples [00:06, 845.20 examples/s]5453 examples [00:06, 813.08 examples/s]5547 examples [00:06, 846.84 examples/s]5638 examples [00:06, 862.52 examples/s]5725 examples [00:06, 847.53 examples/s]5817 examples [00:06, 865.63 examples/s]5905 examples [00:06, 856.20 examples/s]5992 examples [00:06, 859.35 examples/s]6086 examples [00:07, 879.83 examples/s]6175 examples [00:07, 830.11 examples/s]6259 examples [00:07, 831.25 examples/s]6343 examples [00:07, 795.99 examples/s]6424 examples [00:07, 756.60 examples/s]6506 examples [00:07, 774.42 examples/s]6585 examples [00:07, 769.01 examples/s]6663 examples [00:07, 759.39 examples/s]6744 examples [00:07, 771.28 examples/s]6822 examples [00:08, 756.78 examples/s]6900 examples [00:08, 762.19 examples/s]6977 examples [00:08, 756.20 examples/s]7053 examples [00:08, 740.47 examples/s]7147 examples [00:08, 790.64 examples/s]7243 examples [00:08, 834.67 examples/s]7328 examples [00:08, 821.46 examples/s]7412 examples [00:08, 788.86 examples/s]7492 examples [00:08, 767.36 examples/s]7570 examples [00:09, 770.31 examples/s]7648 examples [00:09, 763.88 examples/s]7729 examples [00:09, 776.73 examples/s]7809 examples [00:09, 780.88 examples/s]7893 examples [00:09, 795.42 examples/s]7985 examples [00:09, 826.70 examples/s]8080 examples [00:09, 859.01 examples/s]8167 examples [00:09, 821.32 examples/s]8266 examples [00:09, 863.69 examples/s]8356 examples [00:09, 873.61 examples/s]8445 examples [00:10, 860.15 examples/s]8532 examples [00:10, 840.17 examples/s]8617 examples [00:10, 783.92 examples/s]8697 examples [00:10, 780.82 examples/s]8785 examples [00:10, 807.81 examples/s]8871 examples [00:10, 821.72 examples/s]8954 examples [00:10, 813.81 examples/s]9036 examples [00:10, 809.73 examples/s]9118 examples [00:10, 799.32 examples/s]9202 examples [00:10, 808.92 examples/s]9297 examples [00:11, 846.06 examples/s]9383 examples [00:11, 812.06 examples/s]9465 examples [00:11, 812.43 examples/s]9552 examples [00:11, 828.54 examples/s]9636 examples [00:11, 806.39 examples/s]9718 examples [00:11, 771.86 examples/s]9796 examples [00:11, 751.22 examples/s]9872 examples [00:11, 742.31 examples/s]9955 examples [00:11, 765.26 examples/s]10033 examples [00:12, 769.27 examples/s]10125 examples [00:12, 808.26 examples/s]10214 examples [00:12, 830.10 examples/s]10298 examples [00:12, 795.88 examples/s]10389 examples [00:12, 826.98 examples/s]10482 examples [00:12, 855.04 examples/s]10576 examples [00:12, 878.11 examples/s]10672 examples [00:12, 898.91 examples/s]10770 examples [00:12, 919.63 examples/s]10863 examples [00:12, 864.15 examples/s]10951 examples [00:13, 857.49 examples/s]11038 examples [00:13, 849.17 examples/s]11137 examples [00:13, 886.31 examples/s]11228 examples [00:13, 891.99 examples/s]11322 examples [00:13, 898.96 examples/s]11413 examples [00:13, 900.28 examples/s]11504 examples [00:13, 879.56 examples/s]11593 examples [00:13, 867.52 examples/s]11681 examples [00:13, 844.63 examples/s]11766 examples [00:14, 813.98 examples/s]11848 examples [00:14, 770.92 examples/s]11940 examples [00:14, 807.58 examples/s]12030 examples [00:14, 831.50 examples/s]12121 examples [00:14, 851.14 examples/s]12207 examples [00:14, 842.54 examples/s]12292 examples [00:14, 821.50 examples/s]12375 examples [00:14, 810.68 examples/s]12457 examples [00:14, 811.39 examples/s]12539 examples [00:15, 791.38 examples/s]12624 examples [00:15, 806.64 examples/s]12705 examples [00:15, 807.50 examples/s]12786 examples [00:15, 796.44 examples/s]12883 examples [00:15, 840.68 examples/s]12977 examples [00:15, 867.37 examples/s]13065 examples [00:15, 865.19 examples/s]13153 examples [00:15, 831.35 examples/s]13237 examples [00:15, 828.97 examples/s]13321 examples [00:15, 819.69 examples/s]13415 examples [00:16, 852.39 examples/s]13512 examples [00:16, 883.97 examples/s]13609 examples [00:16, 906.30 examples/s]13701 examples [00:16, 892.33 examples/s]13791 examples [00:16, 861.27 examples/s]13878 examples [00:16, 809.34 examples/s]13960 examples [00:16, 778.67 examples/s]14039 examples [00:16, 745.69 examples/s]14119 examples [00:16, 759.13 examples/s]14213 examples [00:17, 804.81 examples/s]14307 examples [00:17, 839.25 examples/s]14399 examples [00:17, 861.07 examples/s]14492 examples [00:17, 880.11 examples/s]14581 examples [00:17, 871.25 examples/s]14669 examples [00:17, 846.14 examples/s]14755 examples [00:17, 801.64 examples/s]14837 examples [00:17, 796.40 examples/s]14919 examples [00:17, 801.24 examples/s]15001 examples [00:17, 805.18 examples/s]15092 examples [00:18, 833.16 examples/s]15184 examples [00:18, 855.88 examples/s]15272 examples [00:18, 861.13 examples/s]15367 examples [00:18, 885.84 examples/s]15461 examples [00:18, 898.16 examples/s]15552 examples [00:18, 853.52 examples/s]15639 examples [00:18, 808.78 examples/s]15721 examples [00:18, 779.67 examples/s]15800 examples [00:18, 780.80 examples/s]15882 examples [00:19, 790.13 examples/s]15962 examples [00:19, 763.12 examples/s]16047 examples [00:19, 784.88 examples/s]16140 examples [00:19, 822.94 examples/s]16235 examples [00:19, 855.26 examples/s]16322 examples [00:19, 856.79 examples/s]16409 examples [00:19, 829.93 examples/s]16503 examples [00:19, 858.59 examples/s]16590 examples [00:19, 830.32 examples/s]16674 examples [00:19, 811.06 examples/s]16756 examples [00:20, 806.90 examples/s]16838 examples [00:20, 775.52 examples/s]16917 examples [00:20, 759.46 examples/s]16995 examples [00:20, 762.94 examples/s]17072 examples [00:20, 726.12 examples/s]17146 examples [00:20, 706.25 examples/s]17228 examples [00:20, 735.82 examples/s]17306 examples [00:20, 744.31 examples/s]17381 examples [00:20, 721.55 examples/s]17454 examples [00:21, 719.41 examples/s]17527 examples [00:21, 715.30 examples/s]17619 examples [00:21, 764.52 examples/s]17715 examples [00:21, 814.05 examples/s]17811 examples [00:21, 850.80 examples/s]17905 examples [00:21, 875.27 examples/s]17995 examples [00:21, 881.04 examples/s]18085 examples [00:21, 885.45 examples/s]18175 examples [00:21, 859.58 examples/s]18267 examples [00:21, 874.21 examples/s]18355 examples [00:22, 866.59 examples/s]18448 examples [00:22, 883.83 examples/s]18537 examples [00:22, 884.04 examples/s]18626 examples [00:22, 879.00 examples/s]18715 examples [00:22, 875.19 examples/s]18808 examples [00:22, 887.35 examples/s]18897 examples [00:22, 857.02 examples/s]18984 examples [00:22, 855.99 examples/s]19070 examples [00:22, 829.13 examples/s]19154 examples [00:23, 794.44 examples/s]19234 examples [00:23, 779.93 examples/s]19313 examples [00:23, 750.40 examples/s]19389 examples [00:23, 725.13 examples/s]19463 examples [00:23, 719.71 examples/s]19539 examples [00:23, 730.01 examples/s]19613 examples [00:23, 717.01 examples/s]19691 examples [00:23, 734.24 examples/s]19784 examples [00:23, 782.83 examples/s]19878 examples [00:23, 823.48 examples/s]19970 examples [00:24, 848.11 examples/s]20056 examples [00:24, 794.80 examples/s]20142 examples [00:24, 811.86 examples/s]20232 examples [00:24, 834.68 examples/s]20321 examples [00:24, 850.16 examples/s]20412 examples [00:24, 865.01 examples/s]20500 examples [00:24, 837.76 examples/s]20585 examples [00:24, 795.17 examples/s]20666 examples [00:24, 746.21 examples/s]20742 examples [00:25, 725.00 examples/s]20817 examples [00:25, 730.68 examples/s]20891 examples [00:25, 728.96 examples/s]20967 examples [00:25, 737.01 examples/s]21042 examples [00:25, 738.33 examples/s]21118 examples [00:25, 742.57 examples/s]21193 examples [00:25, 739.62 examples/s]21280 examples [00:25, 772.80 examples/s]21360 examples [00:25, 779.86 examples/s]21439 examples [00:25, 759.68 examples/s]21516 examples [00:26, 724.62 examples/s]21590 examples [00:26, 707.00 examples/s]21668 examples [00:26, 727.32 examples/s]21742 examples [00:26, 728.99 examples/s]21824 examples [00:26, 753.74 examples/s]21900 examples [00:26, 742.49 examples/s]21975 examples [00:26, 724.13 examples/s]22053 examples [00:26, 735.71 examples/s]22127 examples [00:26, 728.48 examples/s]22201 examples [00:27, 722.47 examples/s]22274 examples [00:27, 718.16 examples/s]22356 examples [00:27, 744.36 examples/s]22442 examples [00:27, 773.41 examples/s]22524 examples [00:27, 786.15 examples/s]22604 examples [00:27, 780.50 examples/s]22692 examples [00:27, 805.72 examples/s]22784 examples [00:27, 834.72 examples/s]22876 examples [00:27, 856.53 examples/s]22963 examples [00:27, 828.28 examples/s]23047 examples [00:28, 784.75 examples/s]23127 examples [00:28, 767.43 examples/s]23205 examples [00:28, 727.27 examples/s]23282 examples [00:28, 737.30 examples/s]23358 examples [00:28, 742.95 examples/s]23433 examples [00:28, 740.90 examples/s]23508 examples [00:28, 734.23 examples/s]23590 examples [00:28, 757.87 examples/s]23683 examples [00:28, 800.57 examples/s]23782 examples [00:29, 847.16 examples/s]23871 examples [00:29, 857.30 examples/s]23958 examples [00:29, 846.12 examples/s]24044 examples [00:29, 833.68 examples/s]24128 examples [00:29, 829.54 examples/s]24212 examples [00:29, 820.75 examples/s]24295 examples [00:29, 806.13 examples/s]24376 examples [00:29, 790.84 examples/s]24456 examples [00:29, 762.13 examples/s]24533 examples [00:29, 759.48 examples/s]24610 examples [00:30, 754.08 examples/s]24704 examples [00:30, 800.29 examples/s]24796 examples [00:30, 831.38 examples/s]24881 examples [00:30, 822.59 examples/s]24964 examples [00:30, 811.86 examples/s]25046 examples [00:30, 787.03 examples/s]25126 examples [00:30, 774.98 examples/s]25204 examples [00:30, 770.88 examples/s]25282 examples [00:30, 763.53 examples/s]25359 examples [00:31, 738.17 examples/s]25438 examples [00:31, 751.02 examples/s]25514 examples [00:31, 749.56 examples/s]25590 examples [00:31, 729.65 examples/s]25665 examples [00:31, 734.11 examples/s]25744 examples [00:31, 748.54 examples/s]25820 examples [00:31, 735.87 examples/s]25894 examples [00:31, 734.72 examples/s]25968 examples [00:31, 720.88 examples/s]26041 examples [00:31, 707.41 examples/s]26112 examples [00:32, 701.64 examples/s]26185 examples [00:32, 709.58 examples/s]26257 examples [00:32, 705.34 examples/s]26329 examples [00:32, 707.96 examples/s]26407 examples [00:32, 727.78 examples/s]26487 examples [00:32, 746.92 examples/s]26576 examples [00:32, 782.57 examples/s]26665 examples [00:32, 811.52 examples/s]26749 examples [00:32, 817.62 examples/s]26845 examples [00:32, 854.14 examples/s]26940 examples [00:33, 880.43 examples/s]27033 examples [00:33, 893.46 examples/s]27123 examples [00:33, 863.48 examples/s]27210 examples [00:33, 799.55 examples/s]27292 examples [00:33, 804.76 examples/s]27391 examples [00:33, 850.81 examples/s]27478 examples [00:33, 853.83 examples/s]27572 examples [00:33, 875.35 examples/s]27661 examples [00:33, 842.62 examples/s]27747 examples [00:34, 829.71 examples/s]27836 examples [00:34, 846.22 examples/s]27926 examples [00:34, 860.33 examples/s]28017 examples [00:34, 871.69 examples/s]28108 examples [00:34, 881.81 examples/s]28197 examples [00:34, 820.47 examples/s]28281 examples [00:34, 785.37 examples/s]28361 examples [00:34, 775.22 examples/s]28440 examples [00:34, 733.93 examples/s]28515 examples [00:35, 718.99 examples/s]28597 examples [00:35, 745.94 examples/s]28673 examples [00:35, 747.77 examples/s]28749 examples [00:35, 727.87 examples/s]28823 examples [00:35, 730.11 examples/s]28897 examples [00:35, 724.54 examples/s]28970 examples [00:35, 703.49 examples/s]29049 examples [00:35, 724.54 examples/s]29136 examples [00:35, 761.08 examples/s]29215 examples [00:35, 769.09 examples/s]29304 examples [00:36, 799.88 examples/s]29385 examples [00:36, 796.54 examples/s]29466 examples [00:36, 797.76 examples/s]29547 examples [00:36, 794.48 examples/s]29627 examples [00:36, 795.32 examples/s]29712 examples [00:36, 809.06 examples/s]29794 examples [00:36, 791.14 examples/s]29874 examples [00:36, 783.50 examples/s]29957 examples [00:36, 793.64 examples/s]30037 examples [00:36, 762.90 examples/s]30116 examples [00:37, 768.94 examples/s]30194 examples [00:37, 739.37 examples/s]30286 examples [00:37, 783.76 examples/s]30377 examples [00:37, 816.84 examples/s]30460 examples [00:37, 808.95 examples/s]30542 examples [00:37, 774.50 examples/s]30621 examples [00:37, 756.98 examples/s]30698 examples [00:37, 746.65 examples/s]30774 examples [00:37, 740.05 examples/s]30849 examples [00:38, 737.62 examples/s]30924 examples [00:38, 723.32 examples/s]30997 examples [00:38, 648.91 examples/s]31081 examples [00:38, 696.33 examples/s]31176 examples [00:38, 756.44 examples/s]31271 examples [00:38, 805.55 examples/s]31365 examples [00:38, 841.36 examples/s]31457 examples [00:38, 862.61 examples/s]31546 examples [00:38, 787.90 examples/s]31628 examples [00:39, 753.83 examples/s]31711 examples [00:39, 771.87 examples/s]31790 examples [00:39, 751.48 examples/s]31867 examples [00:39, 714.46 examples/s]31940 examples [00:39, 687.90 examples/s]32011 examples [00:39, 693.29 examples/s]32084 examples [00:39, 700.12 examples/s]32155 examples [00:39, 698.50 examples/s]32227 examples [00:39, 702.48 examples/s]32303 examples [00:40, 718.08 examples/s]32376 examples [00:40, 714.61 examples/s]32465 examples [00:40, 757.10 examples/s]32564 examples [00:40, 812.62 examples/s]32652 examples [00:40, 829.01 examples/s]32742 examples [00:40, 848.63 examples/s]32830 examples [00:40, 857.50 examples/s]32920 examples [00:40, 868.33 examples/s]33016 examples [00:40, 893.00 examples/s]33106 examples [00:40, 894.33 examples/s]33199 examples [00:41, 904.36 examples/s]33290 examples [00:41, 872.55 examples/s]33378 examples [00:41, 827.22 examples/s]33463 examples [00:41, 832.77 examples/s]33553 examples [00:41, 849.75 examples/s]33639 examples [00:41, 830.76 examples/s]33723 examples [00:41, 789.27 examples/s]33803 examples [00:41, 743.00 examples/s]33879 examples [00:41, 738.47 examples/s]33963 examples [00:42, 764.01 examples/s]34041 examples [00:42, 748.21 examples/s]34118 examples [00:42, 751.56 examples/s]34197 examples [00:42, 762.60 examples/s]34274 examples [00:42, 728.16 examples/s]34348 examples [00:42, 730.03 examples/s]34422 examples [00:42, 714.28 examples/s]34499 examples [00:42, 729.35 examples/s]34573 examples [00:42, 722.76 examples/s]34646 examples [00:42, 712.90 examples/s]34718 examples [00:43, 701.70 examples/s]34789 examples [00:43, 703.16 examples/s]34860 examples [00:43, 692.16 examples/s]34930 examples [00:43, 690.27 examples/s]35000 examples [00:43, 684.19 examples/s]35069 examples [00:43, 666.85 examples/s]35137 examples [00:43, 668.05 examples/s]35207 examples [00:43, 676.14 examples/s]35276 examples [00:43, 677.54 examples/s]35347 examples [00:43, 686.68 examples/s]35420 examples [00:44, 698.60 examples/s]35519 examples [00:44, 764.89 examples/s]35608 examples [00:44, 796.87 examples/s]35703 examples [00:44, 836.68 examples/s]35798 examples [00:44, 866.61 examples/s]35887 examples [00:44, 832.76 examples/s]35972 examples [00:44, 756.68 examples/s]36050 examples [00:44, 755.24 examples/s]36128 examples [00:44, 725.04 examples/s]36202 examples [00:45, 693.08 examples/s]36273 examples [00:45, 671.77 examples/s]36346 examples [00:45, 688.00 examples/s]36423 examples [00:45, 708.69 examples/s]36496 examples [00:45, 714.37 examples/s]36587 examples [00:45, 762.79 examples/s]36679 examples [00:45, 801.59 examples/s]36768 examples [00:45, 825.08 examples/s]36852 examples [00:45, 828.06 examples/s]36937 examples [00:45, 834.42 examples/s]37022 examples [00:46, 809.50 examples/s]37104 examples [00:46, 785.43 examples/s]37185 examples [00:46, 791.01 examples/s]37269 examples [00:46, 803.48 examples/s]37366 examples [00:46, 845.44 examples/s]37462 examples [00:46, 876.78 examples/s]37559 examples [00:46, 901.12 examples/s]37652 examples [00:46, 906.61 examples/s]37744 examples [00:46, 851.71 examples/s]37831 examples [00:47, 800.22 examples/s]37913 examples [00:47, 792.45 examples/s]37994 examples [00:47, 785.59 examples/s]38074 examples [00:47, 779.21 examples/s]38153 examples [00:47, 722.09 examples/s]38230 examples [00:47, 734.91 examples/s]38305 examples [00:47, 721.47 examples/s]38380 examples [00:47, 729.50 examples/s]38461 examples [00:47, 751.48 examples/s]38549 examples [00:48, 784.61 examples/s]38643 examples [00:48, 823.74 examples/s]38727 examples [00:48, 827.21 examples/s]38816 examples [00:48, 844.42 examples/s]38906 examples [00:48, 859.17 examples/s]38993 examples [00:48, 821.97 examples/s]39076 examples [00:48, 821.96 examples/s]39159 examples [00:48, 779.56 examples/s]39238 examples [00:48, 754.98 examples/s]39315 examples [00:48, 743.54 examples/s]39395 examples [00:49, 757.86 examples/s]39478 examples [00:49, 777.38 examples/s]39560 examples [00:49, 787.54 examples/s]39640 examples [00:49, 789.28 examples/s]39720 examples [00:49, 777.05 examples/s]39806 examples [00:49, 798.73 examples/s]39887 examples [00:49, 797.75 examples/s]39967 examples [00:49, 760.86 examples/s]40044 examples [00:49, 687.16 examples/s]40116 examples [00:50, 696.31 examples/s]40187 examples [00:50, 663.01 examples/s]40259 examples [00:50, 677.92 examples/s]40328 examples [00:50, 657.72 examples/s]40405 examples [00:50, 685.44 examples/s]40478 examples [00:50, 697.96 examples/s]40549 examples [00:50, 699.33 examples/s]40620 examples [00:50, 684.98 examples/s]40693 examples [00:50, 694.72 examples/s]40768 examples [00:50, 708.98 examples/s]40844 examples [00:51, 723.37 examples/s]40917 examples [00:51, 712.88 examples/s]40989 examples [00:51, 696.00 examples/s]41061 examples [00:51, 698.98 examples/s]41155 examples [00:51, 756.17 examples/s]41248 examples [00:51, 799.89 examples/s]41343 examples [00:51, 838.02 examples/s]41429 examples [00:51, 838.74 examples/s]41515 examples [00:51, 781.89 examples/s]41595 examples [00:52, 756.59 examples/s]41675 examples [00:52, 768.29 examples/s]41754 examples [00:52, 771.96 examples/s]41832 examples [00:52, 760.63 examples/s]41909 examples [00:52, 760.77 examples/s]41986 examples [00:52, 744.49 examples/s]42061 examples [00:52, 743.14 examples/s]42141 examples [00:52, 757.46 examples/s]42219 examples [00:52, 762.85 examples/s]42301 examples [00:52, 778.25 examples/s]42399 examples [00:53, 829.21 examples/s]42494 examples [00:53, 860.47 examples/s]42590 examples [00:53, 886.94 examples/s]42680 examples [00:53, 878.87 examples/s]42769 examples [00:53, 828.32 examples/s]42861 examples [00:53, 853.48 examples/s]42952 examples [00:53, 869.51 examples/s]43042 examples [00:53, 877.00 examples/s]43131 examples [00:53, 872.65 examples/s]43219 examples [00:54, 810.22 examples/s]43302 examples [00:54, 768.86 examples/s]43381 examples [00:54, 757.43 examples/s]43458 examples [00:54, 758.36 examples/s]43542 examples [00:54, 780.54 examples/s]43638 examples [00:54, 824.95 examples/s]43734 examples [00:54, 861.10 examples/s]43829 examples [00:54, 885.37 examples/s]43924 examples [00:54, 901.95 examples/s]44016 examples [00:54, 906.34 examples/s]44108 examples [00:55, 860.00 examples/s]44195 examples [00:55, 848.65 examples/s]44281 examples [00:55, 835.21 examples/s]44366 examples [00:55, 804.48 examples/s]44448 examples [00:55, 779.12 examples/s]44527 examples [00:55, 752.24 examples/s]44606 examples [00:55, 760.63 examples/s]44683 examples [00:55, 744.00 examples/s]44763 examples [00:55, 759.79 examples/s]44859 examples [00:56, 807.93 examples/s]44954 examples [00:56, 845.66 examples/s]45040 examples [00:56, 829.57 examples/s]45124 examples [00:56, 802.88 examples/s]45206 examples [00:56, 778.16 examples/s]45285 examples [00:56, 758.27 examples/s]45362 examples [00:56, 759.43 examples/s]45439 examples [00:56, 756.69 examples/s]45525 examples [00:56, 781.39 examples/s]45604 examples [00:57, 747.27 examples/s]45686 examples [00:57, 765.96 examples/s]45783 examples [00:57, 815.34 examples/s]45877 examples [00:57, 847.78 examples/s]45963 examples [00:57, 805.01 examples/s]46045 examples [00:57, 749.86 examples/s]46122 examples [00:57, 747.72 examples/s]46198 examples [00:57, 726.21 examples/s]46272 examples [00:57, 718.87 examples/s]46345 examples [00:57, 703.26 examples/s]46421 examples [00:58, 716.55 examples/s]46494 examples [00:58, 710.18 examples/s]46566 examples [00:58, 700.26 examples/s]46651 examples [00:58, 737.81 examples/s]46726 examples [00:58, 734.42 examples/s]46804 examples [00:58, 746.13 examples/s]46880 examples [00:58, 739.11 examples/s]46956 examples [00:58, 743.91 examples/s]47031 examples [00:58, 733.99 examples/s]47115 examples [00:59, 762.69 examples/s]47192 examples [00:59, 740.89 examples/s]47267 examples [00:59, 723.42 examples/s]47340 examples [00:59, 717.52 examples/s]47414 examples [00:59, 724.02 examples/s]47507 examples [00:59, 775.39 examples/s]47604 examples [00:59, 823.05 examples/s]47699 examples [00:59, 856.03 examples/s]47787 examples [00:59, 847.95 examples/s]47873 examples [00:59, 792.38 examples/s]47954 examples [01:00, 758.14 examples/s]48032 examples [01:00, 727.45 examples/s]48106 examples [01:00, 709.95 examples/s]48186 examples [01:00, 733.15 examples/s]48261 examples [01:00, 729.66 examples/s]48358 examples [01:00, 787.66 examples/s]48441 examples [01:00, 798.48 examples/s]48529 examples [01:00, 820.81 examples/s]48615 examples [01:00, 831.86 examples/s]48699 examples [01:01, 772.63 examples/s]48778 examples [01:01, 738.78 examples/s]48854 examples [01:01, 721.44 examples/s]48928 examples [01:01, 712.06 examples/s]49000 examples [01:01, 710.89 examples/s]49072 examples [01:01, 707.55 examples/s]49149 examples [01:01, 724.35 examples/s]49225 examples [01:01, 733.64 examples/s]49299 examples [01:01, 714.72 examples/s]49371 examples [01:02, 685.13 examples/s]49448 examples [01:02, 708.17 examples/s]49520 examples [01:02, 711.11 examples/s]49592 examples [01:02, 708.41 examples/s]49664 examples [01:02, 685.88 examples/s]49733 examples [01:02, 685.60 examples/s]49815 examples [01:02, 721.00 examples/s]49893 examples [01:02, 737.66 examples/s]49972 examples [01:02, 752.02 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s]  7%|▋         | 3311/50000 [00:00<00:01, 33109.87 examples/s] 23%|██▎       | 11729/50000 [00:00<00:00, 40475.28 examples/s] 39%|███▉      | 19718/50000 [00:00<00:00, 47506.51 examples/s] 55%|█████▌    | 27586/50000 [00:00<00:00, 53914.18 examples/s] 72%|███████▏  | 35836/50000 [00:00<00:00, 60168.10 examples/s] 88%|████████▊ | 43960/50000 [00:00<00:00, 65244.95 examples/s]                                                               0 examples [00:00, ? examples/s]72 examples [00:00, 718.03 examples/s]159 examples [00:00, 756.37 examples/s]255 examples [00:00, 807.03 examples/s]348 examples [00:00, 839.94 examples/s]440 examples [00:00, 860.35 examples/s]541 examples [00:00, 898.18 examples/s]631 examples [00:00, 898.49 examples/s]716 examples [00:00, 829.52 examples/s]797 examples [00:00, 813.57 examples/s]879 examples [00:01, 812.69 examples/s]960 examples [00:01, 801.21 examples/s]1040 examples [00:01, 747.87 examples/s]1115 examples [00:01, 723.62 examples/s]1192 examples [00:01, 735.33 examples/s]1267 examples [00:01, 735.06 examples/s]1353 examples [00:01, 767.38 examples/s]1444 examples [00:01, 802.07 examples/s]1539 examples [00:01, 841.33 examples/s]1633 examples [00:01, 868.42 examples/s]1721 examples [00:02, 825.93 examples/s]1805 examples [00:02, 783.40 examples/s]1885 examples [00:02, 786.27 examples/s]1965 examples [00:02, 743.83 examples/s]2045 examples [00:02, 758.86 examples/s]2122 examples [00:02, 761.28 examples/s]2199 examples [00:02, 743.05 examples/s]2277 examples [00:02, 751.49 examples/s]2353 examples [00:02, 732.87 examples/s]2442 examples [00:03, 773.34 examples/s]2533 examples [00:03, 808.50 examples/s]2630 examples [00:03, 849.14 examples/s]2717 examples [00:03, 789.95 examples/s]2798 examples [00:03, 770.54 examples/s]2877 examples [00:03, 744.86 examples/s]2954 examples [00:03, 750.46 examples/s]3030 examples [00:03, 734.38 examples/s]3109 examples [00:03, 747.83 examples/s]3187 examples [00:04, 755.21 examples/s]3266 examples [00:04, 765.07 examples/s]3343 examples [00:04, 757.17 examples/s]3421 examples [00:04, 762.67 examples/s]3498 examples [00:04, 727.69 examples/s]3574 examples [00:04, 736.84 examples/s]3652 examples [00:04, 747.95 examples/s]3728 examples [00:04, 732.34 examples/s]3802 examples [00:04, 722.05 examples/s]3875 examples [00:04, 717.19 examples/s]3957 examples [00:05, 741.58 examples/s]4032 examples [00:05, 722.25 examples/s]4105 examples [00:05, 702.46 examples/s]4177 examples [00:05, 703.04 examples/s]4248 examples [00:05, 678.92 examples/s]4318 examples [00:05, 683.02 examples/s]4405 examples [00:05, 729.28 examples/s]4503 examples [00:05, 788.11 examples/s]4601 examples [00:05, 835.78 examples/s]4700 examples [00:06, 876.62 examples/s]4790 examples [00:06, 842.87 examples/s]4884 examples [00:06, 868.11 examples/s]4981 examples [00:06, 894.70 examples/s]5072 examples [00:06, 848.56 examples/s]5159 examples [00:06, 830.64 examples/s]5244 examples [00:06, 794.31 examples/s]5332 examples [00:06, 815.91 examples/s]5421 examples [00:06, 835.53 examples/s]5516 examples [00:06, 866.19 examples/s]5604 examples [00:07, 846.67 examples/s]5690 examples [00:07, 821.10 examples/s]5773 examples [00:07, 813.02 examples/s]5870 examples [00:07, 852.47 examples/s]5968 examples [00:07, 885.05 examples/s]6058 examples [00:07, 870.80 examples/s]6146 examples [00:07, 803.01 examples/s]6228 examples [00:07, 762.10 examples/s]6307 examples [00:07, 768.40 examples/s]6385 examples [00:08, 738.72 examples/s]6472 examples [00:08, 772.06 examples/s]6569 examples [00:08, 822.28 examples/s]6653 examples [00:08, 820.58 examples/s]6748 examples [00:08, 855.42 examples/s]6835 examples [00:08, 822.93 examples/s]6919 examples [00:08, 802.08 examples/s]7001 examples [00:08, 774.23 examples/s]7080 examples [00:08, 751.20 examples/s]7160 examples [00:09, 763.74 examples/s]7237 examples [00:09, 751.62 examples/s]7313 examples [00:09, 753.99 examples/s]7390 examples [00:09, 757.82 examples/s]7467 examples [00:09, 743.55 examples/s]7544 examples [00:09, 749.70 examples/s]7620 examples [00:09, 750.36 examples/s]7702 examples [00:09, 769.14 examples/s]7790 examples [00:09, 797.37 examples/s]7888 examples [00:09, 844.54 examples/s]7985 examples [00:10, 877.66 examples/s]8083 examples [00:10, 904.67 examples/s]8177 examples [00:10, 912.71 examples/s]8269 examples [00:10, 888.35 examples/s]8359 examples [00:10, 841.32 examples/s]8445 examples [00:10, 791.16 examples/s]8526 examples [00:10, 745.76 examples/s]8603 examples [00:10, 717.83 examples/s]8677 examples [00:10, 701.33 examples/s]8760 examples [00:11, 734.46 examples/s]8835 examples [00:11, 730.66 examples/s]8928 examples [00:11, 779.59 examples/s]9023 examples [00:11, 823.10 examples/s]9117 examples [00:11, 854.00 examples/s]9210 examples [00:11, 872.98 examples/s]9305 examples [00:11, 893.44 examples/s]9396 examples [00:11, 860.13 examples/s]9483 examples [00:11, 798.75 examples/s]9565 examples [00:12, 778.37 examples/s]9645 examples [00:12, 772.70 examples/s]9724 examples [00:12, 740.53 examples/s]9799 examples [00:12, 734.99 examples/s]9885 examples [00:12, 767.41 examples/s]9966 examples [00:12, 776.00 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 73%|███████▎  | 7281/10000 [00:00<00:00, 72808.85 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP8HTZ8/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP8HTZ8/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f628a526488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f628a526488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f628a526488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6289437fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6224739080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f628a526488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f628a526488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f628a526488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f62716473c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6271647630>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f620da82ea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f620da82ea0> 

  function with postional parmater data_info <function split_train_valid at 0x7f620da82ea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f620da74048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f620da74048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f620da74048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=53bcdad97c1741743f3e8d3b0ff025d537c84c154ae505ee74524692f61be870
  Stored in directory: /tmp/pip-ephem-wheel-cache-jty2k8o9/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<9:52:53, 24.2kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<6:55:58, 34.5kB/s] .vector_cache/glove.6B.zip:   1%|          | 5.73M/862M [00:00<4:49:29, 49.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.11M/862M [00:00<3:21:57, 70.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.6M/862M [00:00<2:20:32, 101kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.3M/862M [00:00<1:37:10, 144kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:00<1:07:18, 205kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:01<46:37, 292kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:01<32:25, 417kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:01<23:48, 567kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<17:01, 790kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<10:47:24, 20.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:03<7:32:29, 29.7kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<5:18:48, 42.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:05<3:44:18, 59.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<2:38:14, 84.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:07<1:51:23, 119kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<1:19:42, 166kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:09<56:23, 235kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:11<41:27, 318kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:11<29:47, 442kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<22:46, 575kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:13<16:55, 774kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<12:25, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<5:55:43, 36.7kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:14<4:08:00, 52.4kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<3:05:16, 70.1kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:16<2:10:25, 99.6kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:18<1:32:51, 139kB/s] .vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<1:05:33, 197kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:20<47:44, 269kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:20<34:13, 375kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<24:08, 531kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:20<17:05, 748kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:22<45:29, 281kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<33:27, 382kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:22<23:38, 540kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:22<16:43, 761kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:23<21:53, 581kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:24<16:18, 780kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<11:37, 1.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<12:28, 1.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<5:27:44, 38.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<3:49:16, 55.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:25<2:40:21, 78.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<1:58:36, 106kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<1:24:49, 149kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<59:29, 211kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:27<41:42, 301kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<55:33, 226kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<39:53, 314kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:29<28:00, 446kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<24:20, 512kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<18:02, 691kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<12:47, 972kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<12:40, 978kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<09:55, 1.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<07:08, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<10:31, 1.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<5:07:55, 40.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<3:35:26, 57.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:34<2:30:33, 81.7kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<1:55:19, 107kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<1:21:33, 151kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<57:09, 214kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<43:21, 282kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<31:29, 388kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<22:11, 549kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<18:40, 651kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<14:47, 822kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<10:34, 1.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:40<07:32, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<34:45, 348kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<25:30, 474kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<18:01, 668kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<15:35, 771kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<11:57, 1.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<09:01, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<4:56:16, 40.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<3:27:13, 57.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:45<2:24:45, 82.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<1:57:15, 102kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<1:23:17, 143kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<58:18, 204kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<44:20, 267kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<32:38, 363kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<23:04, 512kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:49<16:15, 724kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<29:47, 395kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<22:14, 529kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<15:40, 748kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<14:36, 801kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<11:10, 1.05MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:53<07:57, 1.47MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<09:59, 1.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<08:08, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<05:53, 1.97MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<06:53, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<4:42:38, 41.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<3:17:39, 58.5kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:56<2:18:00, 83.5kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<2:15:40, 85.0kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<1:35:54, 120kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<1:07:06, 171kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<50:25, 227kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<36:25, 314kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<25:37, 446kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<20:51, 546kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<15:24, 739kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<10:56, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<10:54, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<09:13, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<06:37, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<07:30, 1.50MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<07:13, 1.56MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<05:36, 2.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<3:56:52, 47.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<2:45:30, 67.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<1:58:21, 94.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<1:23:45, 133kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<58:36, 189kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<44:25, 249kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<32:02, 345kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<22:31, 490kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<18:57, 580kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<14:12, 774kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:13<10:03, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<11:01, 992kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<09:08, 1.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<06:32, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<07:37, 1.43MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<06:09, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<04:55, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<4:31:28, 39.9kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<3:09:48, 56.9kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:18<2:12:29, 81.2kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:18<1:36:33, 111kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<11:17:00, 15.9kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<7:55:47, 22.6kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<5:32:45, 32.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:20<3:52:18, 46.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<2:49:59, 62.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<2:00:24, 88.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<1:24:13, 126kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<1:01:24, 173kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<43:51, 242kB/s]  .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:24<30:45, 344kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<24:37, 428kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<18:47, 561kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<13:17, 791kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<12:00, 873kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<09:17, 1.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<06:38, 1.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<07:47, 1.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<06:22, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<04:37, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<05:58, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<05:13, 1.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<03:49, 2.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<05:16, 1.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<4:30:04, 38.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<3:09:00, 54.4kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:33<2:11:59, 77.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<1:38:25, 104kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<1:09:35, 147kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<48:47, 209kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:35<34:13, 297kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<36:35, 278kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<27:02, 376kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<19:04, 531kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:37<13:28, 750kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<21:43, 465kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<16:37, 607kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<11:47, 854kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:39<08:22, 1.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<23:53, 420kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<17:30, 572kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<12:23, 806kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<11:35, 859kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<08:57, 1.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<06:24, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<35:01, 283kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<4:36:27, 35.9kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<3:13:23, 51.2kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:44<2:15:03, 73.0kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<1:40:00, 98.5kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:46<1:11:16, 138kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<49:52, 197kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<37:24, 261kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<26:57, 362kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<18:56, 514kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<16:32, 587kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<12:46, 759kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<09:05, 1.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:50<06:28, 1.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<49:43, 194kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<35:34, 271kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<25:22, 378kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<4:02:12, 39.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<2:49:18, 56.6kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:53<1:58:09, 80.7kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<1:38:01, 97.3kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<1:09:36, 137kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<48:45, 195kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:55<34:08, 277kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<1:22:00, 115kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<58:20, 162kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<40:48, 231kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<31:33, 298kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<22:50, 411kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<16:02, 583kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<15:04, 619kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<11:55, 782kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<08:29, 1.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:01<06:06, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<11:19, 817kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<08:36, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<06:11, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<05:55, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<3:40:20, 41.8kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<2:34:08, 59.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:04<1:47:41, 85.1kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<1:19:24, 115kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<56:10, 163kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:06<39:21, 232kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:06<27:40, 328kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<27:44, 327kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<20:25, 444kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<14:25, 627kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:08<10:12, 883kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<16:27, 547kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<12:15, 735kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<08:42, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:10<06:13, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<49:10, 182kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<35:12, 254kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<24:43, 360kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:12<17:23, 510kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<32:14, 275kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<23:16, 381kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<16:22, 539kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<17:34, 503kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<3:33:17, 41.4kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<2:29:11, 59.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:15<1:44:10, 84.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<1:17:41, 113kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<55:38, 157kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<38:59, 224kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:17<27:19, 318kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:19<34:25, 253kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<24:15, 358kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:19<17:03, 507kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<16:20, 528kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<12:43, 677kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<09:02, 950kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:21<06:25, 1.33MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<20:17, 422kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<15:19, 558kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<10:50, 786kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:23<07:41, 1.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<21:04, 403kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<15:58, 531kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<11:18, 748kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<09:08, 924kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<3:28:20, 40.5kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<2:25:42, 57.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:26<1:41:43, 82.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<1:15:51, 110kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<54:03, 155kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<37:52, 220kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:28<26:31, 313kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<43:56, 189kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<31:55, 260kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<22:25, 369kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:30<15:45, 523kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<27:16, 302kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<19:48, 416kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<13:58, 587kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<11:44, 696kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<08:55, 915kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<06:20, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<06:51, 1.18MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<05:57, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<04:17, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<04:17, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<3:22:50, 39.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<2:21:40, 56.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:37<1:38:50, 81.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<1:21:24, 98.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<57:37, 139kB/s]   .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<40:18, 197kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<30:26, 260kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<21:57, 361kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<15:25, 511kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<13:17, 591kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<10:26, 753kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<07:28, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<05:31, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<06:00, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<04:50, 1.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:45<03:28, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<05:24, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<04:27, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<03:30, 2.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<3:17:51, 38.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<2:18:25, 55.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<1:36:36, 79.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<1:10:54, 107kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<50:24, 151kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<35:14, 215kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<26:42, 282kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<19:25, 388kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<13:39, 550kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<11:40, 640kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<09:17, 804kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<06:36, 1.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<06:23, 1.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<05:03, 1.46MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<03:38, 2.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<04:31, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:03, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:56, 2.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<05:47, 1.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<3:15:06, 37.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<2:16:17, 53.4kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [02:59<1:35:01, 76.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<1:13:43, 98.0kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<52:31, 138kB/s]   .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<36:44, 196kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<27:21, 262kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<19:42, 363kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:03<13:47, 515kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<13:04, 542kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<10:11, 695kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<07:15, 972kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:05<05:08, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<18:53, 372kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<13:42, 511kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<09:39, 722kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<08:53, 782kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<06:47, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<04:52, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<04:22, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<2:57:17, 39.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<2:03:54, 55.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<1:26:32, 79.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<1:02:58, 109kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<44:57, 152kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<31:32, 216kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:12<22:06, 307kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<19:03, 355kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<14:06, 480kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<09:57, 677kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:14<07:02, 953kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<21:57, 305kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<15:59, 419kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<11:16, 591kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:16<07:57, 835kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<20:35, 322kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<15:21, 432kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<10:48, 611kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<09:11, 715kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<07:09, 917kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<05:05, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<05:33, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<2:37:03, 41.6kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<1:49:42, 59.3kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:21<1:16:28, 84.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<58:20, 111kB/s]   .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<41:45, 155kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<29:12, 220kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<21:53, 292kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<16:07, 396kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<11:20, 560kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<09:27, 669kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<07:24, 852kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<05:15, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<05:25, 1.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<04:28, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:29<03:12, 1.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:03, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<03:46, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<02:43, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<03:29, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<2:21:48, 43.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<1:39:00, 61.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:32<1:08:57, 88.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<56:12, 108kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<40:10, 151kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<28:05, 215kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<21:02, 285kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<15:12, 394kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:36<10:40, 559kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<09:19, 637kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<06:56, 855kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:38<04:54, 1.20MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<05:30, 1.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<04:42, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<03:23, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<03:34, 1.63MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<2:25:54, 39.9kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<1:41:48, 57.0kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:41<1:10:49, 81.3kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<1:12:45, 79.1kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<51:19, 112kB/s]   .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<35:46, 160kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<26:57, 211kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<19:39, 289kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<13:45, 411kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<11:13, 500kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<08:18, 675kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<05:51, 951kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<05:53, 942kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<04:55, 1.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<03:31, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<04:01, 1.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<2:05:12, 44.0kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<1:27:22, 62.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:50<1:00:51, 89.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<47:26, 115kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<33:53, 160kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<23:44, 228kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:52<16:36, 324kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<16:05, 334kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<11:41, 459kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<08:12, 650kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<07:21, 720kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<05:33, 954kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<03:58, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 547M/862M [03:56<02:51, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<08:02, 652kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<05:52, 889kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<04:23, 1.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<2:08:36, 40.4kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<1:29:41, 57.7kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [03:59<1:02:28, 82.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<47:43, 107kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<33:55, 151kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<23:40, 215kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<18:01, 281kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<13:06, 386kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<09:11, 547kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<07:50, 637kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<05:54, 844kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:05<04:10, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<04:28, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<03:35, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<02:35, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<02:58, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<1:55:21, 42.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<1:20:26, 60.4kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:08<55:58, 86.2kB/s]  .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<44:24, 108kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<31:41, 152kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<22:07, 216kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<16:27, 288kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<12:13, 388kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<08:35, 549kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:12<06:02, 775kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<10:15, 456kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<07:32, 618kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:14<05:17, 874kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<05:33, 829kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<04:20, 1.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<03:06, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<03:31, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<03:07, 1.45MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<02:14, 2.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:23, 1.87MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:09, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:34, 2.83MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:16, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:56, 2.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:27, 3.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:22<01:31, 2.85MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<1:44:31, 41.7kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<1:12:42, 59.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<51:31, 83.3kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<36:19, 118kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<25:17, 168kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<18:55, 223kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<13:36, 310kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<09:31, 440kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<07:45, 535kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<05:48, 713kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<04:05, 1.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<04:02, 1.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<03:09, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:31<02:23, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<1:43:32, 39.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<1:12:01, 55.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:32<50:00, 79.5kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<48:28, 82.0kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<34:20, 116kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<23:55, 165kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:34<16:48, 232kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<3:38:29, 17.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<2:32:56, 25.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<1:46:16, 36.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<1:14:54, 51.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<52:55, 72.4kB/s]  .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<36:48, 103kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<26:30, 142kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<19:06, 197kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<13:19, 280kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<10:10, 363kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<07:38, 483kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:42<05:22, 683kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<04:40, 777kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<03:45, 963kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<02:40, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:44<02:42, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<1:25:43, 41.9kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<59:39, 59.7kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:45<41:21, 85.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:45<29:42, 118kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<3:36:45, 16.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<2:31:42, 23.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:47<1:45:02, 33.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<1:14:26, 46.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<52:32, 65.6kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<36:29, 93.6kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<26:11, 129kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<18:32, 182kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:51<12:53, 259kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<09:58, 332kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<07:14, 457kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:53<05:04, 646kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<04:23, 739kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<03:19, 972kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<02:29, 1.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<1:16:33, 41.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<53:27, 59.7kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<37:18, 85.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<26:10, 121kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:56<18:27, 171kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:56<13:01, 241kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<12:53, 243kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<09:23, 333kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<06:36, 471kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [04:58<04:39, 662kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:58<03:19, 922kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<08:20, 368kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<06:12, 493kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<06:07, 500kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:00<04:14, 709kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<05:26, 551kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<04:02, 740kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<02:51, 1.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:02<02:03, 1.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<03:18, 885kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<02:44, 1.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:58, 1.47MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<01:43, 1.67MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<1:08:08, 42.4kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<47:31, 60.5kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<33:02, 86.3kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:05<22:59, 123kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:05<16:54, 167kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<2:40:12, 17.6kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<1:52:05, 25.1kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<1:17:46, 35.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:07<53:58, 51.1kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<40:20, 68.2kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<28:23, 96.7kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<19:44, 138kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:09<13:44, 196kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<11:47, 228kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<08:28, 316kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<06:01, 442kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:11<04:14, 624kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:11<02:59, 876kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<17:04, 153kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<12:06, 215kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<08:27, 305kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:13<05:54, 433kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<06:11, 411kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<04:32, 560kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<03:10, 789kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<03:21, 747kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<56:04, 44.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<39:02, 63.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<27:07, 90.9kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:16<18:49, 129kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<28:08, 86.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<20:00, 122kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<13:56, 173kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:18<09:42, 246kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<08:12, 288kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<06:25, 369kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<04:32, 518kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<03:11, 730kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<02:59, 768kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<02:18, 993kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:39, 1.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<01:11, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<02:02, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<01:38, 1.35MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:11, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<00:51, 2.54MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<03:06, 697kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<02:29, 865kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:46, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<01:16, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<02:00, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<01:34, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:07, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:48, 2.49MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<02:26, 829kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:52, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:19, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<00:56, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<03:09, 618kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<02:21, 828kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:40, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<01:10, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<03:21, 561kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<02:35, 728kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:49, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<01:17, 1.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<02:17, 790kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<01:43, 1.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<01:13, 1.45MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<00:52, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<04:25, 395kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<03:13, 540kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<02:14, 761kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<01:35, 1.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<02:37, 639kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:58, 850kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<01:23, 1.18MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:41<00:59, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:35, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:14, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:52, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:38, 2.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:27, 1.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:08, 1.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:48, 1.85MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<00:35, 2.52MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<04:03, 363kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<02:56, 497kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<02:03, 701kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:46<01:26, 984kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<02:52, 488kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<02:07, 658kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<01:28, 925kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:48<01:02, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<04:57, 269kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<03:33, 374kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<02:28, 528kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:50<01:42, 745kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<02:15, 559kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<01:41, 747kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<01:10, 1.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:52<00:49, 1.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:45, 681kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<01:18, 906kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:55, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:54<00:38, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<14:05, 80.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<09:53, 113kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<06:44, 162kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<04:55, 215kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<03:32, 298kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<02:25, 422kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<01:40, 597kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<02:04, 477kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:34, 627kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<01:05, 881kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:00<00:45, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:20, 688kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:00, 904kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:42, 1.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:02<00:29, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:14, 688kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:55, 916kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:38, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:38, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:31, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:22, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:06<00:15, 2.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<01:12, 593kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:56, 756kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:38, 1.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<00:26, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<05:13, 124kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<03:40, 174kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<02:27, 247kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<01:48, 321kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<01:17, 443kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:52, 625kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:12<00:35, 880kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:58, 525kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:42, 706kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:28, 991kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:14<00:19, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<01:37, 272kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<01:10, 369kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:47, 522kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:30, 737kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:41, 533kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:32, 684kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:21, 955kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:14, 1.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:18, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:13, 1.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:09, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:06, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:16, 826kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:12, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:08, 1.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:05, 2.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:12, 771kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:09, 1.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:05, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:03, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 834kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:02, 1.50MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:00, 2.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:06, 246kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:03, 341kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<29:05:13,  3.82it/s]  0%|          | 724/400000 [00:00<20:19:43,  5.46it/s]  0%|          | 1444/400000 [00:00<14:12:32,  7.79it/s]  1%|          | 2125/400000 [00:00<9:56:03, 11.13it/s]   1%|          | 2823/400000 [00:00<6:56:47, 15.88it/s]  1%|          | 3533/400000 [00:00<4:51:30, 22.67it/s]  1%|          | 4198/400000 [00:00<3:24:00, 32.33it/s]  1%|          | 4878/400000 [00:00<2:22:51, 46.10it/s]  1%|▏         | 5573/400000 [00:01<1:40:06, 65.67it/s]  2%|▏         | 6265/400000 [00:01<1:10:14, 93.43it/s]  2%|▏         | 6989/400000 [00:01<49:20, 132.74it/s]   2%|▏         | 7720/400000 [00:01<34:44, 188.16it/s]  2%|▏         | 8459/400000 [00:01<24:32, 265.89it/s]  2%|▏         | 9190/400000 [00:01<17:24, 374.01it/s]  2%|▏         | 9912/400000 [00:01<12:26, 522.69it/s]  3%|▎         | 10629/400000 [00:01<08:57, 723.76it/s]  3%|▎         | 11370/400000 [00:01<06:31, 992.37it/s]  3%|▎         | 12092/400000 [00:01<04:50, 1335.68it/s]  3%|▎         | 12806/400000 [00:02<03:39, 1766.06it/s]  3%|▎         | 13519/400000 [00:02<02:50, 2269.81it/s]  4%|▎         | 14274/400000 [00:02<02:14, 2872.43it/s]  4%|▎         | 14993/400000 [00:02<01:50, 3477.08it/s]  4%|▍         | 15726/400000 [00:02<01:33, 4127.24it/s]  4%|▍         | 16463/400000 [00:02<01:20, 4754.47it/s]  4%|▍         | 17185/400000 [00:02<01:12, 5295.60it/s]  4%|▍         | 17919/400000 [00:02<01:06, 5777.93it/s]  5%|▍         | 18645/400000 [00:02<01:02, 6127.74it/s]  5%|▍         | 19367/400000 [00:02<01:00, 6282.73it/s]  5%|▌         | 20073/400000 [00:03<00:58, 6466.38it/s]  5%|▌         | 20776/400000 [00:03<00:57, 6598.05it/s]  5%|▌         | 21494/400000 [00:03<00:55, 6761.03it/s]  6%|▌         | 22207/400000 [00:03<00:55, 6867.07it/s]  6%|▌         | 22917/400000 [00:03<00:54, 6933.82it/s]  6%|▌         | 23625/400000 [00:03<00:54, 6969.62it/s]  6%|▌         | 24334/400000 [00:03<00:53, 7003.66it/s]  6%|▋         | 25062/400000 [00:03<00:52, 7082.46it/s]  6%|▋         | 25808/400000 [00:03<00:52, 7189.95it/s]  7%|▋         | 26532/400000 [00:03<00:52, 7164.19it/s]  7%|▋         | 27252/400000 [00:04<00:52, 7112.66it/s]  7%|▋         | 27966/400000 [00:04<00:52, 7059.71it/s]  7%|▋         | 28686/400000 [00:04<00:52, 7099.06it/s]  7%|▋         | 29398/400000 [00:04<00:52, 7096.29it/s]  8%|▊         | 30129/400000 [00:04<00:51, 7158.40it/s]  8%|▊         | 30874/400000 [00:04<00:50, 7241.87it/s]  8%|▊         | 31599/400000 [00:04<00:51, 7176.81it/s]  8%|▊         | 32318/400000 [00:04<00:51, 7071.75it/s]  8%|▊         | 33026/400000 [00:04<00:52, 6996.34it/s]  8%|▊         | 33727/400000 [00:05<00:52, 6953.78it/s]  9%|▊         | 34437/400000 [00:05<00:52, 6995.14it/s]  9%|▉         | 35139/400000 [00:05<00:52, 7000.40it/s]  9%|▉         | 35840/400000 [00:05<00:52, 6994.36it/s]  9%|▉         | 36540/400000 [00:05<00:52, 6957.80it/s]  9%|▉         | 37275/400000 [00:05<00:51, 7068.99it/s] 10%|▉         | 38020/400000 [00:05<00:50, 7176.51it/s] 10%|▉         | 38739/400000 [00:05<00:51, 7068.93it/s] 10%|▉         | 39464/400000 [00:05<00:50, 7122.20it/s] 10%|█         | 40181/400000 [00:05<00:50, 7133.89it/s] 10%|█         | 40899/400000 [00:06<00:50, 7146.89it/s] 10%|█         | 41655/400000 [00:06<00:49, 7265.12it/s] 11%|█         | 42387/400000 [00:06<00:49, 7280.18it/s] 11%|█         | 43136/400000 [00:06<00:48, 7341.31it/s] 11%|█         | 43871/400000 [00:06<00:49, 7226.37it/s] 11%|█         | 44595/400000 [00:06<00:49, 7172.79it/s] 11%|█▏        | 45313/400000 [00:06<00:50, 6985.53it/s] 12%|█▏        | 46058/400000 [00:06<00:49, 7117.04it/s] 12%|█▏        | 46816/400000 [00:06<00:48, 7247.47it/s] 12%|█▏        | 47562/400000 [00:06<00:48, 7309.81it/s] 12%|█▏        | 48321/400000 [00:07<00:47, 7391.15it/s] 12%|█▏        | 49082/400000 [00:07<00:47, 7452.73it/s] 12%|█▏        | 49829/400000 [00:07<00:47, 7367.35it/s] 13%|█▎        | 50567/400000 [00:07<00:47, 7370.70it/s] 13%|█▎        | 51305/400000 [00:07<00:47, 7338.74it/s] 13%|█▎        | 52040/400000 [00:07<00:47, 7323.92it/s] 13%|█▎        | 52784/400000 [00:07<00:47, 7357.91it/s] 13%|█▎        | 53521/400000 [00:07<00:47, 7333.05it/s] 14%|█▎        | 54255/400000 [00:07<00:47, 7214.25it/s] 14%|█▎        | 54978/400000 [00:07<00:48, 7176.14it/s] 14%|█▍        | 55724/400000 [00:08<00:47, 7256.91it/s] 14%|█▍        | 56464/400000 [00:08<00:47, 7297.91it/s] 14%|█▍        | 57195/400000 [00:08<00:47, 7198.58it/s] 14%|█▍        | 57916/400000 [00:08<00:47, 7180.48it/s] 15%|█▍        | 58635/400000 [00:08<00:47, 7176.18it/s] 15%|█▍        | 59367/400000 [00:08<00:47, 7216.98it/s] 15%|█▌        | 60132/400000 [00:08<00:46, 7341.24it/s] 15%|█▌        | 60867/400000 [00:08<00:46, 7236.22it/s] 15%|█▌        | 61603/400000 [00:08<00:46, 7270.97it/s] 16%|█▌        | 62331/400000 [00:08<00:46, 7202.29it/s] 16%|█▌        | 63052/400000 [00:09<00:47, 7050.24it/s] 16%|█▌        | 63759/400000 [00:09<00:47, 7037.14it/s] 16%|█▌        | 64464/400000 [00:09<00:48, 6976.17it/s] 16%|█▋        | 65163/400000 [00:09<00:48, 6927.79it/s] 16%|█▋        | 65894/400000 [00:09<00:47, 7036.47it/s] 17%|█▋        | 66627/400000 [00:09<00:46, 7121.56it/s] 17%|█▋        | 67340/400000 [00:09<00:47, 7037.90it/s] 17%|█▋        | 68046/400000 [00:09<00:47, 7043.37it/s] 17%|█▋        | 68751/400000 [00:09<00:49, 6738.28it/s] 17%|█▋        | 69505/400000 [00:10<00:47, 6960.02it/s] 18%|█▊        | 70231/400000 [00:10<00:46, 7046.04it/s] 18%|█▊        | 70963/400000 [00:10<00:46, 7124.70it/s] 18%|█▊        | 71698/400000 [00:10<00:45, 7188.70it/s] 18%|█▊        | 72431/400000 [00:10<00:45, 7229.25it/s] 18%|█▊        | 73216/400000 [00:10<00:44, 7402.87it/s] 19%|█▊        | 74014/400000 [00:10<00:43, 7564.67it/s] 19%|█▊        | 74814/400000 [00:10<00:42, 7688.20it/s] 19%|█▉        | 75585/400000 [00:10<00:42, 7625.20it/s] 19%|█▉        | 76368/400000 [00:10<00:42, 7685.01it/s] 19%|█▉        | 77172/400000 [00:11<00:41, 7786.46it/s] 19%|█▉        | 77986/400000 [00:11<00:40, 7887.63it/s] 20%|█▉        | 78776/400000 [00:11<00:40, 7872.61it/s] 20%|█▉        | 79576/400000 [00:11<00:40, 7909.84it/s] 20%|██        | 80368/400000 [00:11<00:40, 7837.03it/s] 20%|██        | 81153/400000 [00:11<00:40, 7783.80it/s] 20%|██        | 81932/400000 [00:11<00:40, 7785.31it/s] 21%|██        | 82711/400000 [00:11<00:41, 7730.66it/s] 21%|██        | 83486/400000 [00:11<00:40, 7734.51it/s] 21%|██        | 84260/400000 [00:11<00:40, 7729.27it/s] 21%|██▏       | 85034/400000 [00:12<00:41, 7679.83it/s] 21%|██▏       | 85807/400000 [00:12<00:40, 7694.34it/s] 22%|██▏       | 86621/400000 [00:12<00:40, 7822.23it/s] 22%|██▏       | 87404/400000 [00:12<00:40, 7768.61it/s] 22%|██▏       | 88182/400000 [00:12<00:40, 7769.54it/s] 22%|██▏       | 88976/400000 [00:12<00:39, 7817.69it/s] 22%|██▏       | 89773/400000 [00:12<00:39, 7862.06it/s] 23%|██▎       | 90584/400000 [00:12<00:38, 7933.92it/s] 23%|██▎       | 91378/400000 [00:12<00:39, 7859.94it/s] 23%|██▎       | 92165/400000 [00:12<00:39, 7780.06it/s] 23%|██▎       | 92944/400000 [00:13<00:39, 7684.98it/s] 23%|██▎       | 93730/400000 [00:13<00:39, 7733.02it/s] 24%|██▎       | 94504/400000 [00:13<00:39, 7682.23it/s] 24%|██▍       | 95273/400000 [00:13<00:39, 7632.47it/s] 24%|██▍       | 96037/400000 [00:13<00:39, 7621.83it/s] 24%|██▍       | 96862/400000 [00:13<00:38, 7799.55it/s] 24%|██▍       | 97644/400000 [00:13<00:39, 7679.64it/s] 25%|██▍       | 98414/400000 [00:13<00:39, 7559.07it/s] 25%|██▍       | 99172/400000 [00:13<00:40, 7507.86it/s] 25%|██▍       | 99947/400000 [00:13<00:39, 7577.34it/s] 25%|██▌       | 100706/400000 [00:14<00:39, 7537.98it/s] 25%|██▌       | 101461/400000 [00:14<00:39, 7538.45it/s] 26%|██▌       | 102217/400000 [00:14<00:39, 7544.08it/s] 26%|██▌       | 102972/400000 [00:14<00:39, 7520.52it/s] 26%|██▌       | 103767/400000 [00:14<00:38, 7643.55it/s] 26%|██▌       | 104548/400000 [00:14<00:38, 7692.14it/s] 26%|██▋       | 105351/400000 [00:14<00:37, 7789.22it/s] 27%|██▋       | 106131/400000 [00:14<00:38, 7712.22it/s] 27%|██▋       | 106903/400000 [00:14<00:38, 7582.62it/s] 27%|██▋       | 107681/400000 [00:14<00:38, 7640.13it/s] 27%|██▋       | 108473/400000 [00:15<00:37, 7720.74it/s] 27%|██▋       | 109267/400000 [00:15<00:37, 7782.75it/s] 28%|██▊       | 110046/400000 [00:15<00:37, 7640.55it/s] 28%|██▊       | 110825/400000 [00:15<00:37, 7682.40it/s] 28%|██▊       | 111644/400000 [00:15<00:36, 7827.55it/s] 28%|██▊       | 112436/400000 [00:15<00:36, 7854.97it/s] 28%|██▊       | 113223/400000 [00:15<00:36, 7795.87it/s] 29%|██▊       | 114004/400000 [00:15<00:37, 7679.65it/s] 29%|██▊       | 114773/400000 [00:15<00:37, 7605.23it/s] 29%|██▉       | 115545/400000 [00:15<00:37, 7638.53it/s] 29%|██▉       | 116321/400000 [00:16<00:36, 7672.16it/s] 29%|██▉       | 117120/400000 [00:16<00:36, 7763.81it/s] 29%|██▉       | 117907/400000 [00:16<00:36, 7792.66it/s] 30%|██▉       | 118687/400000 [00:16<00:36, 7642.90it/s] 30%|██▉       | 119453/400000 [00:16<00:36, 7618.41it/s] 30%|███       | 120216/400000 [00:16<00:37, 7557.17it/s] 30%|███       | 121013/400000 [00:16<00:36, 7674.33it/s] 30%|███       | 121782/400000 [00:16<00:36, 7675.84it/s] 31%|███       | 122551/400000 [00:16<00:36, 7638.55it/s] 31%|███       | 123316/400000 [00:17<00:36, 7511.61it/s] 31%|███       | 124078/400000 [00:17<00:36, 7541.10it/s] 31%|███       | 124833/400000 [00:17<00:36, 7539.66it/s] 31%|███▏      | 125597/400000 [00:17<00:36, 7567.36it/s] 32%|███▏      | 126355/400000 [00:17<00:36, 7538.56it/s] 32%|███▏      | 127133/400000 [00:17<00:35, 7607.37it/s] 32%|███▏      | 127895/400000 [00:17<00:35, 7560.00it/s] 32%|███▏      | 128663/400000 [00:17<00:35, 7595.48it/s] 32%|███▏      | 129430/400000 [00:17<00:35, 7616.62it/s] 33%|███▎      | 130200/400000 [00:17<00:35, 7640.84it/s] 33%|███▎      | 130992/400000 [00:18<00:34, 7721.71it/s] 33%|███▎      | 131765/400000 [00:18<00:34, 7694.16it/s] 33%|███▎      | 132537/400000 [00:18<00:34, 7699.90it/s] 33%|███▎      | 133323/400000 [00:18<00:34, 7745.95it/s] 34%|███▎      | 134132/400000 [00:18<00:33, 7843.82it/s] 34%|███▎      | 134950/400000 [00:18<00:33, 7941.67it/s] 34%|███▍      | 135754/400000 [00:18<00:33, 7969.60it/s] 34%|███▍      | 136552/400000 [00:18<00:33, 7911.48it/s] 34%|███▍      | 137361/400000 [00:18<00:32, 7962.64it/s] 35%|███▍      | 138158/400000 [00:18<00:33, 7837.54it/s] 35%|███▍      | 138959/400000 [00:19<00:33, 7887.08it/s] 35%|███▍      | 139749/400000 [00:19<00:33, 7692.56it/s] 35%|███▌      | 140520/400000 [00:19<00:37, 6865.55it/s] 35%|███▌      | 141225/400000 [00:19<00:37, 6884.93it/s] 36%|███▌      | 142029/400000 [00:19<00:35, 7193.68it/s] 36%|███▌      | 142806/400000 [00:19<00:34, 7356.08it/s] 36%|███▌      | 143587/400000 [00:19<00:34, 7484.70it/s] 36%|███▌      | 144359/400000 [00:19<00:33, 7551.05it/s] 36%|███▋      | 145120/400000 [00:19<00:34, 7465.29it/s] 36%|███▋      | 145925/400000 [00:19<00:33, 7629.86it/s] 37%|███▋      | 146715/400000 [00:20<00:32, 7708.01it/s] 37%|███▋      | 147507/400000 [00:20<00:32, 7770.13it/s] 37%|███▋      | 148287/400000 [00:20<00:32, 7632.99it/s] 37%|███▋      | 149053/400000 [00:20<00:33, 7412.17it/s] 37%|███▋      | 149839/400000 [00:20<00:33, 7538.67it/s] 38%|███▊      | 150641/400000 [00:20<00:32, 7676.13it/s] 38%|███▊      | 151450/400000 [00:20<00:31, 7794.92it/s] 38%|███▊      | 152232/400000 [00:20<00:32, 7719.27it/s] 38%|███▊      | 153006/400000 [00:20<00:32, 7643.69it/s] 38%|███▊      | 153772/400000 [00:21<00:32, 7536.06it/s] 39%|███▊      | 154567/400000 [00:21<00:32, 7652.55it/s] 39%|███▉      | 155352/400000 [00:21<00:31, 7707.87it/s] 39%|███▉      | 156132/400000 [00:21<00:31, 7734.78it/s] 39%|███▉      | 156908/400000 [00:21<00:31, 7740.88it/s] 39%|███▉      | 157683/400000 [00:21<00:31, 7693.64it/s] 40%|███▉      | 158488/400000 [00:21<00:30, 7796.47it/s] 40%|███▉      | 159292/400000 [00:21<00:30, 7866.60it/s] 40%|████      | 160080/400000 [00:21<00:31, 7718.02it/s] 40%|████      | 160853/400000 [00:21<00:31, 7701.51it/s] 40%|████      | 161635/400000 [00:22<00:30, 7734.02it/s] 41%|████      | 162409/400000 [00:22<00:30, 7732.88it/s] 41%|████      | 163248/400000 [00:22<00:29, 7916.24it/s] 41%|████      | 164041/400000 [00:22<00:30, 7840.76it/s] 41%|████      | 164827/400000 [00:22<00:30, 7759.93it/s] 41%|████▏     | 165609/400000 [00:22<00:30, 7775.11it/s] 42%|████▏     | 166391/400000 [00:22<00:30, 7786.05it/s] 42%|████▏     | 167196/400000 [00:22<00:29, 7862.48it/s] 42%|████▏     | 167983/400000 [00:22<00:29, 7860.56it/s] 42%|████▏     | 168770/400000 [00:22<00:29, 7810.83it/s] 42%|████▏     | 169552/400000 [00:23<00:29, 7733.83it/s] 43%|████▎     | 170326/400000 [00:23<00:29, 7681.12it/s] 43%|████▎     | 171131/400000 [00:23<00:29, 7786.82it/s] 43%|████▎     | 171911/400000 [00:23<00:29, 7749.93it/s] 43%|████▎     | 172687/400000 [00:23<00:29, 7727.73it/s] 43%|████▎     | 173462/400000 [00:23<00:29, 7729.43it/s] 44%|████▎     | 174255/400000 [00:23<00:28, 7787.29it/s] 44%|████▍     | 175043/400000 [00:23<00:28, 7813.36it/s] 44%|████▍     | 175825/400000 [00:23<00:28, 7738.52it/s] 44%|████▍     | 176616/400000 [00:23<00:28, 7786.52it/s] 44%|████▍     | 177398/400000 [00:24<00:28, 7796.10it/s] 45%|████▍     | 178200/400000 [00:24<00:28, 7860.61it/s] 45%|████▍     | 178987/400000 [00:24<00:29, 7613.13it/s] 45%|████▍     | 179779/400000 [00:24<00:28, 7699.88it/s] 45%|████▌     | 180551/400000 [00:24<00:28, 7619.22it/s] 45%|████▌     | 181315/400000 [00:24<00:28, 7580.73it/s] 46%|████▌     | 182137/400000 [00:24<00:28, 7758.92it/s] 46%|████▌     | 182915/400000 [00:24<00:28, 7707.44it/s] 46%|████▌     | 183688/400000 [00:24<00:28, 7572.49it/s] 46%|████▌     | 184447/400000 [00:24<00:28, 7524.03it/s] 46%|████▋     | 185244/400000 [00:25<00:28, 7651.93it/s] 47%|████▋     | 186064/400000 [00:25<00:27, 7806.60it/s] 47%|████▋     | 186881/400000 [00:25<00:26, 7909.97it/s] 47%|████▋     | 187674/400000 [00:25<00:27, 7705.97it/s] 47%|████▋     | 188447/400000 [00:25<00:27, 7580.68it/s] 47%|████▋     | 189231/400000 [00:25<00:27, 7655.30it/s] 48%|████▊     | 190057/400000 [00:25<00:26, 7825.40it/s] 48%|████▊     | 190842/400000 [00:25<00:26, 7771.30it/s] 48%|████▊     | 191630/400000 [00:25<00:26, 7802.94it/s] 48%|████▊     | 192412/400000 [00:25<00:26, 7715.54it/s] 48%|████▊     | 193231/400000 [00:26<00:26, 7849.37it/s] 49%|████▊     | 194026/400000 [00:26<00:26, 7877.83it/s] 49%|████▊     | 194842/400000 [00:26<00:25, 7959.53it/s] 49%|████▉     | 195639/400000 [00:26<00:25, 7878.48it/s] 49%|████▉     | 196428/400000 [00:26<00:26, 7591.17it/s] 49%|████▉     | 197215/400000 [00:26<00:26, 7671.16it/s] 49%|████▉     | 197991/400000 [00:26<00:26, 7697.09it/s] 50%|████▉     | 198765/400000 [00:26<00:26, 7709.23it/s] 50%|████▉     | 199537/400000 [00:26<00:26, 7582.64it/s] 50%|█████     | 200297/400000 [00:27<00:26, 7513.57it/s] 50%|█████     | 201050/400000 [00:27<00:26, 7483.34it/s] 50%|█████     | 201817/400000 [00:27<00:26, 7537.75it/s] 51%|█████     | 202581/400000 [00:27<00:26, 7566.16it/s] 51%|█████     | 203339/400000 [00:27<00:26, 7527.23it/s] 51%|█████     | 204093/400000 [00:27<00:26, 7476.15it/s] 51%|█████     | 204844/400000 [00:27<00:26, 7484.26it/s] 51%|█████▏    | 205607/400000 [00:27<00:25, 7526.15it/s] 52%|█████▏    | 206378/400000 [00:27<00:25, 7579.33it/s] 52%|█████▏    | 207137/400000 [00:27<00:25, 7548.44it/s] 52%|█████▏    | 207893/400000 [00:28<00:25, 7443.20it/s] 52%|█████▏    | 208638/400000 [00:28<00:25, 7419.06it/s] 52%|█████▏    | 209381/400000 [00:28<00:25, 7419.22it/s] 53%|█████▎    | 210165/400000 [00:28<00:25, 7539.95it/s] 53%|█████▎    | 210974/400000 [00:28<00:24, 7696.14it/s] 53%|█████▎    | 211762/400000 [00:28<00:24, 7749.99it/s] 53%|█████▎    | 212538/400000 [00:28<00:24, 7722.34it/s] 53%|█████▎    | 213311/400000 [00:28<00:24, 7655.00it/s] 54%|█████▎    | 214097/400000 [00:28<00:24, 7714.20it/s] 54%|█████▎    | 214889/400000 [00:28<00:23, 7774.11it/s] 54%|█████▍    | 215667/400000 [00:29<00:23, 7737.20it/s] 54%|█████▍    | 216447/400000 [00:29<00:23, 7755.64it/s] 54%|█████▍    | 217225/400000 [00:29<00:23, 7761.19it/s] 55%|█████▍    | 218002/400000 [00:29<00:24, 7530.25it/s] 55%|█████▍    | 218763/400000 [00:29<00:23, 7552.05it/s] 55%|█████▍    | 219520/400000 [00:29<00:24, 7478.85it/s] 55%|█████▌    | 220269/400000 [00:29<00:24, 7439.50it/s] 55%|█████▌    | 221037/400000 [00:29<00:23, 7509.99it/s] 55%|█████▌    | 221789/400000 [00:29<00:23, 7438.55it/s] 56%|█████▌    | 222556/400000 [00:29<00:23, 7506.15it/s] 56%|█████▌    | 223327/400000 [00:30<00:23, 7563.63it/s] 56%|█████▌    | 224105/400000 [00:30<00:23, 7626.70it/s] 56%|█████▌    | 224884/400000 [00:30<00:22, 7672.44it/s] 56%|█████▋    | 225652/400000 [00:30<00:22, 7655.13it/s] 57%|█████▋    | 226418/400000 [00:30<00:23, 7532.77it/s] 57%|█████▋    | 227182/400000 [00:30<00:22, 7563.54it/s] 57%|█████▋    | 227956/400000 [00:30<00:22, 7615.30it/s] 57%|█████▋    | 228737/400000 [00:30<00:22, 7671.44it/s] 57%|█████▋    | 229505/400000 [00:30<00:22, 7527.66it/s] 58%|█████▊    | 230259/400000 [00:30<00:22, 7480.76it/s] 58%|█████▊    | 231048/400000 [00:31<00:22, 7596.68it/s] 58%|█████▊    | 231848/400000 [00:31<00:21, 7712.47it/s] 58%|█████▊    | 232639/400000 [00:31<00:21, 7770.39it/s] 58%|█████▊    | 233417/400000 [00:31<00:22, 7457.50it/s] 59%|█████▊    | 234167/400000 [00:31<00:22, 7385.80it/s] 59%|█████▊    | 234922/400000 [00:31<00:22, 7432.26it/s] 59%|█████▉    | 235678/400000 [00:31<00:22, 7468.39it/s] 59%|█████▉    | 236452/400000 [00:31<00:21, 7546.53it/s] 59%|█████▉    | 237208/400000 [00:31<00:21, 7474.95it/s] 59%|█████▉    | 237957/400000 [00:31<00:21, 7407.39it/s] 60%|█████▉    | 238699/400000 [00:32<00:21, 7373.39it/s] 60%|█████▉    | 239450/400000 [00:32<00:21, 7411.20it/s] 60%|██████    | 240212/400000 [00:32<00:21, 7470.03it/s] 60%|██████    | 241005/400000 [00:32<00:20, 7601.15it/s] 60%|██████    | 241786/400000 [00:32<00:20, 7660.71it/s] 61%|██████    | 242580/400000 [00:32<00:20, 7741.23it/s] 61%|██████    | 243360/400000 [00:32<00:20, 7756.49it/s] 61%|██████    | 244172/400000 [00:32<00:19, 7860.85it/s] 61%|██████    | 244959/400000 [00:32<00:20, 7692.98it/s] 61%|██████▏   | 245730/400000 [00:33<00:20, 7558.52it/s] 62%|██████▏   | 246530/400000 [00:33<00:19, 7684.23it/s] 62%|██████▏   | 247320/400000 [00:33<00:19, 7745.78it/s] 62%|██████▏   | 248100/400000 [00:33<00:19, 7760.25it/s] 62%|██████▏   | 248877/400000 [00:33<00:19, 7716.19it/s] 62%|██████▏   | 249650/400000 [00:33<00:19, 7623.68it/s] 63%|██████▎   | 250414/400000 [00:33<00:19, 7618.48it/s] 63%|██████▎   | 251177/400000 [00:33<00:20, 7255.25it/s] 63%|██████▎   | 251923/400000 [00:33<00:20, 7314.30it/s] 63%|██████▎   | 252711/400000 [00:33<00:19, 7474.16it/s] 63%|██████▎   | 253462/400000 [00:34<00:19, 7454.77it/s] 64%|██████▎   | 254235/400000 [00:34<00:19, 7533.39it/s] 64%|██████▍   | 255006/400000 [00:34<00:19, 7584.36it/s] 64%|██████▍   | 255766/400000 [00:34<00:19, 7529.42it/s] 64%|██████▍   | 256539/400000 [00:34<00:18, 7588.38it/s] 64%|██████▍   | 257308/400000 [00:34<00:18, 7616.91it/s] 65%|██████▍   | 258082/400000 [00:34<00:18, 7653.23it/s] 65%|██████▍   | 258848/400000 [00:34<00:18, 7550.87it/s] 65%|██████▍   | 259604/400000 [00:34<00:18, 7394.07it/s] 65%|██████▌   | 260345/400000 [00:34<00:18, 7351.05it/s] 65%|██████▌   | 261097/400000 [00:35<00:18, 7398.82it/s] 65%|██████▌   | 261881/400000 [00:35<00:18, 7523.95it/s] 66%|██████▌   | 262635/400000 [00:35<00:18, 7344.01it/s] 66%|██████▌   | 263396/400000 [00:35<00:18, 7420.08it/s] 66%|██████▌   | 264167/400000 [00:35<00:18, 7503.91it/s] 66%|██████▌   | 264919/400000 [00:35<00:18, 7418.43it/s] 66%|██████▋   | 265667/400000 [00:35<00:18, 7433.92it/s] 67%|██████▋   | 266414/400000 [00:35<00:17, 7442.18it/s] 67%|██████▋   | 267159/400000 [00:35<00:17, 7384.79it/s] 67%|██████▋   | 267924/400000 [00:35<00:17, 7462.01it/s] 67%|██████▋   | 268674/400000 [00:36<00:17, 7472.59it/s] 67%|██████▋   | 269442/400000 [00:36<00:17, 7531.76it/s] 68%|██████▊   | 270204/400000 [00:36<00:17, 7556.79it/s] 68%|██████▊   | 270982/400000 [00:36<00:16, 7620.25it/s] 68%|██████▊   | 271774/400000 [00:36<00:16, 7707.67it/s] 68%|██████▊   | 272546/400000 [00:36<00:16, 7657.70it/s] 68%|██████▊   | 273313/400000 [00:36<00:16, 7579.77it/s] 69%|██████▊   | 274072/400000 [00:36<00:16, 7512.54it/s] 69%|██████▊   | 274839/400000 [00:36<00:16, 7556.89it/s] 69%|██████▉   | 275611/400000 [00:36<00:16, 7602.81it/s] 69%|██████▉   | 276372/400000 [00:37<00:16, 7547.83it/s] 69%|██████▉   | 277134/400000 [00:37<00:16, 7566.52it/s] 69%|██████▉   | 277891/400000 [00:37<00:16, 7495.82it/s] 70%|██████▉   | 278641/400000 [00:37<00:16, 7481.89it/s] 70%|██████▉   | 279392/400000 [00:37<00:16, 7487.65it/s] 70%|███████   | 280141/400000 [00:37<00:16, 7472.89it/s] 70%|███████   | 280903/400000 [00:37<00:15, 7512.83it/s] 70%|███████   | 281655/400000 [00:37<00:15, 7489.10it/s] 71%|███████   | 282430/400000 [00:37<00:15, 7561.83it/s] 71%|███████   | 283195/400000 [00:37<00:15, 7585.36it/s] 71%|███████   | 283954/400000 [00:38<00:15, 7514.42it/s] 71%|███████   | 284734/400000 [00:38<00:15, 7597.56it/s] 71%|███████▏  | 285508/400000 [00:38<00:14, 7637.48it/s] 72%|███████▏  | 286273/400000 [00:38<00:14, 7614.62it/s] 72%|███████▏  | 287055/400000 [00:38<00:14, 7673.54it/s] 72%|███████▏  | 287846/400000 [00:38<00:14, 7741.23it/s] 72%|███████▏  | 288644/400000 [00:38<00:14, 7809.07it/s] 72%|███████▏  | 289426/400000 [00:38<00:14, 7800.19it/s] 73%|███████▎  | 290207/400000 [00:38<00:14, 7707.02it/s] 73%|███████▎  | 290979/400000 [00:38<00:14, 7690.86it/s] 73%|███████▎  | 291763/400000 [00:39<00:13, 7734.22it/s] 73%|███████▎  | 292575/400000 [00:39<00:13, 7844.55it/s] 73%|███████▎  | 293361/400000 [00:39<00:13, 7832.18it/s] 74%|███████▎  | 294145/400000 [00:39<00:13, 7797.94it/s] 74%|███████▎  | 294939/400000 [00:39<00:13, 7838.25it/s] 74%|███████▍  | 295725/400000 [00:39<00:13, 7842.95it/s] 74%|███████▍  | 296510/400000 [00:39<00:13, 7756.43it/s] 74%|███████▍  | 297286/400000 [00:39<00:13, 7684.02it/s] 75%|███████▍  | 298057/400000 [00:39<00:13, 7690.06it/s] 75%|███████▍  | 298827/400000 [00:40<00:13, 7522.18it/s] 75%|███████▍  | 299594/400000 [00:40<00:13, 7564.04it/s] 75%|███████▌  | 300392/400000 [00:40<00:12, 7682.37it/s] 75%|███████▌  | 301175/400000 [00:40<00:12, 7723.68it/s] 75%|███████▌  | 301967/400000 [00:40<00:12, 7779.10it/s] 76%|███████▌  | 302746/400000 [00:40<00:12, 7591.63it/s] 76%|███████▌  | 303507/400000 [00:40<00:12, 7503.63it/s] 76%|███████▌  | 304309/400000 [00:40<00:12, 7651.12it/s] 76%|███████▋  | 305076/400000 [00:40<00:12, 7596.66it/s] 76%|███████▋  | 305871/400000 [00:40<00:12, 7699.27it/s] 77%|███████▋  | 306698/400000 [00:41<00:11, 7861.58it/s] 77%|███████▋  | 307486/400000 [00:41<00:11, 7763.07it/s] 77%|███████▋  | 308287/400000 [00:41<00:11, 7832.32it/s] 77%|███████▋  | 309079/400000 [00:41<00:11, 7855.34it/s] 77%|███████▋  | 309866/400000 [00:41<00:11, 7800.19it/s] 78%|███████▊  | 310665/400000 [00:41<00:11, 7855.31it/s] 78%|███████▊  | 311452/400000 [00:41<00:11, 7834.36it/s] 78%|███████▊  | 312242/400000 [00:41<00:11, 7853.85it/s] 78%|███████▊  | 313028/400000 [00:41<00:11, 7735.39it/s] 78%|███████▊  | 313803/400000 [00:41<00:11, 7705.87it/s] 79%|███████▊  | 314575/400000 [00:42<00:11, 7653.00it/s] 79%|███████▉  | 315346/400000 [00:42<00:11, 7669.76it/s] 79%|███████▉  | 316170/400000 [00:42<00:10, 7831.37it/s] 79%|███████▉  | 316955/400000 [00:42<00:10, 7776.61it/s] 79%|███████▉  | 317763/400000 [00:42<00:10, 7862.57it/s] 80%|███████▉  | 318551/400000 [00:42<00:10, 7855.31it/s] 80%|███████▉  | 319338/400000 [00:42<00:10, 7829.33it/s] 80%|████████  | 320141/400000 [00:42<00:10, 7888.32it/s] 80%|████████  | 320931/400000 [00:42<00:10, 7886.43it/s] 80%|████████  | 321720/400000 [00:42<00:09, 7830.69it/s] 81%|████████  | 322504/400000 [00:43<00:09, 7762.89it/s] 81%|████████  | 323281/400000 [00:43<00:09, 7697.92it/s] 81%|████████  | 324052/400000 [00:43<00:10, 7592.06it/s] 81%|████████  | 324812/400000 [00:43<00:09, 7567.47it/s] 81%|████████▏ | 325618/400000 [00:43<00:09, 7707.15it/s] 82%|████████▏ | 326392/400000 [00:43<00:09, 7715.77it/s] 82%|████████▏ | 327184/400000 [00:43<00:09, 7775.32it/s] 82%|████████▏ | 327982/400000 [00:43<00:09, 7831.94it/s] 82%|████████▏ | 328766/400000 [00:43<00:09, 7647.48it/s] 82%|████████▏ | 329542/400000 [00:43<00:09, 7679.37it/s] 83%|████████▎ | 330311/400000 [00:44<00:09, 7642.25it/s] 83%|████████▎ | 331076/400000 [00:44<00:09, 7489.31it/s] 83%|████████▎ | 331845/400000 [00:44<00:09, 7546.49it/s] 83%|████████▎ | 332601/400000 [00:44<00:08, 7531.73it/s] 83%|████████▎ | 333355/400000 [00:44<00:08, 7507.25it/s] 84%|████████▎ | 334150/400000 [00:44<00:08, 7633.87it/s] 84%|████████▎ | 334917/400000 [00:44<00:08, 7644.17it/s] 84%|████████▍ | 335683/400000 [00:44<00:08, 7528.27it/s] 84%|████████▍ | 336437/400000 [00:44<00:08, 7460.35it/s] 84%|████████▍ | 337211/400000 [00:44<00:08, 7540.49it/s] 84%|████████▍ | 337975/400000 [00:45<00:08, 7569.69it/s] 85%|████████▍ | 338733/400000 [00:45<00:08, 7446.40it/s] 85%|████████▍ | 339479/400000 [00:45<00:08, 7432.01it/s] 85%|████████▌ | 340252/400000 [00:45<00:07, 7516.57it/s] 85%|████████▌ | 341044/400000 [00:45<00:07, 7631.62it/s] 85%|████████▌ | 341821/400000 [00:45<00:07, 7671.17it/s] 86%|████████▌ | 342593/400000 [00:45<00:07, 7685.14it/s] 86%|████████▌ | 343384/400000 [00:45<00:07, 7750.02it/s] 86%|████████▌ | 344160/400000 [00:45<00:07, 7608.56it/s] 86%|████████▌ | 344923/400000 [00:46<00:07, 7611.98it/s] 86%|████████▋ | 345685/400000 [00:46<00:07, 7386.16it/s] 87%|████████▋ | 346449/400000 [00:46<00:07, 7459.86it/s] 87%|████████▋ | 347209/400000 [00:46<00:07, 7499.01it/s] 87%|████████▋ | 347970/400000 [00:46<00:06, 7528.47it/s] 87%|████████▋ | 348751/400000 [00:46<00:06, 7609.09it/s] 87%|████████▋ | 349526/400000 [00:46<00:06, 7648.81it/s] 88%|████████▊ | 350292/400000 [00:46<00:06, 7591.18it/s] 88%|████████▊ | 351061/400000 [00:46<00:06, 7618.40it/s] 88%|████████▊ | 351824/400000 [00:46<00:06, 7603.58it/s] 88%|████████▊ | 352592/400000 [00:47<00:06, 7625.94it/s] 88%|████████▊ | 353363/400000 [00:47<00:06, 7648.28it/s] 89%|████████▊ | 354128/400000 [00:47<00:06, 7600.92it/s] 89%|████████▊ | 354907/400000 [00:47<00:05, 7654.53it/s] 89%|████████▉ | 355673/400000 [00:47<00:05, 7637.76it/s] 89%|████████▉ | 356443/400000 [00:47<00:05, 7653.92it/s] 89%|████████▉ | 357209/400000 [00:47<00:05, 7644.88it/s] 89%|████████▉ | 357980/400000 [00:47<00:05, 7663.27it/s] 90%|████████▉ | 358747/400000 [00:47<00:05, 7531.54it/s] 90%|████████▉ | 359502/400000 [00:47<00:05, 7534.64it/s] 90%|█████████ | 360258/400000 [00:48<00:05, 7542.01it/s] 90%|█████████ | 361013/400000 [00:48<00:05, 7529.23it/s] 90%|█████████ | 361767/400000 [00:48<00:05, 7436.15it/s] 91%|█████████ | 362512/400000 [00:48<00:05, 7395.93it/s] 91%|█████████ | 363252/400000 [00:48<00:04, 7368.48it/s] 91%|█████████ | 364013/400000 [00:48<00:04, 7438.09it/s] 91%|█████████ | 364758/400000 [00:48<00:04, 7406.84it/s] 91%|█████████▏| 365512/400000 [00:48<00:04, 7444.23it/s] 92%|█████████▏| 366285/400000 [00:48<00:04, 7525.36it/s] 92%|█████████▏| 367038/400000 [00:48<00:04, 7477.10it/s] 92%|█████████▏| 367817/400000 [00:49<00:04, 7567.35it/s] 92%|█████████▏| 368594/400000 [00:49<00:04, 7625.37it/s] 92%|█████████▏| 369357/400000 [00:49<00:04, 7588.58it/s] 93%|█████████▎| 370117/400000 [00:49<00:04, 7448.52it/s] 93%|█████████▎| 370863/400000 [00:49<00:03, 7407.66it/s] 93%|█████████▎| 371605/400000 [00:49<00:03, 7384.66it/s] 93%|█████████▎| 372353/400000 [00:49<00:03, 7411.95it/s] 93%|█████████▎| 373124/400000 [00:49<00:03, 7496.53it/s] 93%|█████████▎| 373941/400000 [00:49<00:03, 7684.50it/s] 94%|█████████▎| 374711/400000 [00:49<00:03, 7628.66it/s] 94%|█████████▍| 375476/400000 [00:50<00:03, 7565.69it/s] 94%|█████████▍| 376270/400000 [00:50<00:03, 7670.75it/s] 94%|█████████▍| 377039/400000 [00:50<00:03, 7622.75it/s] 94%|█████████▍| 377803/400000 [00:50<00:02, 7520.56it/s] 95%|█████████▍| 378572/400000 [00:50<00:02, 7568.99it/s] 95%|█████████▍| 379377/400000 [00:50<00:02, 7707.03it/s] 95%|█████████▌| 380149/400000 [00:50<00:02, 7663.83it/s] 95%|█████████▌| 380924/400000 [00:50<00:02, 7687.77it/s] 95%|█████████▌| 381708/400000 [00:50<00:02, 7730.06it/s] 96%|█████████▌| 382482/400000 [00:50<00:02, 7542.54it/s] 96%|█████████▌| 383258/400000 [00:51<00:02, 7605.62it/s] 96%|█████████▌| 384020/400000 [00:51<00:02, 7143.34it/s] 96%|█████████▌| 384741/400000 [00:51<00:02, 7105.56it/s] 96%|█████████▋| 385480/400000 [00:51<00:02, 7186.56it/s] 97%|█████████▋| 386222/400000 [00:51<00:01, 7252.36it/s] 97%|█████████▋| 386995/400000 [00:51<00:01, 7389.29it/s] 97%|█████████▋| 387737/400000 [00:51<00:01, 7389.01it/s] 97%|█████████▋| 388486/400000 [00:51<00:01, 7419.00it/s] 97%|█████████▋| 389250/400000 [00:51<00:01, 7482.75it/s] 98%|█████████▊| 390006/400000 [00:52<00:01, 7505.37it/s] 98%|█████████▊| 390769/400000 [00:52<00:01, 7541.97it/s] 98%|█████████▊| 391580/400000 [00:52<00:01, 7702.19it/s] 98%|█████████▊| 392352/400000 [00:52<00:01, 7612.79it/s] 98%|█████████▊| 393131/400000 [00:52<00:00, 7662.83it/s] 98%|█████████▊| 393899/400000 [00:52<00:00, 7652.56it/s] 99%|█████████▊| 394669/400000 [00:52<00:00, 7665.41it/s] 99%|█████████▉| 395443/400000 [00:52<00:00, 7685.38it/s] 99%|█████████▉| 396212/400000 [00:52<00:00, 7491.14it/s] 99%|█████████▉| 396974/400000 [00:52<00:00, 7528.15it/s] 99%|█████████▉| 397734/400000 [00:53<00:00, 7547.46it/s]100%|█████████▉| 398500/400000 [00:53<00:00, 7579.10it/s]100%|█████████▉| 399275/400000 [00:53<00:00, 7628.53it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7503.31it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f620daa1fd0>, <torchtext.data.dataset.TabularDataset object at 0x7f620daa1208>, <torchtext.vocab.Vocab object at 0x7f620daa1128>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 22.43 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 43.47 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 20.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 20.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.29 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.29 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.10 file/s]2020-06-26 00:19:40.069352: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-26 00:19:40.073583: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394445000 Hz
2020-06-26 00:19:40.073771: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563a939c8980 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-26 00:19:40.073790: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:01<?, ?it/s]  0%|          | 49152/9912422 [00:01<00:32, 302677.86it/s]  2%|▏         | 212992/9912422 [00:01<00:24, 391668.81it/s]  9%|▉         | 876544/9912422 [00:01<00:16, 539504.38it/s] 35%|███▌      | 3473408/9912422 [00:01<00:08, 762195.37it/s] 77%|███████▋  | 7659520/9912422 [00:01<00:02, 1078516.45it/s]9920512it [00:01, 5235739.02it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 141456.17it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 293909.11it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 379773.13it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 526726.29it/s]1654784it [00:00, 2633907.33it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 51342.27it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
