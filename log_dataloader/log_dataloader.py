
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7b69ec2ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7b69ec2ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f7bd51811e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f7bd51811e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f7bf34c9e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f7bf34c9e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7b824a9488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7b824a9488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7b824a9488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 155684.21it/s] 79%|███████▉  | 7839744/9912422 [00:00<00:09, 222216.48it/s]9920512it [00:00, 44297057.02it/s]                           
0it [00:00, ?it/s]32768it [00:00, 587840.85it/s]
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 945817.54it/s]1654784it [00:00, 12244598.78it/s]                         
0it [00:00, ?it/s]8192it [00:00, 226759.53it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7b7f963e10>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7b7f980668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f7b824a90d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f7b824a90d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f7b824a90d0> , (data_info, **args) 

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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:48,  3.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:48,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:48,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:48,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:47,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:47,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:47,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:46,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:33,  4.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:33,  4.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:33,  4.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:32,  4.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:32,  4.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:32,  4.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:22,  6.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:22,  6.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:22,  6.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:22,  6.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:22,  6.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:15,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 11.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 11.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 11.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 15.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 15.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 15.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 20.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 20.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 20.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 20.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 20.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 38.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 38.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 38.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 38.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 38.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 38.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 38.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 45.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 45.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 45.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 45.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 45.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 51.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 51.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 56.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 56.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 56.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 56.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 56.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 56.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 56.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 56.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 61.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 61.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 61.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 61.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 61.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 61.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 61.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 61.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 63.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 63.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 63.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 63.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 63.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 63.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 63.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 63.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 63.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 65.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 65.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 65.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 65.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 69.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 69.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.73s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.73s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.73s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.71 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.73s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 69.71 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.78 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ url]
0 examples [00:00, ? examples/s]2020-06-23 18:09:53.944869: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-23 18:09:53.958725: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-06-23 18:09:53.958912: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fc3e611f00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-23 18:09:53.958932: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
50 examples [00:00, 499.55 examples/s]122 examples [00:00, 548.87 examples/s]212 examples [00:00, 620.35 examples/s]305 examples [00:00, 688.40 examples/s]403 examples [00:00, 755.64 examples/s]505 examples [00:00, 818.40 examples/s]602 examples [00:00, 858.43 examples/s]706 examples [00:00, 904.14 examples/s]807 examples [00:00, 932.14 examples/s]910 examples [00:01, 958.01 examples/s]1013 examples [00:01, 976.57 examples/s]1112 examples [00:01, 969.28 examples/s]1210 examples [00:01, 964.64 examples/s]1310 examples [00:01, 974.51 examples/s]1410 examples [00:01, 980.72 examples/s]1509 examples [00:01, 975.27 examples/s]1607 examples [00:01, 957.12 examples/s]1703 examples [00:01, 928.26 examples/s]1804 examples [00:01, 949.44 examples/s]1904 examples [00:02, 963.61 examples/s]2001 examples [00:02, 964.55 examples/s]2100 examples [00:02, 971.52 examples/s]2198 examples [00:02, 973.59 examples/s]2296 examples [00:02, 959.85 examples/s]2393 examples [00:02, 937.43 examples/s]2487 examples [00:02, 934.74 examples/s]2586 examples [00:02, 950.10 examples/s]2684 examples [00:02, 957.00 examples/s]2780 examples [00:02, 957.74 examples/s]2876 examples [00:03, 938.98 examples/s]2971 examples [00:03, 932.09 examples/s]3065 examples [00:03, 916.76 examples/s]3163 examples [00:03, 932.80 examples/s]3260 examples [00:03, 942.60 examples/s]3355 examples [00:03, 894.64 examples/s]3446 examples [00:03, 898.86 examples/s]3538 examples [00:03, 902.08 examples/s]3629 examples [00:03, 886.70 examples/s]3724 examples [00:03, 902.28 examples/s]3822 examples [00:04, 921.69 examples/s]3915 examples [00:04, 916.99 examples/s]4007 examples [00:04, 894.38 examples/s]4097 examples [00:04, 877.13 examples/s]4185 examples [00:04, 847.77 examples/s]4279 examples [00:04, 871.23 examples/s]4369 examples [00:04, 876.45 examples/s]4470 examples [00:04, 911.90 examples/s]4571 examples [00:04, 938.25 examples/s]4673 examples [00:05, 960.52 examples/s]4774 examples [00:05, 974.73 examples/s]4873 examples [00:05, 977.15 examples/s]4972 examples [00:05, 974.94 examples/s]5072 examples [00:05, 980.03 examples/s]5171 examples [00:05, 974.54 examples/s]5271 examples [00:05, 981.05 examples/s]5370 examples [00:05, 983.63 examples/s]5471 examples [00:05, 990.00 examples/s]5571 examples [00:05, 988.76 examples/s]5671 examples [00:06, 991.54 examples/s]5771 examples [00:06, 981.25 examples/s]5870 examples [00:06, 979.18 examples/s]5972 examples [00:06, 988.42 examples/s]6073 examples [00:06, 993.80 examples/s]6173 examples [00:06, 992.47 examples/s]6273 examples [00:06, 986.27 examples/s]6372 examples [00:06, 986.37 examples/s]6471 examples [00:06, 986.59 examples/s]6570 examples [00:06, 982.00 examples/s]6669 examples [00:07, 980.76 examples/s]6770 examples [00:07, 987.20 examples/s]6869 examples [00:07, 983.12 examples/s]6968 examples [00:07, 983.36 examples/s]7067 examples [00:07, 985.31 examples/s]7166 examples [00:07, 977.07 examples/s]7266 examples [00:07, 978.87 examples/s]7364 examples [00:07, 941.60 examples/s]7459 examples [00:07, 927.27 examples/s]7556 examples [00:07, 938.29 examples/s]7651 examples [00:08, 930.44 examples/s]7745 examples [00:08, 928.24 examples/s]7838 examples [00:08, 925.14 examples/s]7937 examples [00:08, 943.56 examples/s]8034 examples [00:08, 948.56 examples/s]8134 examples [00:08, 960.78 examples/s]8235 examples [00:08, 973.71 examples/s]8333 examples [00:08, 967.02 examples/s]8430 examples [00:08, 952.51 examples/s]8526 examples [00:09, 945.96 examples/s]8624 examples [00:09, 954.90 examples/s]8724 examples [00:09, 967.88 examples/s]8826 examples [00:09, 982.11 examples/s]8927 examples [00:09, 988.23 examples/s]9026 examples [00:09, 983.05 examples/s]9126 examples [00:09, 987.15 examples/s]9225 examples [00:09, 965.40 examples/s]9322 examples [00:09, 949.57 examples/s]9421 examples [00:09, 961.00 examples/s]9520 examples [00:10, 968.85 examples/s]9621 examples [00:10, 980.80 examples/s]9721 examples [00:10, 985.55 examples/s]9820 examples [00:10, 981.26 examples/s]9919 examples [00:10, 983.28 examples/s]10018 examples [00:10, 941.47 examples/s]10118 examples [00:10, 957.91 examples/s]10219 examples [00:10, 972.58 examples/s]10317 examples [00:10, 944.54 examples/s]10418 examples [00:10, 962.12 examples/s]10518 examples [00:11, 972.46 examples/s]10621 examples [00:11, 986.37 examples/s]10721 examples [00:11, 989.46 examples/s]10821 examples [00:11, 966.72 examples/s]10918 examples [00:11, 953.93 examples/s]11014 examples [00:11, 946.45 examples/s]11115 examples [00:11, 962.33 examples/s]11217 examples [00:11, 976.98 examples/s]11315 examples [00:11, 977.69 examples/s]11413 examples [00:11, 956.28 examples/s]11509 examples [00:12, 952.68 examples/s]11606 examples [00:12, 955.59 examples/s]11705 examples [00:12, 965.43 examples/s]11802 examples [00:12, 951.31 examples/s]11900 examples [00:12, 957.37 examples/s]11996 examples [00:12, 953.83 examples/s]12093 examples [00:12, 956.93 examples/s]12191 examples [00:12, 961.79 examples/s]12288 examples [00:12, 899.14 examples/s]12382 examples [00:13, 910.99 examples/s]12476 examples [00:13, 910.76 examples/s]12570 examples [00:13, 916.61 examples/s]12668 examples [00:13, 934.07 examples/s]12762 examples [00:13, 927.15 examples/s]12857 examples [00:13, 931.93 examples/s]12958 examples [00:13, 953.83 examples/s]13058 examples [00:13, 966.32 examples/s]13155 examples [00:13, 920.26 examples/s]13254 examples [00:13, 939.13 examples/s]13354 examples [00:14, 955.73 examples/s]13452 examples [00:14, 960.74 examples/s]13552 examples [00:14, 969.95 examples/s]13650 examples [00:14, 972.57 examples/s]13748 examples [00:14, 958.62 examples/s]13845 examples [00:14, 929.50 examples/s]13939 examples [00:14, 920.11 examples/s]14032 examples [00:14, 907.41 examples/s]14123 examples [00:14, 898.73 examples/s]14214 examples [00:14, 880.36 examples/s]14308 examples [00:15, 895.82 examples/s]14407 examples [00:15, 921.03 examples/s]14503 examples [00:15, 932.27 examples/s]14598 examples [00:15, 933.94 examples/s]14692 examples [00:15, 915.46 examples/s]14787 examples [00:15, 923.18 examples/s]14884 examples [00:15, 936.28 examples/s]14982 examples [00:15, 947.94 examples/s]15082 examples [00:15, 960.75 examples/s]15182 examples [00:15, 970.13 examples/s]15280 examples [00:16, 957.71 examples/s]15376 examples [00:16, 945.77 examples/s]15471 examples [00:16, 937.90 examples/s]15565 examples [00:16, 936.38 examples/s]15659 examples [00:16, 926.00 examples/s]15759 examples [00:16, 945.46 examples/s]15855 examples [00:16, 947.57 examples/s]15950 examples [00:16, 939.51 examples/s]16045 examples [00:16, 899.36 examples/s]16137 examples [00:17, 904.92 examples/s]16237 examples [00:17, 931.23 examples/s]16335 examples [00:17, 945.22 examples/s]16436 examples [00:17, 961.09 examples/s]16535 examples [00:17, 969.31 examples/s]16635 examples [00:17, 977.55 examples/s]16735 examples [00:17, 983.12 examples/s]16834 examples [00:17, 967.25 examples/s]16931 examples [00:17, 932.71 examples/s]17025 examples [00:17, 904.76 examples/s]17116 examples [00:18, 903.10 examples/s]17216 examples [00:18, 929.58 examples/s]17319 examples [00:18, 955.54 examples/s]17417 examples [00:18, 941.71 examples/s]17512 examples [00:18, 898.41 examples/s]17603 examples [00:18, 899.81 examples/s]17694 examples [00:18, 901.54 examples/s]17787 examples [00:18, 909.68 examples/s]17880 examples [00:18, 915.03 examples/s]17979 examples [00:18, 935.22 examples/s]18080 examples [00:19, 953.56 examples/s]18176 examples [00:19, 951.22 examples/s]18277 examples [00:19, 967.10 examples/s]18379 examples [00:19, 981.34 examples/s]18478 examples [00:19, 953.24 examples/s]18574 examples [00:19, 932.16 examples/s]18668 examples [00:19, 932.37 examples/s]18762 examples [00:19, 933.44 examples/s]18856 examples [00:19, 930.52 examples/s]18950 examples [00:20, 918.07 examples/s]19050 examples [00:20, 937.75 examples/s]19148 examples [00:20, 948.41 examples/s]19244 examples [00:20, 949.80 examples/s]19340 examples [00:20, 931.18 examples/s]19437 examples [00:20, 939.72 examples/s]19536 examples [00:20, 953.28 examples/s]19637 examples [00:20, 969.30 examples/s]19737 examples [00:20, 976.66 examples/s]19838 examples [00:20, 985.89 examples/s]19939 examples [00:21, 990.55 examples/s]20039 examples [00:21, 934.95 examples/s]20134 examples [00:21, 922.42 examples/s]20231 examples [00:21, 933.63 examples/s]20328 examples [00:21, 943.97 examples/s]20429 examples [00:21, 960.44 examples/s]20529 examples [00:21, 970.90 examples/s]20631 examples [00:21, 984.02 examples/s]20732 examples [00:21, 989.84 examples/s]20832 examples [00:21, 957.87 examples/s]20929 examples [00:22, 951.19 examples/s]21025 examples [00:22, 945.90 examples/s]21120 examples [00:22, 941.76 examples/s]21215 examples [00:22, 941.65 examples/s]21310 examples [00:22, 937.90 examples/s]21410 examples [00:22, 953.81 examples/s]21506 examples [00:22, 953.82 examples/s]21605 examples [00:22, 964.00 examples/s]21702 examples [00:22, 965.14 examples/s]21799 examples [00:23, 931.86 examples/s]21893 examples [00:23, 931.61 examples/s]21988 examples [00:23, 934.37 examples/s]22082 examples [00:23, 930.08 examples/s]22181 examples [00:23, 945.76 examples/s]22276 examples [00:23, 921.17 examples/s]22369 examples [00:23, 901.96 examples/s]22465 examples [00:23, 917.25 examples/s]22562 examples [00:23, 931.37 examples/s]22658 examples [00:23, 939.52 examples/s]22753 examples [00:24, 927.19 examples/s]22846 examples [00:24, 927.41 examples/s]22939 examples [00:24, 927.24 examples/s]23039 examples [00:24, 945.00 examples/s]23140 examples [00:24, 962.53 examples/s]23237 examples [00:24, 934.96 examples/s]23335 examples [00:24, 946.75 examples/s]23434 examples [00:24, 957.67 examples/s]23530 examples [00:24, 955.27 examples/s]23630 examples [00:24, 965.89 examples/s]23727 examples [00:25, 948.74 examples/s]23823 examples [00:25, 942.82 examples/s]23918 examples [00:25, 944.38 examples/s]24014 examples [00:25, 947.09 examples/s]24114 examples [00:25, 960.24 examples/s]24212 examples [00:25, 963.64 examples/s]24309 examples [00:25, 949.83 examples/s]24405 examples [00:25, 918.38 examples/s]24498 examples [00:25, 895.32 examples/s]24590 examples [00:25, 900.33 examples/s]24681 examples [00:26, 876.60 examples/s]24774 examples [00:26, 891.48 examples/s]24873 examples [00:26, 918.74 examples/s]24973 examples [00:26, 941.19 examples/s]25071 examples [00:26, 952.19 examples/s]25168 examples [00:26, 957.20 examples/s]25264 examples [00:26, 948.00 examples/s]25359 examples [00:26, 916.71 examples/s]25452 examples [00:26, 914.04 examples/s]25544 examples [00:27, 906.86 examples/s]25635 examples [00:27, 891.48 examples/s]25725 examples [00:27, 885.25 examples/s]25823 examples [00:27, 909.06 examples/s]25918 examples [00:27, 919.14 examples/s]26014 examples [00:27, 929.19 examples/s]26109 examples [00:27, 932.62 examples/s]26210 examples [00:27, 952.76 examples/s]26310 examples [00:27, 966.02 examples/s]26410 examples [00:27, 975.70 examples/s]26511 examples [00:28, 983.65 examples/s]26610 examples [00:28, 964.02 examples/s]26709 examples [00:28, 969.68 examples/s]26807 examples [00:28, 969.67 examples/s]26905 examples [00:28, 968.30 examples/s]27004 examples [00:28, 972.36 examples/s]27102 examples [00:28, 933.95 examples/s]27196 examples [00:28, 899.45 examples/s]27287 examples [00:28, 896.09 examples/s]27385 examples [00:28, 917.63 examples/s]27479 examples [00:29, 923.83 examples/s]27572 examples [00:29, 854.91 examples/s]27659 examples [00:29, 848.48 examples/s]27747 examples [00:29, 855.57 examples/s]27840 examples [00:29, 874.29 examples/s]27928 examples [00:29, 858.06 examples/s]28019 examples [00:29, 870.31 examples/s]28108 examples [00:29, 875.26 examples/s]28197 examples [00:29, 878.47 examples/s]28293 examples [00:30, 900.70 examples/s]28384 examples [00:30, 900.17 examples/s]28475 examples [00:30, 896.75 examples/s]28566 examples [00:30, 897.22 examples/s]28658 examples [00:30, 903.78 examples/s]28749 examples [00:30, 885.93 examples/s]28848 examples [00:30, 913.66 examples/s]28948 examples [00:30, 935.95 examples/s]29045 examples [00:30, 945.08 examples/s]29145 examples [00:30, 958.50 examples/s]29243 examples [00:31, 964.42 examples/s]29340 examples [00:31, 958.79 examples/s]29437 examples [00:31, 947.04 examples/s]29532 examples [00:31, 925.57 examples/s]29625 examples [00:31, 915.29 examples/s]29717 examples [00:31, 916.22 examples/s]29810 examples [00:31, 919.27 examples/s]29904 examples [00:31, 922.19 examples/s]29997 examples [00:31, 909.37 examples/s]30089 examples [00:31, 887.53 examples/s]30185 examples [00:32, 906.50 examples/s]30283 examples [00:32, 924.44 examples/s]30379 examples [00:32, 934.14 examples/s]30474 examples [00:32, 936.50 examples/s]30571 examples [00:32, 946.13 examples/s]30669 examples [00:32, 954.30 examples/s]30769 examples [00:32, 965.52 examples/s]30866 examples [00:32, 953.84 examples/s]30962 examples [00:32, 948.82 examples/s]31062 examples [00:32, 961.93 examples/s]31159 examples [00:33, 964.34 examples/s]31256 examples [00:33, 900.44 examples/s]31347 examples [00:33, 894.74 examples/s]31438 examples [00:33, 884.47 examples/s]31527 examples [00:33, 848.34 examples/s]31623 examples [00:33, 875.34 examples/s]31722 examples [00:33, 905.30 examples/s]31818 examples [00:33, 919.01 examples/s]31912 examples [00:33, 923.05 examples/s]32010 examples [00:34, 939.36 examples/s]32105 examples [00:34, 927.54 examples/s]32203 examples [00:34, 940.11 examples/s]32303 examples [00:34, 955.80 examples/s]32403 examples [00:34, 967.44 examples/s]32500 examples [00:34, 967.47 examples/s]32597 examples [00:34, 934.69 examples/s]32691 examples [00:34, 909.24 examples/s]32783 examples [00:34, 905.06 examples/s]32874 examples [00:34, 898.63 examples/s]32965 examples [00:35, 891.04 examples/s]33055 examples [00:35, 873.54 examples/s]33153 examples [00:35, 900.93 examples/s]33254 examples [00:35, 929.85 examples/s]33354 examples [00:35, 949.49 examples/s]33453 examples [00:35, 961.28 examples/s]33550 examples [00:35, 961.70 examples/s]33647 examples [00:35, 941.48 examples/s]33746 examples [00:35, 954.29 examples/s]33846 examples [00:36, 966.03 examples/s]33945 examples [00:36, 972.00 examples/s]34043 examples [00:36, 938.64 examples/s]34138 examples [00:36, 934.03 examples/s]34232 examples [00:36, 925.51 examples/s]34332 examples [00:36, 942.91 examples/s]34430 examples [00:36, 951.24 examples/s]34526 examples [00:36, 929.35 examples/s]34627 examples [00:36, 951.57 examples/s]34728 examples [00:36, 967.22 examples/s]34828 examples [00:37, 974.65 examples/s]34926 examples [00:37, 976.09 examples/s]35024 examples [00:37, 949.79 examples/s]35120 examples [00:37, 941.73 examples/s]35215 examples [00:37, 941.36 examples/s]35310 examples [00:37, 932.40 examples/s]35404 examples [00:37, 920.60 examples/s]35497 examples [00:37, 915.52 examples/s]35596 examples [00:37, 935.95 examples/s]35693 examples [00:37, 942.62 examples/s]35793 examples [00:38, 957.54 examples/s]35891 examples [00:38, 961.41 examples/s]35988 examples [00:38, 951.05 examples/s]36084 examples [00:38, 952.06 examples/s]36185 examples [00:38, 968.54 examples/s]36286 examples [00:38, 979.24 examples/s]36387 examples [00:38, 988.10 examples/s]36486 examples [00:38, 984.88 examples/s]36585 examples [00:38, 937.90 examples/s]36680 examples [00:38, 933.62 examples/s]36777 examples [00:39, 942.05 examples/s]36872 examples [00:39, 922.05 examples/s]36971 examples [00:39, 939.29 examples/s]37072 examples [00:39, 957.17 examples/s]37173 examples [00:39, 970.96 examples/s]37271 examples [00:39, 971.22 examples/s]37370 examples [00:39, 971.94 examples/s]37469 examples [00:39, 974.65 examples/s]37567 examples [00:39, 969.11 examples/s]37666 examples [00:40, 975.27 examples/s]37764 examples [00:40, 958.64 examples/s]37860 examples [00:40, 914.25 examples/s]37953 examples [00:40, 917.35 examples/s]38052 examples [00:40, 936.69 examples/s]38150 examples [00:40, 946.75 examples/s]38246 examples [00:40, 948.07 examples/s]38342 examples [00:40, 949.74 examples/s]38438 examples [00:40, 947.60 examples/s]38533 examples [00:40, 930.28 examples/s]38627 examples [00:41, 920.45 examples/s]38720 examples [00:41, 915.51 examples/s]38813 examples [00:41, 917.47 examples/s]38909 examples [00:41, 928.27 examples/s]39006 examples [00:41, 938.44 examples/s]39105 examples [00:41, 952.65 examples/s]39205 examples [00:41, 966.28 examples/s]39304 examples [00:41, 970.54 examples/s]39402 examples [00:41, 950.34 examples/s]39498 examples [00:41, 930.42 examples/s]39592 examples [00:42, 922.47 examples/s]39685 examples [00:42, 914.42 examples/s]39777 examples [00:42, 879.36 examples/s]39867 examples [00:42, 885.19 examples/s]39964 examples [00:42, 906.81 examples/s]40056 examples [00:42, 877.89 examples/s]40156 examples [00:42, 909.20 examples/s]40256 examples [00:42, 932.79 examples/s]40351 examples [00:42, 936.64 examples/s]40452 examples [00:43, 955.21 examples/s]40548 examples [00:43, 934.34 examples/s]40645 examples [00:43, 941.91 examples/s]40743 examples [00:43, 951.83 examples/s]40841 examples [00:43, 959.39 examples/s]40941 examples [00:43, 971.10 examples/s]41039 examples [00:43, 973.59 examples/s]41139 examples [00:43, 979.67 examples/s]41238 examples [00:43, 982.68 examples/s]41337 examples [00:43, 979.98 examples/s]41436 examples [00:44, 979.73 examples/s]41535 examples [00:44, 965.76 examples/s]41635 examples [00:44, 973.53 examples/s]41733 examples [00:44, 974.93 examples/s]41831 examples [00:44, 951.91 examples/s]41927 examples [00:44, 933.17 examples/s]42021 examples [00:44, 905.63 examples/s]42121 examples [00:44, 930.01 examples/s]42215 examples [00:44, 928.18 examples/s]42309 examples [00:44, 905.88 examples/s]42401 examples [00:45, 908.28 examples/s]42501 examples [00:45, 932.59 examples/s]42600 examples [00:45, 948.36 examples/s]42696 examples [00:45, 949.15 examples/s]42793 examples [00:45, 952.82 examples/s]42894 examples [00:45, 966.96 examples/s]42995 examples [00:45, 977.30 examples/s]43095 examples [00:45, 983.98 examples/s]43194 examples [00:45, 979.04 examples/s]43292 examples [00:45, 979.23 examples/s]43390 examples [00:46, 975.59 examples/s]43488 examples [00:46, 965.67 examples/s]43585 examples [00:46, 957.68 examples/s]43685 examples [00:46, 968.20 examples/s]43787 examples [00:46, 981.33 examples/s]43888 examples [00:46, 989.71 examples/s]43988 examples [00:46, 980.68 examples/s]44087 examples [00:46, 951.00 examples/s]44184 examples [00:46, 955.83 examples/s]44285 examples [00:47, 969.06 examples/s]44385 examples [00:47, 975.69 examples/s]44483 examples [00:47, 970.95 examples/s]44582 examples [00:47, 974.02 examples/s]44680 examples [00:47, 957.85 examples/s]44776 examples [00:47, 946.05 examples/s]44874 examples [00:47, 955.85 examples/s]44970 examples [00:47, 955.89 examples/s]45068 examples [00:47, 962.48 examples/s]45165 examples [00:47, 947.96 examples/s]45260 examples [00:48, 915.37 examples/s]45360 examples [00:48, 937.29 examples/s]45459 examples [00:48, 951.15 examples/s]45559 examples [00:48, 963.06 examples/s]45659 examples [00:48, 971.23 examples/s]45759 examples [00:48, 978.76 examples/s]45860 examples [00:48, 986.46 examples/s]45959 examples [00:48, 985.88 examples/s]46058 examples [00:48, 969.96 examples/s]46156 examples [00:48, 934.20 examples/s]46250 examples [00:49, 924.53 examples/s]46345 examples [00:49, 930.78 examples/s]46441 examples [00:49, 937.77 examples/s]46535 examples [00:49, 936.27 examples/s]46629 examples [00:49, 887.16 examples/s]46720 examples [00:49, 893.83 examples/s]46815 examples [00:49, 908.00 examples/s]46910 examples [00:49, 919.32 examples/s]47011 examples [00:49, 942.46 examples/s]47106 examples [00:49, 930.69 examples/s]47200 examples [00:50, 909.03 examples/s]47292 examples [00:50, 911.77 examples/s]47391 examples [00:50, 932.02 examples/s]47485 examples [00:50, 925.56 examples/s]47578 examples [00:50, 922.49 examples/s]47671 examples [00:50, 920.12 examples/s]47768 examples [00:50, 934.11 examples/s]47863 examples [00:50, 936.93 examples/s]47961 examples [00:50, 947.77 examples/s]48056 examples [00:51, 910.21 examples/s]48148 examples [00:51, 899.25 examples/s]48244 examples [00:51, 906.16 examples/s]48335 examples [00:51, 869.20 examples/s]48428 examples [00:51, 885.20 examples/s]48520 examples [00:51, 894.45 examples/s]48620 examples [00:51, 923.01 examples/s]48721 examples [00:51, 946.12 examples/s]48822 examples [00:51, 962.08 examples/s]48921 examples [00:51, 968.02 examples/s]49020 examples [00:52, 972.76 examples/s]49118 examples [00:52, 972.26 examples/s]49217 examples [00:52, 976.24 examples/s]49315 examples [00:52, 956.21 examples/s]49411 examples [00:52, 939.49 examples/s]49506 examples [00:52, 941.95 examples/s]49604 examples [00:52, 949.78 examples/s]49705 examples [00:52, 966.61 examples/s]49805 examples [00:52, 975.60 examples/s]49903 examples [00:52, 951.14 examples/s]50000 examples [00:53, 956.15 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6849/50000 [00:00<00:00, 68489.41 examples/s] 37%|███▋      | 18492/50000 [00:00<00:00, 78141.69 examples/s] 59%|█████▊    | 29370/50000 [00:00<00:00, 85352.19 examples/s] 82%|████████▏ | 41013/50000 [00:00<00:00, 92780.04 examples/s]                                                               0 examples [00:00, ? examples/s]78 examples [00:00, 776.15 examples/s]169 examples [00:00, 810.59 examples/s]255 examples [00:00, 822.96 examples/s]354 examples [00:00, 865.78 examples/s]450 examples [00:00, 891.03 examples/s]536 examples [00:00, 881.00 examples/s]631 examples [00:00, 898.48 examples/s]729 examples [00:00, 919.17 examples/s]823 examples [00:00, 924.66 examples/s]913 examples [00:01, 912.63 examples/s]1003 examples [00:01, 905.31 examples/s]1102 examples [00:01, 927.69 examples/s]1203 examples [00:01, 948.48 examples/s]1301 examples [00:01, 957.58 examples/s]1397 examples [00:01, 957.31 examples/s]1493 examples [00:01, 952.45 examples/s]1593 examples [00:01, 963.87 examples/s]1692 examples [00:01, 970.78 examples/s]1790 examples [00:01, 967.94 examples/s]1887 examples [00:02, 966.00 examples/s]1984 examples [00:02, 960.09 examples/s]2081 examples [00:02, 950.32 examples/s]2177 examples [00:02, 948.32 examples/s]2274 examples [00:02, 952.32 examples/s]2372 examples [00:02, 957.68 examples/s]2470 examples [00:02, 963.35 examples/s]2568 examples [00:02, 967.72 examples/s]2665 examples [00:02, 965.31 examples/s]2762 examples [00:02, 965.48 examples/s]2860 examples [00:03, 968.76 examples/s]2957 examples [00:03, 928.88 examples/s]3056 examples [00:03, 946.01 examples/s]3155 examples [00:03, 958.29 examples/s]3253 examples [00:03, 962.16 examples/s]3350 examples [00:03, 956.47 examples/s]3446 examples [00:03, 946.87 examples/s]3542 examples [00:03, 948.71 examples/s]3637 examples [00:03, 940.96 examples/s]3732 examples [00:03, 939.72 examples/s]3829 examples [00:04, 947.07 examples/s]3925 examples [00:04, 949.08 examples/s]4020 examples [00:04, 949.00 examples/s]4117 examples [00:04, 952.54 examples/s]4216 examples [00:04, 962.70 examples/s]4313 examples [00:04, 937.98 examples/s]4414 examples [00:04, 955.96 examples/s]4510 examples [00:04, 956.68 examples/s]4608 examples [00:04, 962.17 examples/s]4707 examples [00:04, 970.09 examples/s]4810 examples [00:05, 984.45 examples/s]4911 examples [00:05, 990.49 examples/s]5014 examples [00:05, 1000.14 examples/s]5116 examples [00:05, 1003.33 examples/s]5217 examples [00:05, 988.60 examples/s] 5316 examples [00:05, 987.43 examples/s]5419 examples [00:05, 998.53 examples/s]5519 examples [00:05, 946.87 examples/s]5615 examples [00:05, 933.63 examples/s]5713 examples [00:06, 946.24 examples/s]5812 examples [00:06, 956.02 examples/s]5908 examples [00:06, 946.45 examples/s]6010 examples [00:06, 966.96 examples/s]6110 examples [00:06, 975.82 examples/s]6210 examples [00:06, 982.05 examples/s]6309 examples [00:06, 949.33 examples/s]6407 examples [00:06, 957.57 examples/s]6504 examples [00:06, 951.33 examples/s]6600 examples [00:06, 945.99 examples/s]6695 examples [00:07, 907.42 examples/s]6787 examples [00:07, 906.83 examples/s]6880 examples [00:07, 912.91 examples/s]6972 examples [00:07, 911.20 examples/s]7064 examples [00:07, 912.06 examples/s]7158 examples [00:07, 912.61 examples/s]7256 examples [00:07, 931.31 examples/s]7355 examples [00:07, 945.60 examples/s]7456 examples [00:07, 961.85 examples/s]7555 examples [00:07, 967.48 examples/s]7656 examples [00:08, 977.92 examples/s]7754 examples [00:08, 977.88 examples/s]7852 examples [00:08, 975.02 examples/s]7950 examples [00:08, 975.71 examples/s]8049 examples [00:08, 976.76 examples/s]8147 examples [00:08, 973.75 examples/s]8247 examples [00:08, 978.70 examples/s]8345 examples [00:08, 976.54 examples/s]8443 examples [00:08, 955.74 examples/s]8539 examples [00:08, 940.31 examples/s]8634 examples [00:09, 934.21 examples/s]8728 examples [00:09, 920.96 examples/s]8821 examples [00:09, 906.65 examples/s]8912 examples [00:09, 906.24 examples/s]9009 examples [00:09, 924.18 examples/s]9102 examples [00:09, 923.67 examples/s]9195 examples [00:09, 923.76 examples/s]9295 examples [00:09, 945.33 examples/s]9390 examples [00:09, 930.85 examples/s]9484 examples [00:10, 919.53 examples/s]9578 examples [00:10, 924.63 examples/s]9671 examples [00:10, 880.08 examples/s]9772 examples [00:10, 913.45 examples/s]9873 examples [00:10, 939.66 examples/s]9972 examples [00:10, 953.66 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteU5678M/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteU5678M/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7b824a9488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7b824a9488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7b824a9488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7b696565f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7b1c6f3400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7b824a9488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7b824a9488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7b824a9488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7b7f980320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7b7f980128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f7afa2b0ea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f7afa2b0ea0> 

  function with postional parmater data_info <function split_train_valid at 0x7f7afa2b0ea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f7afa2af048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f7afa2af048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f7afa2af048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=70413dd89536b97a8061deffe7ebe09d211f847346452d0496dd19066c54434e
  Stored in directory: /tmp/pip-ephem-wheel-cache-dmza42iq/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<17:57:10, 13.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:48:00, 18.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:00:54, 26.6kB/s]  .vector_cache/glove.6B.zip:   0%|          | 737k/862M [00:01<6:19:41, 37.8kB/s].vector_cache/glove.6B.zip:   0%|          | 2.69M/862M [00:01<4:25:28, 54.0kB/s].vector_cache/glove.6B.zip:   1%|          | 6.90M/862M [00:01<3:05:01, 77.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.4M/862M [00:01<2:08:56, 110kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 13.8M/862M [00:01<1:30:10, 157kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<1:02:52, 224kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:01<43:56, 319kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:01<30:40, 454kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.5M/862M [00:01<21:29, 645kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:01<15:02, 916kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:02<10:37, 1.29MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.4M/862M [00:02<07:30, 1.82MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.0M/862M [00:02<05:24, 2.52MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.6M/862M [00:02<03:51, 3.51MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.9M/862M [00:03<04:03, 3.33MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<03:02, 4.41MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<06:27, 2.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:05<08:13, 1.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<06:31, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:05<05:03, 2.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<06:01, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:07<05:58, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:07<04:33, 2.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:07<03:21, 3.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<21:22, 621kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<18:34, 715kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:09<13:53, 955kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<09:52, 1.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<12:01, 1.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:11<10:22, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:11<08:08, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.5M/862M [00:11<06:01, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:11<04:23, 2.99MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<32:06, 409kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:13<24:18, 540kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:13<17:25, 753kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:13<12:21, 1.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<18:58, 689kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:15<15:00, 870kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<10:55, 1.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<10:08, 1.28MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<08:45, 1.48MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:17<06:33, 1.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:19<07:06, 1.82MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:19<07:03, 1.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:19<05:51, 2.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:19<04:29, 2.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:19<03:23, 3.80MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:21<08:25, 1.53MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:21<08:09, 1.58MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:21<06:15, 2.05MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:21<04:30, 2.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:23<44:48, 286kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:23<33:06, 386kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:23<23:30, 543kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:23<16:38, 766kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:25<17:56, 709kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:25<16:04, 792kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:25<12:02, 1.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:25<10:30, 1.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<07:33, 1.68MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<05:46, 2.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<11:21, 1.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<12:26, 1.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<09:41, 1.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<07:05, 1.78MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<05:29, 2.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<04:16, 2.94MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<10:51, 1.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<19:53, 633kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<16:24, 767kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<12:13, 1.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<08:59, 1.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<06:42, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:02, 2.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<19:22, 646kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<15:08, 827kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<11:40, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<08:38, 1.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:36, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:10, 2.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:10, 2.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<03:36, 3.46MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<32:49, 379kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<27:02, 460kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<19:56, 624kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<14:18, 867kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<10:17, 1.20MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<07:38, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:14:16, 167kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<55:33, 223kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<39:54, 310kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<28:23, 435kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<20:10, 612kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<14:28, 851kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<17:19, 710kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<20:45, 593kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<16:24, 750kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<11:59, 1.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<08:41, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<06:23, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<21:41, 565kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<20:14, 605kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<15:30, 789kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<11:19, 1.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<08:23, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<06:17, 1.94MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:39<04:55, 2.48MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<2:20:39, 86.5kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<1:43:24, 118kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<1:13:33, 165kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<51:47, 235kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<36:50, 330kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<26:12, 463kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:41<18:46, 645kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<25:25, 476kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<22:51, 529kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<17:01, 710kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<12:19, 980kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<09:05, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<07:13, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<05:24, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<13:27, 894kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<16:03, 749kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<12:48, 939kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<09:31, 1.26MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<07:15, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:35, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<04:27, 2.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<16:54, 708kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<17:44, 675kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<13:52, 862kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<10:16, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<07:42, 1.55MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:51, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<04:41, 2.54MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<10:56, 1.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<13:26, 885kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<10:48, 1.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<08:05, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:09, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<04:48, 2.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<03:47, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<54:38, 216kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<43:34, 271kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<31:48, 372kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<22:45, 519kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<16:21, 721kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<12:00, 982kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<08:49, 1.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<14:49, 793kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<15:39, 751kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<12:06, 971kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<08:59, 1.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<06:49, 1.72MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:17, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<04:09, 2.81MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<03:24, 3.43MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<28:19, 413kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<25:06, 466kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<20:29, 570kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<15:09, 771kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<11:22, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:25, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:23, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:03, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:09, 2.79MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<13:12, 880kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<12:49, 906kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<09:52, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<07:33, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:46, 2.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<04:56, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<03:56, 2.93MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<03:36, 3.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<10:17, 1.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<10:34, 1.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<08:15, 1.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<06:22, 1.81MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:02, 2.29MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<03:58, 2.90MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<03:50, 2.99MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:45, 3.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:26, 3.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<3:39:03, 52.4kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<2:36:00, 73.6kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<1:50:48, 104kB/s] .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<1:18:09, 147kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<55:16, 207kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<39:18, 291kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<28:01, 408kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<20:10, 566kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<14:38, 780kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<28:02, 407kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<25:22, 450kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<19:36, 582kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<14:25, 791kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<10:48, 1.06MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<08:03, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:16, 1.81MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:57, 2.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<04:01, 2.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<12:08, 934kB/s] .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<11:28, 989kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<08:51, 1.28MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:49, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<05:17, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:16, 2.64MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<03:30, 3.22MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<03:01, 3.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<09:50, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<13:18, 848kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<11:25, 986kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<09:35, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<07:46, 1.45MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:33, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:33, 2.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<04:38, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<03:52, 2.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<03:20, 3.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<02:52, 3.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<03:18, 3.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<14:02, 799kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<11:53, 942kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<09:04, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<07:04, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:41, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:38, 2.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<03:57, 2.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:23, 3.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:05, 3.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<11:48, 943kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<11:40, 954kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<09:08, 1.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:57, 1.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:35, 1.99MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:28, 2.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<03:48, 2.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<03:12, 3.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<02:54, 3.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<16:48, 658kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<14:20, 772kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<11:46, 940kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<08:52, 1.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<06:39, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<05:28, 2.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:23, 2.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<03:39, 3.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:07, 3.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<20:59, 524kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<17:42, 621kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<13:09, 836kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<09:46, 1.12MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<07:26, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:49, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:40, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:50, 2.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<19:23, 564kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<16:26, 665kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<12:17, 889kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<09:08, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<06:55, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:21, 2.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:17, 2.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<31:53, 341kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<28:13, 385kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<21:08, 514kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<15:24, 704kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<11:19, 957kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<08:23, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:27, 1.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:03, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<13:15, 814kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<12:47, 843kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<11:19, 953kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<10:52, 992kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<09:34, 1.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<07:48, 1.38MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:48, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<06:06, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<06:01, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:28, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:07, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:37, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:46, 2.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:39, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:24, 2.44MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:57, 2.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:55, 2.73MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<09:14, 1.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:09, 1.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<06:18, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<04:48, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:06, 2.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<03:23, 3.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<02:53, 3.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<02:37, 4.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<34:59, 305kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<29:41, 359kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<22:10, 480kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<16:04, 662kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<11:46, 903kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<08:43, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<06:36, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<12:48, 827kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<11:42, 904kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<08:54, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<06:43, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:07, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:12, 2.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<03:23, 3.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:02, 3.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<51:57, 203kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<41:30, 253kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<30:19, 347kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<21:44, 483kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<15:39, 670kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<11:23, 919kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:26, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<11:45, 889kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<13:20, 783kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<10:32, 991kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:50, 1.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:58, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:32, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:37, 2.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<14:09, 733kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<14:31, 715kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<11:19, 917kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<08:22, 1.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:15, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:44, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:44, 2.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<22:03, 467kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<19:41, 524kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<14:49, 695kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<10:46, 955kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<08:19, 1.23MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:07, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:48, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<20:02, 511kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<19:34, 523kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<15:00, 682kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<10:59, 930kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<08:07, 1.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:09, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:57, 2.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<17:56, 567kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<17:29, 582kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<13:26, 757kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<09:51, 1.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<07:18, 1.39MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:31, 1.83MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:12, 2.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<2:31:23, 66.8kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<1:50:27, 91.5kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<1:18:35, 129kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<55:32, 182kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<39:12, 257kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<27:52, 361kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<19:50, 507kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<14:18, 702kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<26:07, 384kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<22:21, 449kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<16:39, 602kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<12:03, 830kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<08:48, 1.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:31, 1.53MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<11:48, 844kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<11:39, 856kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<10:26, 954kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:57, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:59, 1.66MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:32, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<03:32, 2.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<02:50, 3.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<19:27, 509kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<17:23, 569kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<13:08, 753kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<09:36, 1.03MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<07:03, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:17, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<11:32, 852kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<11:45, 836kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<09:02, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:45, 1.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:04, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:53, 2.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<13:07, 744kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<12:53, 757kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<09:58, 979kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<07:18, 1.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:29, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<04:10, 2.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<03:17, 2.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<21:18, 455kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<18:51, 514kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<14:08, 685kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<10:16, 941kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:28, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<05:37, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<12:49, 751kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<12:57, 742kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<11:13, 857kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<09:08, 1.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<07:04, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:20, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:06, 2.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:12, 2.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<02:37, 3.64MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<24:19, 393kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<20:53, 457kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<15:32, 614kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<11:14, 848kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<08:10, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<06:03, 1.57MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<04:46, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<26:08, 363kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<23:24, 405kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<17:42, 535kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<12:50, 737kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<09:25, 1.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<07:00, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:18, 1.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<14:09, 665kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<14:21, 656kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<13:16, 709kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<11:22, 828kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<09:52, 953kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<09:39, 975kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<09:19, 1.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<09:06, 1.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<08:21, 1.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<07:48, 1.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<07:37, 1.23MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<07:21, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<06:55, 1.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<06:35, 1.42MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<05:28, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:21, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<03:24, 2.74MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<02:46, 3.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<14:09, 661kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<10:14, 911kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<07:30, 1.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:35, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<08:33, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<13:45, 675kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<11:39, 795kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<08:41, 1.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<06:23, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:44, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:46, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<09:22, 983kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<10:20, 890kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<07:57, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<06:14, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:46, 1.92MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:42, 2.47MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:58, 3.09MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:31, 3.62MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<09:07, 1.00MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<09:50, 929kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<07:47, 1.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:46, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:17, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:28, 2.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<06:16, 1.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<07:48, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<06:20, 1.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:46, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:43, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:52, 3.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<08:40, 1.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<09:10, 982kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<08:58, 1.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<07:40, 1.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<06:32, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:49, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:17, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:57, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<04:29, 2.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:05, 2.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<03:40, 2.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:12, 2.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<02:48, 3.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<02:28, 3.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<02:20, 3.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<19:28, 459kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<14:10, 629kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<10:15, 868kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:29, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<05:31, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<20:03, 442kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<21:11, 418kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<16:33, 535kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<12:03, 734kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<08:45, 1.01MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<06:24, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<04:49, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<12:23, 710kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<12:14, 718kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<09:20, 942kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<07:02, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:17, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:07, 2.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:13, 2.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:34, 3.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<15:51, 550kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<14:23, 607kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<10:53, 801kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<07:56, 1.10MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:46, 1.50MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:27, 1.95MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<07:27, 1.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<08:15, 1.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<06:33, 1.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:54, 1.76MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:39, 2.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:57, 2.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:54, 1.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<07:50, 1.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<06:15, 1.37MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:38, 1.84MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:34, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:44, 3.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:31, 1.31MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:32, 1.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<06:01, 1.41MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:28, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:23, 2.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:37, 3.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<08:17, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<08:32, 989kB/s] .vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:28<06:36, 1.28MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<05:00, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:46, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:52, 2.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:56, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:43, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<05:14, 1.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:56, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:01, 2.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:39, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:57, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<06:47, 1.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<05:09, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:06, 2.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<03:01, 2.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:26, 3.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<19:04, 432kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<16:28, 500kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<12:20, 667kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<08:55, 921kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:28, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:47, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<10:13, 799kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<10:00, 816kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:38, 1.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:35, 1.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:16, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:14, 2.50MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:57, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<11:36, 699kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<09:47, 828kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<07:34, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<05:43, 1.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:19, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:16, 2.46MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<02:32, 3.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<10:19, 779kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<10:54, 737kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<08:33, 938kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<06:23, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:51, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:41, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:58, 2.69MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:22, 3.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<16:42, 477kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<15:14, 523kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<11:34, 688kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<08:26, 943kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:13, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:40, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:34, 2.22MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<06:54, 1.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<08:02, 982kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:25, 1.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:48, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:39, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<02:50, 2.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<08:03, 971kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<08:54, 878kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<08:45, 894kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:22, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:08, 1.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<05:19, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<04:55, 1.59MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:32, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:06, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:34, 2.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<02:57, 2.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<02:24, 3.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:28, 3.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:02, 1.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:54, 1.98MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:04, 2.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:39, 2.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<02:11, 3.52MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<01:59, 3.88MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<01:43, 4.45MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<01:39, 4.64MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<11:02, 696kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<11:56, 644kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<09:20, 823kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:52, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:14, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:01, 1.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:10, 2.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:33, 2.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<06:42, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<08:35, 888kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<07:26, 1.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:54, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:33, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:32, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<02:55, 2.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<02:39, 2.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:28, 3.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:29, 3.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:36, 2.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:43, 2.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:55, 2.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<41:23, 183kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<30:14, 250kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<22:05, 342kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<15:47, 477kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<11:23, 661kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<08:16, 907kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<06:07, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:35, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<12:47, 586kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<12:36, 593kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<09:42, 771kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<07:10, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:20, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:01, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:11, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:34, 2.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<08:41, 854kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<09:42, 764kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<07:55, 936kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<05:56, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:31, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:28, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<02:44, 2.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<02:13, 3.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<07:01, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<08:15, 891kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<06:38, 1.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:58, 1.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:46, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:59, 2.44MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:24, 3.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:01, 3.60MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<27:12, 268kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<22:35, 322kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<16:36, 438kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<11:57, 607kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<08:41, 835kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:25, 1.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:46, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:42, 1.95MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<08:36, 838kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<09:32, 756kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<07:25, 971kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<05:33, 1.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:09, 1.73MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:12, 2.23MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:34, 2.77MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<1:44:31, 68.3kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<1:16:57, 92.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<56:20, 127kB/s]   .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<41:03, 174kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<30:57, 230kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<23:31, 303kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<18:21, 389kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:06<14:44, 484kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:06<12:05, 589kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<10:23, 686kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<08:00, 890kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<05:54, 1.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<04:29, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:36, 1.97MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:54, 2.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:59, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:13, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:30, 2.81MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:05, 3.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<01:45, 4.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<01:30, 4.64MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<06:52, 1.02MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<11:36, 603kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<09:38, 726kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<07:07, 982kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<05:15, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:57, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:00, 2.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:22, 2.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<1:45:30, 65.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<1:16:28, 90.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<54:09, 128kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<38:03, 182kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<26:57, 256kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<19:00, 362kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<13:36, 505kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<15:56, 431kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<13:46, 498kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<10:14, 669kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<07:25, 922kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:24, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<03:57, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<19:21, 351kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<15:44, 432kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<11:34, 587kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<08:18, 815kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<06:01, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<04:24, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<06:58, 964kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<06:56, 969kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<05:19, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:59, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:57, 2.27MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:17, 2.91MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<05:53, 1.13MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<05:56, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<04:37, 1.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:27, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:35, 2.55MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<05:21, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<07:18, 902kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<05:54, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:24, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:17, 1.99MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:25, 2.70MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<09:07, 714kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<09:37, 678kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<07:31, 865kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<05:29, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:56, 1.64MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<05:47, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<06:53, 935kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<05:27, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:01, 1.60MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:57, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:12, 2.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<31:47, 201kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<24:50, 257kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<17:56, 355kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<12:43, 500kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<09:01, 703kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<08:25, 749kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<08:24, 751kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<06:35, 957kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:54, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:36, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:40, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<05:03, 1.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<05:06, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:56, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:51, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:07, 2.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<12:28, 495kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<10:37, 581kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<07:54, 780kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:38, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<05:33, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<05:34, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<04:15, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:10, 1.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:16, 2.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<24:14, 249kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<18:30, 326kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<13:20, 452kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<09:23, 638kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<08:36, 694kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<07:32, 790kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<05:39, 1.05MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:03, 1.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:20, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<04:30, 1.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:50, 1.53MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:55, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:12, 2.66MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<01:38, 3.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<16:09, 361kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:43<13:12, 441kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<09:41, 601kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<06:50, 846kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<06:38, 868kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<05:55, 972kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<04:27, 1.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:10, 1.80MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:55, 1.16MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<04:55, 1.15MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<03:45, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:49, 2.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:02, 2.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<07:00, 802kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<06:11, 907kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<04:51, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:34, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:33, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<07:21, 755kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<06:25, 864kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<04:48, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:25, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:33, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:08, 1.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:05, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:16, 2.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:11, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:26, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<02:40, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<01:58, 2.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<01:31, 3.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:43, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<06:10, 865kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<05:02, 1.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:42, 1.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:41, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:03, 2.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<10:34, 499kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<08:45, 602kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<06:28, 813kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:38, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<03:19, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<15:07, 344kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<13:13, 394kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<10:12, 510kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<07:31, 690kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<05:25, 955kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:55, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:51, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<13:53, 370kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<10:55, 470kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<07:56, 645kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<05:39, 902kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<04:02, 1.26MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<16:33, 306kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<13:51, 366kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<10:11, 497kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<07:13, 697kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<05:11, 967kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:47, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<06:10, 811kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<06:36, 756kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<05:06, 978kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:51, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:48, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:04, 2.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<04:36, 1.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<05:27, 904kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<04:21, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:21, 1.46MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:24, 2.03MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:51, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:42, 1.03MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:47, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:46, 1.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:00, 2.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<05:48, 826kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<06:00, 797kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:41, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:23, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:27, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<07:49, 603kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<07:24, 638kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<05:39, 834kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<04:03, 1.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:56, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:54, 1.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<04:38, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:43, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:42, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<01:59, 2.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:55, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:37, 991kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:40, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:40, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:57, 2.32MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<04:01, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<04:36, 978kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:40, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:40, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:58, 2.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:07, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:52, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:05, 1.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:16, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:41, 2.59MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:40, 1.63MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:55, 1.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:26, 1.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:45, 1.58MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:09, 2.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:41, 2.58MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:20, 3.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:08, 3.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:09, 3.73MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:36, 936kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:46, 1.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:44, 1.56MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:59, 2.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<03:25, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<04:00, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:09, 1.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:21, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:46, 2.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<03:48, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<06:04, 687kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<05:12, 799kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<03:53, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:32<02:48, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:03, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<10:22, 395kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<07:57, 515kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<05:52, 697kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<04:16, 955kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<03:06, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:19, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:35<03:12, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:35<04:26, 906kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<03:40, 1.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<02:40, 1.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:58, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:27, 2.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:37<13:37, 291kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:37<10:33, 375kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:37<07:34, 521kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<05:25, 726kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:50, 1.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<04:39, 836kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<05:14, 743kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<04:15, 912kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<03:13, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:22, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:56, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:26, 2.67MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<02:56, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<02:57, 1.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:41<02:17, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:41, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<02:12, 1.70MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<03:25, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<02:51, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<02:06, 1.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:31, 2.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<03:04, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<04:04, 904kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<03:26, 1.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<02:37, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:55, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:25, 2.54MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<02:32, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<03:32, 1.02MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<02:53, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<02:07, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:34, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<02:21, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<03:15, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<02:41, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<01:59, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:27, 2.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<03:10, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<03:54, 888kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<03:08, 1.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<02:23, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<01:45, 1.97MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:17, 2.63MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<02:49, 1.20MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<03:41, 923kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<02:59, 1.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<02:10, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:36, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<02:04, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<03:07, 1.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<02:33, 1.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<01:59, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<01:35, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:16, 2.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:02, 3.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<00:50, 3.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<02:21, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<02:35, 1.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<02:02, 1.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:28, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:05, 2.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<1:26:40, 37.0kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<1:02:13, 51.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<43:51, 72.9kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<30:36, 104kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<21:17, 148kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<16:54, 185kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<13:12, 237kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<09:35, 326kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<06:46, 459kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:45, 646kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<05:12, 588kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<05:07, 598kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:03<03:52, 788kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<02:47, 1.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:59, 1.51MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<03:16, 913kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<03:36, 829kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:05<02:52, 1.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<02:05, 1.42MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:30, 1.94MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<03:02, 963kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<03:32, 827kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<02:48, 1.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<02:02, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:28, 1.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<02:53, 987kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<03:17, 870kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<02:37, 1.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<01:54, 1.48MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:23, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:50, 984kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<03:12, 868kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<02:32, 1.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:51, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:20, 2.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<03:17, 825kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<03:29, 777kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<02:44, 987kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:59, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:25, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<03:45, 706kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<03:41, 717kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<02:51, 925kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<02:03, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:28, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<05:27, 473kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<04:51, 531kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<03:39, 704kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<02:36, 981kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:50, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:18<05:09, 487kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:18<04:37, 544kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<03:26, 727kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<02:27, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:45, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<02:31, 970kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<02:30, 971kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<02:18, 1.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:41, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:15, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<00:55, 2.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:22<02:34, 924kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:22<02:42, 876kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<02:04, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<01:32, 1.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:06, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:43, 1.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<02:00, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:35, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<01:18, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<01:02, 2.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<00:51, 2.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<00:43, 3.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<00:38, 3.47MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<00:35, 3.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<00:36, 3.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<02:16, 981kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<02:01, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:37, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:12, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<00:52, 2.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:38, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:49, 1.18MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<01:26, 1.50MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<01:02, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:45, 2.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<03:21, 625kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<02:57, 709kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<02:12, 944kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:34, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<01:42, 1.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<01:59, 1.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<01:51, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<01:31, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<01:09, 1.74MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<00:52, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:40, 2.90MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:31, 3.72MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<02:58, 659kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<02:27, 795kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:34<01:48, 1.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:15, 1.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<02:46, 682kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<02:25, 778kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<01:48, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:16, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<01:29, 1.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<01:27, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<01:06, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:48, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<00:59, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<01:04, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<00:50, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:36, 2.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<01:26, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<01:21, 1.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:00, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:43, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:44<01:17, 1.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:19, 1.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:09, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:53, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:40, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:29, 3.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<01:11, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<01:15, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:58, 1.57MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:41, 2.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<01:01, 1.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:59, 1.50MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:44, 1.95MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:31, 2.68MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<01:06, 1.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<01:01, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<00:46, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:44, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:50, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<00:39, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:28, 2.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:42, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:46, 1.62MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:54<00:36, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:26, 2.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:38, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:38, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:32, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:24, 2.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:17, 3.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<01:19, 856kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<01:09, 978kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<00:51, 1.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:35, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<01:50, 575kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<01:30, 704kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<01:05, 965kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:46, 1.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:45, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:43, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:33, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:23, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<00:42, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<00:40, 1.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:30, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:21, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<00:32, 1.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<00:33, 1.54MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:25, 1.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<00:24, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<00:25, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<00:22, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:16, 2.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:11, 3.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:10<01:18, 552kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:10<01:02, 691kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:47, 893kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:33, 1.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:22, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<09:29, 68.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:12<06:43, 96.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<04:43, 136kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<03:11, 193kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<02:14, 259kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<01:40, 344kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<01:10, 480kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:48, 676kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:32, 945kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<01:01, 497kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:16<00:49, 618kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:16<00:35, 845kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:23, 1.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:32, 809kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:28, 939kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:20, 1.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:22, 1.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:20, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:16, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:11, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:07, 2.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:14, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:16, 1.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:14, 1.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:11, 1.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:07, 2.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:04, 2.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<04:16, 55.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<02:59, 78.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<02:00, 112kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<01:17, 159kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:47, 215kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:34, 289kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:22, 405kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:11, 575kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:11, 516kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:09, 640kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:05, 872kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:03, 1.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.68MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:54, 36.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:34, 51.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:13, 72.9kB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<24:44:59,  4.49it/s]  0%|          | 666/400000 [00:00<17:18:03,  6.41it/s]  0%|          | 1325/400000 [00:00<12:05:44,  9.16it/s]  0%|          | 1969/400000 [00:00<8:27:30, 13.07it/s]   1%|          | 2635/400000 [00:00<5:54:57, 18.66it/s]  1%|          | 3311/400000 [00:00<4:08:20, 26.62it/s]  1%|          | 4066/400000 [00:00<2:53:46, 37.97it/s]  1%|          | 4825/400000 [00:00<2:01:40, 54.13it/s]  1%|▏         | 5526/400000 [00:01<1:25:17, 77.08it/s]  2%|▏         | 6196/400000 [00:01<59:55, 109.54it/s]   2%|▏         | 6949/400000 [00:01<42:07, 155.52it/s]  2%|▏         | 7663/400000 [00:01<29:42, 220.11it/s]  2%|▏         | 8359/400000 [00:01<21:02, 310.12it/s]  2%|▏         | 9049/400000 [00:01<14:59, 434.54it/s]  2%|▏         | 9736/400000 [00:01<10:46, 603.80it/s]  3%|▎         | 10416/400000 [00:01<07:49, 830.06it/s]  3%|▎         | 11163/400000 [00:01<05:43, 1131.88it/s]  3%|▎         | 11923/400000 [00:01<04:15, 1519.94it/s]  3%|▎         | 12688/400000 [00:02<03:13, 2000.76it/s]  3%|▎         | 13443/400000 [00:02<02:30, 2566.45it/s]  4%|▎         | 14181/400000 [00:02<02:01, 3173.65it/s]  4%|▎         | 14917/400000 [00:02<01:40, 3826.38it/s]  4%|▍         | 15648/400000 [00:02<01:26, 4428.59it/s]  4%|▍         | 16406/400000 [00:02<01:15, 5058.79it/s]  4%|▍         | 17139/400000 [00:02<01:08, 5574.30it/s]  4%|▍         | 17905/400000 [00:02<01:02, 6069.81it/s]  5%|▍         | 18648/400000 [00:02<00:59, 6373.47it/s]  5%|▍         | 19384/400000 [00:02<00:57, 6622.57it/s]  5%|▌         | 20144/400000 [00:03<00:55, 6886.93it/s]  5%|▌         | 20888/400000 [00:03<00:53, 7043.87it/s]  5%|▌         | 21631/400000 [00:03<00:53, 7137.53it/s]  6%|▌         | 22401/400000 [00:03<00:51, 7295.71it/s]  6%|▌         | 23151/400000 [00:03<00:51, 7328.09it/s]  6%|▌         | 23898/400000 [00:03<00:51, 7359.74it/s]  6%|▌         | 24648/400000 [00:03<00:50, 7399.60it/s]  6%|▋         | 25395/400000 [00:03<00:50, 7374.86it/s]  7%|▋         | 26151/400000 [00:03<00:50, 7427.25it/s]  7%|▋         | 26899/400000 [00:03<00:50, 7442.46it/s]  7%|▋         | 27646/400000 [00:04<00:50, 7399.95it/s]  7%|▋         | 28402/400000 [00:04<00:49, 7445.18it/s]  7%|▋         | 29148/400000 [00:04<00:49, 7433.05it/s]  7%|▋         | 29917/400000 [00:04<00:49, 7505.85it/s]  8%|▊         | 30690/400000 [00:04<00:48, 7571.53it/s]  8%|▊         | 31461/400000 [00:04<00:48, 7610.28it/s]  8%|▊         | 32223/400000 [00:04<00:48, 7560.31it/s]  8%|▊         | 32980/400000 [00:04<00:48, 7509.33it/s]  8%|▊         | 33744/400000 [00:04<00:48, 7547.98it/s]  9%|▊         | 34515/400000 [00:04<00:48, 7594.79it/s]  9%|▉         | 35275/400000 [00:05<00:48, 7594.67it/s]  9%|▉         | 36035/400000 [00:05<00:47, 7585.12it/s]  9%|▉         | 36794/400000 [00:05<00:50, 7196.88it/s]  9%|▉         | 37518/400000 [00:05<00:51, 7087.04it/s] 10%|▉         | 38278/400000 [00:05<00:50, 7231.91it/s] 10%|▉         | 39005/400000 [00:05<00:49, 7233.89it/s] 10%|▉         | 39757/400000 [00:05<00:49, 7314.98it/s] 10%|█         | 40510/400000 [00:05<00:48, 7377.94it/s] 10%|█         | 41250/400000 [00:05<00:48, 7344.38it/s] 10%|█         | 41986/400000 [00:06<00:50, 7159.04it/s] 11%|█         | 42739/400000 [00:06<00:49, 7265.80it/s] 11%|█         | 43480/400000 [00:06<00:48, 7307.50it/s] 11%|█         | 44212/400000 [00:06<00:50, 7080.55it/s] 11%|█         | 44923/400000 [00:06<00:51, 6945.88it/s] 11%|█▏        | 45620/400000 [00:06<00:51, 6844.33it/s] 12%|█▏        | 46327/400000 [00:06<00:51, 6909.57it/s] 12%|█▏        | 47079/400000 [00:06<00:49, 7079.83it/s] 12%|█▏        | 47861/400000 [00:06<00:48, 7284.46it/s] 12%|█▏        | 48593/400000 [00:06<00:48, 7186.54it/s] 12%|█▏        | 49315/400000 [00:07<00:49, 7037.42it/s] 13%|█▎        | 50022/400000 [00:07<00:50, 6983.85it/s] 13%|█▎        | 50757/400000 [00:07<00:49, 7089.14it/s] 13%|█▎        | 51524/400000 [00:07<00:48, 7251.46it/s] 13%|█▎        | 52283/400000 [00:07<00:47, 7349.14it/s] 13%|█▎        | 53042/400000 [00:07<00:46, 7417.35it/s] 13%|█▎        | 53786/400000 [00:07<00:47, 7325.98it/s] 14%|█▎        | 54544/400000 [00:07<00:46, 7399.53it/s] 14%|█▍        | 55317/400000 [00:07<00:45, 7494.37it/s] 14%|█▍        | 56083/400000 [00:07<00:45, 7542.74it/s] 14%|█▍        | 56839/400000 [00:08<00:45, 7547.44it/s] 14%|█▍        | 57595/400000 [00:08<00:46, 7373.93it/s] 15%|█▍        | 58334/400000 [00:08<00:46, 7313.02it/s] 15%|█▍        | 59076/400000 [00:08<00:46, 7342.94it/s] 15%|█▍        | 59814/400000 [00:08<00:46, 7352.14it/s] 15%|█▌        | 60561/400000 [00:08<00:45, 7385.73it/s] 15%|█▌        | 61300/400000 [00:08<00:46, 7351.58it/s] 16%|█▌        | 62061/400000 [00:08<00:45, 7425.75it/s] 16%|█▌        | 62804/400000 [00:08<00:45, 7416.07it/s] 16%|█▌        | 63546/400000 [00:08<00:46, 7302.52it/s] 16%|█▌        | 64311/400000 [00:09<00:45, 7402.12it/s] 16%|█▋        | 65052/400000 [00:09<00:45, 7379.75it/s] 16%|█▋        | 65791/400000 [00:09<00:45, 7300.05it/s] 17%|█▋        | 66562/400000 [00:09<00:44, 7415.44it/s] 17%|█▋        | 67305/400000 [00:09<00:45, 7393.14it/s] 17%|█▋        | 68050/400000 [00:09<00:44, 7408.18it/s] 17%|█▋        | 68815/400000 [00:09<00:44, 7475.76it/s] 17%|█▋        | 69564/400000 [00:09<00:44, 7438.48it/s] 18%|█▊        | 70339/400000 [00:09<00:43, 7528.28it/s] 18%|█▊        | 71093/400000 [00:09<00:44, 7471.18it/s] 18%|█▊        | 71841/400000 [00:10<00:44, 7449.12it/s] 18%|█▊        | 72600/400000 [00:10<00:43, 7490.62it/s] 18%|█▊        | 73350/400000 [00:10<00:43, 7453.14it/s] 19%|█▊        | 74118/400000 [00:10<00:43, 7519.31it/s] 19%|█▊        | 74871/400000 [00:10<00:43, 7443.89it/s] 19%|█▉        | 75616/400000 [00:10<00:44, 7268.63it/s] 19%|█▉        | 76361/400000 [00:10<00:44, 7319.92it/s] 19%|█▉        | 77113/400000 [00:10<00:43, 7376.80it/s] 19%|█▉        | 77852/400000 [00:10<00:44, 7313.88it/s] 20%|█▉        | 78585/400000 [00:10<00:43, 7315.56it/s] 20%|█▉        | 79339/400000 [00:11<00:43, 7379.68it/s] 20%|██        | 80078/400000 [00:11<00:43, 7319.57it/s] 20%|██        | 80811/400000 [00:11<00:45, 7034.42it/s] 20%|██        | 81554/400000 [00:11<00:44, 7148.58it/s] 21%|██        | 82293/400000 [00:11<00:44, 7217.31it/s] 21%|██        | 83017/400000 [00:11<00:44, 7154.98it/s] 21%|██        | 83767/400000 [00:11<00:43, 7254.48it/s] 21%|██        | 84494/400000 [00:11<00:43, 7245.20it/s] 21%|██▏       | 85235/400000 [00:11<00:43, 7291.16it/s] 21%|██▏       | 85990/400000 [00:12<00:42, 7366.88it/s] 22%|██▏       | 86728/400000 [00:12<00:42, 7294.53it/s] 22%|██▏       | 87493/400000 [00:12<00:42, 7395.22it/s] 22%|██▏       | 88245/400000 [00:12<00:41, 7429.55it/s] 22%|██▏       | 88992/400000 [00:12<00:41, 7441.05it/s] 22%|██▏       | 89742/400000 [00:12<00:41, 7452.00it/s] 23%|██▎       | 90488/400000 [00:12<00:43, 7097.43it/s] 23%|██▎       | 91202/400000 [00:12<00:43, 7042.34it/s] 23%|██▎       | 91929/400000 [00:12<00:43, 7108.09it/s] 23%|██▎       | 92688/400000 [00:12<00:42, 7244.43it/s] 23%|██▎       | 93421/400000 [00:13<00:42, 7267.86it/s] 24%|██▎       | 94184/400000 [00:13<00:41, 7370.90it/s] 24%|██▎       | 94950/400000 [00:13<00:40, 7453.01it/s] 24%|██▍       | 95697/400000 [00:13<00:40, 7430.30it/s] 24%|██▍       | 96455/400000 [00:13<00:40, 7471.55it/s] 24%|██▍       | 97203/400000 [00:13<00:42, 7205.39it/s] 24%|██▍       | 97927/400000 [00:13<00:42, 7116.97it/s] 25%|██▍       | 98669/400000 [00:13<00:41, 7203.75it/s] 25%|██▍       | 99422/400000 [00:13<00:41, 7296.37it/s] 25%|██▌       | 100184/400000 [00:13<00:40, 7389.34it/s] 25%|██▌       | 100945/400000 [00:14<00:40, 7451.98it/s] 25%|██▌       | 101692/400000 [00:14<00:41, 7254.24it/s] 26%|██▌       | 102420/400000 [00:14<00:41, 7105.01it/s] 26%|██▌       | 103133/400000 [00:14<00:42, 6907.78it/s] 26%|██▌       | 103883/400000 [00:14<00:41, 7074.18it/s] 26%|██▌       | 104594/400000 [00:14<00:42, 6954.68it/s] 26%|██▋       | 105338/400000 [00:14<00:41, 7091.50it/s] 27%|██▋       | 106071/400000 [00:14<00:41, 7158.83it/s] 27%|██▋       | 106802/400000 [00:14<00:40, 7201.50it/s] 27%|██▋       | 107575/400000 [00:14<00:39, 7350.17it/s] 27%|██▋       | 108337/400000 [00:15<00:39, 7427.49it/s] 27%|██▋       | 109103/400000 [00:15<00:38, 7493.04it/s] 27%|██▋       | 109864/400000 [00:15<00:38, 7527.59it/s] 28%|██▊       | 110618/400000 [00:15<00:39, 7298.45it/s] 28%|██▊       | 111383/400000 [00:15<00:39, 7398.76it/s] 28%|██▊       | 112125/400000 [00:15<00:38, 7389.25it/s] 28%|██▊       | 112882/400000 [00:15<00:38, 7439.87it/s] 28%|██▊       | 113627/400000 [00:15<00:38, 7396.93it/s] 29%|██▊       | 114368/400000 [00:15<00:40, 7080.49it/s] 29%|██▉       | 115101/400000 [00:16<00:39, 7151.34it/s] 29%|██▉       | 115840/400000 [00:16<00:39, 7221.03it/s] 29%|██▉       | 116565/400000 [00:16<00:40, 7037.81it/s] 29%|██▉       | 117343/400000 [00:16<00:39, 7244.77it/s] 30%|██▉       | 118083/400000 [00:16<00:38, 7289.46it/s] 30%|██▉       | 118837/400000 [00:16<00:38, 7361.08it/s] 30%|██▉       | 119587/400000 [00:16<00:37, 7401.26it/s] 30%|███       | 120337/400000 [00:16<00:37, 7428.83it/s] 30%|███       | 121099/400000 [00:16<00:37, 7482.78it/s] 30%|███       | 121849/400000 [00:16<00:37, 7434.56it/s] 31%|███       | 122594/400000 [00:17<00:37, 7418.21it/s] 31%|███       | 123337/400000 [00:17<00:38, 7181.16it/s] 31%|███       | 124058/400000 [00:17<00:38, 7079.96it/s] 31%|███       | 124821/400000 [00:17<00:38, 7235.77it/s] 31%|███▏      | 125547/400000 [00:17<00:38, 7184.29it/s] 32%|███▏      | 126278/400000 [00:17<00:37, 7219.52it/s] 32%|███▏      | 127002/400000 [00:17<00:38, 7075.02it/s] 32%|███▏      | 127740/400000 [00:17<00:38, 7162.51it/s] 32%|███▏      | 128510/400000 [00:17<00:37, 7315.67it/s] 32%|███▏      | 129244/400000 [00:17<00:37, 7139.87it/s] 33%|███▎      | 130015/400000 [00:18<00:36, 7299.86it/s] 33%|███▎      | 130777/400000 [00:18<00:36, 7391.80it/s] 33%|███▎      | 131522/400000 [00:18<00:36, 7408.29it/s] 33%|███▎      | 132265/400000 [00:18<00:36, 7324.85it/s] 33%|███▎      | 132999/400000 [00:18<00:36, 7323.08it/s] 33%|███▎      | 133738/400000 [00:18<00:36, 7341.53it/s] 34%|███▎      | 134473/400000 [00:18<00:36, 7253.69it/s] 34%|███▍      | 135229/400000 [00:18<00:36, 7340.64it/s] 34%|███▍      | 135964/400000 [00:18<00:36, 7226.62it/s] 34%|███▍      | 136705/400000 [00:18<00:36, 7279.04it/s] 34%|███▍      | 137459/400000 [00:19<00:35, 7354.27it/s] 35%|███▍      | 138223/400000 [00:19<00:35, 7437.23it/s] 35%|███▍      | 138968/400000 [00:19<00:35, 7333.26it/s] 35%|███▍      | 139732/400000 [00:19<00:35, 7421.46it/s] 35%|███▌      | 140475/400000 [00:19<00:35, 7373.50it/s] 35%|███▌      | 141215/400000 [00:19<00:35, 7381.09it/s] 35%|███▌      | 141955/400000 [00:19<00:34, 7386.12it/s] 36%|███▌      | 142723/400000 [00:19<00:34, 7469.73it/s] 36%|███▌      | 143471/400000 [00:19<00:34, 7410.59it/s] 36%|███▌      | 144213/400000 [00:19<00:35, 7264.86it/s] 36%|███▌      | 144941/400000 [00:20<00:36, 7025.39it/s] 36%|███▋      | 145697/400000 [00:20<00:35, 7175.21it/s] 37%|███▋      | 146431/400000 [00:20<00:35, 7223.61it/s] 37%|███▋      | 147181/400000 [00:20<00:34, 7304.09it/s] 37%|███▋      | 147934/400000 [00:20<00:34, 7369.77it/s] 37%|███▋      | 148673/400000 [00:20<00:34, 7320.85it/s] 37%|███▋      | 149421/400000 [00:20<00:34, 7364.78it/s] 38%|███▊      | 150171/400000 [00:20<00:33, 7402.70it/s] 38%|███▊      | 150912/400000 [00:20<00:34, 7278.05it/s] 38%|███▊      | 151641/400000 [00:21<00:34, 7117.02it/s] 38%|███▊      | 152355/400000 [00:21<00:35, 7044.65it/s] 38%|███▊      | 153061/400000 [00:21<00:35, 7031.12it/s] 38%|███▊      | 153814/400000 [00:21<00:34, 7172.34it/s] 39%|███▊      | 154534/400000 [00:21<00:34, 7179.05it/s] 39%|███▉      | 155255/400000 [00:21<00:34, 7186.45it/s] 39%|███▉      | 155992/400000 [00:21<00:33, 7239.81it/s] 39%|███▉      | 156743/400000 [00:21<00:33, 7318.79it/s] 39%|███▉      | 157476/400000 [00:21<00:33, 7166.70it/s] 40%|███▉      | 158214/400000 [00:21<00:33, 7227.55it/s] 40%|███▉      | 158953/400000 [00:22<00:33, 7273.39it/s] 40%|███▉      | 159710/400000 [00:22<00:32, 7358.58it/s] 40%|████      | 160447/400000 [00:22<00:32, 7314.37it/s] 40%|████      | 161196/400000 [00:22<00:32, 7363.70it/s] 40%|████      | 161933/400000 [00:22<00:32, 7352.31it/s] 41%|████      | 162696/400000 [00:22<00:31, 7432.21it/s] 41%|████      | 163452/400000 [00:22<00:31, 7468.39it/s] 41%|████      | 164223/400000 [00:22<00:31, 7537.58it/s] 41%|████      | 164978/400000 [00:22<00:31, 7537.97it/s] 41%|████▏     | 165733/400000 [00:22<00:31, 7529.31it/s] 42%|████▏     | 166487/400000 [00:23<00:32, 7262.65it/s] 42%|████▏     | 167253/400000 [00:23<00:31, 7374.97it/s] 42%|████▏     | 168016/400000 [00:23<00:31, 7448.52it/s] 42%|████▏     | 168768/400000 [00:23<00:30, 7467.52it/s] 42%|████▏     | 169528/400000 [00:23<00:30, 7504.48it/s] 43%|████▎     | 170280/400000 [00:23<00:30, 7433.44it/s] 43%|████▎     | 171040/400000 [00:23<00:30, 7480.59it/s] 43%|████▎     | 171789/400000 [00:23<00:30, 7454.94it/s] 43%|████▎     | 172543/400000 [00:23<00:30, 7479.02it/s] 43%|████▎     | 173292/400000 [00:23<00:30, 7453.90it/s] 44%|████▎     | 174044/400000 [00:24<00:30, 7471.97it/s] 44%|████▎     | 174792/400000 [00:24<00:30, 7466.94it/s] 44%|████▍     | 175559/400000 [00:24<00:29, 7525.97it/s] 44%|████▍     | 176330/400000 [00:24<00:29, 7578.95it/s] 44%|████▍     | 177089/400000 [00:24<00:29, 7576.50it/s] 44%|████▍     | 177860/400000 [00:24<00:29, 7615.04it/s] 45%|████▍     | 178622/400000 [00:24<00:29, 7394.25it/s] 45%|████▍     | 179363/400000 [00:24<00:30, 7135.23it/s] 45%|████▌     | 180080/400000 [00:24<00:31, 7013.67it/s] 45%|████▌     | 180784/400000 [00:24<00:31, 6922.28it/s] 45%|████▌     | 181523/400000 [00:25<00:30, 7055.18it/s] 46%|████▌     | 182244/400000 [00:25<00:30, 7100.30it/s] 46%|████▌     | 182990/400000 [00:25<00:30, 7202.29it/s] 46%|████▌     | 183725/400000 [00:25<00:29, 7244.68it/s] 46%|████▌     | 184451/400000 [00:25<00:29, 7243.34it/s] 46%|████▋     | 185210/400000 [00:25<00:29, 7343.02it/s] 46%|████▋     | 185970/400000 [00:25<00:28, 7416.67it/s] 47%|████▋     | 186733/400000 [00:25<00:28, 7477.62it/s] 47%|████▋     | 187482/400000 [00:25<00:28, 7442.20it/s] 47%|████▋     | 188233/400000 [00:25<00:28, 7459.75it/s] 47%|████▋     | 189001/400000 [00:26<00:28, 7522.25it/s] 47%|████▋     | 189754/400000 [00:26<00:28, 7479.41it/s] 48%|████▊     | 190505/400000 [00:26<00:27, 7486.45it/s] 48%|████▊     | 191254/400000 [00:26<00:27, 7482.20it/s] 48%|████▊     | 192003/400000 [00:26<00:28, 7303.96it/s] 48%|████▊     | 192743/400000 [00:26<00:28, 7330.97it/s] 48%|████▊     | 193499/400000 [00:26<00:27, 7398.04it/s] 49%|████▊     | 194264/400000 [00:26<00:27, 7469.08it/s] 49%|████▉     | 195012/400000 [00:26<00:27, 7403.27it/s] 49%|████▉     | 195753/400000 [00:27<00:28, 7043.08it/s] 49%|████▉     | 196497/400000 [00:27<00:28, 7155.19it/s] 49%|████▉     | 197254/400000 [00:27<00:27, 7272.52it/s] 49%|████▉     | 197985/400000 [00:27<00:27, 7282.92it/s] 50%|████▉     | 198743/400000 [00:27<00:27, 7367.58it/s] 50%|████▉     | 199482/400000 [00:27<00:27, 7310.14it/s] 50%|█████     | 200215/400000 [00:27<00:28, 7046.38it/s] 50%|█████     | 200948/400000 [00:27<00:27, 7126.72it/s] 50%|█████     | 201714/400000 [00:27<00:27, 7273.43it/s] 51%|█████     | 202444/400000 [00:27<00:27, 7253.81it/s] 51%|█████     | 203172/400000 [00:28<00:27, 7190.06it/s] 51%|█████     | 203893/400000 [00:28<00:27, 7018.22it/s] 51%|█████     | 204630/400000 [00:28<00:27, 7118.22it/s] 51%|█████▏    | 205399/400000 [00:28<00:26, 7278.98it/s] 52%|█████▏    | 206130/400000 [00:28<00:26, 7287.07it/s] 52%|█████▏    | 206861/400000 [00:28<00:27, 7121.57it/s] 52%|█████▏    | 207575/400000 [00:28<00:27, 6997.17it/s] 52%|█████▏    | 208277/400000 [00:28<00:27, 6911.06it/s] 52%|█████▏    | 208970/400000 [00:28<00:27, 6827.31it/s] 52%|█████▏    | 209707/400000 [00:28<00:27, 6979.53it/s] 53%|█████▎    | 210407/400000 [00:29<00:27, 6979.22it/s] 53%|█████▎    | 211167/400000 [00:29<00:26, 7153.28it/s] 53%|█████▎    | 211894/400000 [00:29<00:26, 7187.36it/s] 53%|█████▎    | 212651/400000 [00:29<00:25, 7295.58it/s] 53%|█████▎    | 213382/400000 [00:29<00:25, 7254.42it/s] 54%|█████▎    | 214109/400000 [00:29<00:25, 7199.43it/s] 54%|█████▎    | 214866/400000 [00:29<00:25, 7304.85it/s] 54%|█████▍    | 215628/400000 [00:29<00:24, 7396.34it/s] 54%|█████▍    | 216392/400000 [00:29<00:24, 7466.54it/s] 54%|█████▍    | 217149/400000 [00:29<00:24, 7495.00it/s] 54%|█████▍    | 217900/400000 [00:30<00:25, 7243.71it/s] 55%|█████▍    | 218627/400000 [00:30<00:25, 7135.38it/s] 55%|█████▍    | 219343/400000 [00:30<00:25, 7020.27it/s] 55%|█████▌    | 220047/400000 [00:30<00:25, 7000.82it/s] 55%|█████▌    | 220806/400000 [00:30<00:25, 7167.71it/s] 55%|█████▌    | 221530/400000 [00:30<00:24, 7187.36it/s] 56%|█████▌    | 222288/400000 [00:30<00:24, 7298.76it/s] 56%|█████▌    | 223049/400000 [00:30<00:23, 7388.89it/s] 56%|█████▌    | 223816/400000 [00:30<00:23, 7469.23it/s] 56%|█████▌    | 224580/400000 [00:31<00:23, 7517.25it/s] 56%|█████▋    | 225333/400000 [00:31<00:24, 7120.55it/s] 57%|█████▋    | 226050/400000 [00:31<00:25, 6953.34it/s] 57%|█████▋    | 226782/400000 [00:31<00:24, 7058.93it/s] 57%|█████▋    | 227546/400000 [00:31<00:23, 7221.20it/s] 57%|█████▋    | 228296/400000 [00:31<00:23, 7301.12it/s] 57%|█████▋    | 229029/400000 [00:31<00:24, 7114.75it/s] 57%|█████▋    | 229744/400000 [00:31<00:24, 6997.00it/s] 58%|█████▊    | 230447/400000 [00:31<00:24, 6898.12it/s] 58%|█████▊    | 231162/400000 [00:31<00:24, 6970.54it/s] 58%|█████▊    | 231915/400000 [00:32<00:23, 7128.79it/s] 58%|█████▊    | 232657/400000 [00:32<00:23, 7213.65it/s] 58%|█████▊    | 233387/400000 [00:32<00:23, 7237.77it/s] 59%|█████▊    | 234112/400000 [00:32<00:23, 7025.17it/s] 59%|█████▊    | 234825/400000 [00:32<00:23, 7053.15it/s] 59%|█████▉    | 235532/400000 [00:32<00:23, 6981.38it/s] 59%|█████▉    | 236232/400000 [00:32<00:24, 6718.40it/s] 59%|█████▉    | 236907/400000 [00:32<00:24, 6619.84it/s] 59%|█████▉    | 237633/400000 [00:32<00:23, 6796.93it/s] 60%|█████▉    | 238316/400000 [00:32<00:23, 6748.25it/s] 60%|█████▉    | 238993/400000 [00:33<00:24, 6654.52it/s] 60%|█████▉    | 239752/400000 [00:33<00:23, 6908.05it/s] 60%|██████    | 240456/400000 [00:33<00:22, 6945.70it/s] 60%|██████    | 241154/400000 [00:33<00:23, 6890.73it/s] 60%|██████    | 241874/400000 [00:33<00:22, 6980.46it/s] 61%|██████    | 242621/400000 [00:33<00:22, 7117.84it/s] 61%|██████    | 243335/400000 [00:33<00:22, 7107.85it/s] 61%|██████    | 244094/400000 [00:33<00:21, 7244.80it/s] 61%|██████    | 244866/400000 [00:33<00:21, 7380.38it/s] 61%|██████▏   | 245630/400000 [00:34<00:20, 7456.27it/s] 62%|██████▏   | 246391/400000 [00:34<00:20, 7501.71it/s] 62%|██████▏   | 247143/400000 [00:34<00:20, 7411.90it/s] 62%|██████▏   | 247886/400000 [00:34<00:20, 7398.08it/s] 62%|██████▏   | 248647/400000 [00:34<00:20, 7460.01it/s] 62%|██████▏   | 249409/400000 [00:34<00:20, 7505.30it/s] 63%|██████▎   | 250161/400000 [00:34<00:19, 7500.64it/s] 63%|██████▎   | 250912/400000 [00:34<00:20, 7397.15it/s] 63%|██████▎   | 251653/400000 [00:34<00:21, 7040.19it/s] 63%|██████▎   | 252362/400000 [00:34<00:21, 6987.56it/s] 63%|██████▎   | 253064/400000 [00:35<00:21, 6942.54it/s] 63%|██████▎   | 253783/400000 [00:35<00:20, 7013.63it/s] 64%|██████▎   | 254536/400000 [00:35<00:20, 7160.89it/s] 64%|██████▍   | 255288/400000 [00:35<00:19, 7263.76it/s] 64%|██████▍   | 256046/400000 [00:35<00:19, 7354.11it/s] 64%|██████▍   | 256783/400000 [00:35<00:20, 7072.43it/s] 64%|██████▍   | 257494/400000 [00:35<00:20, 6887.63it/s] 65%|██████▍   | 258256/400000 [00:35<00:19, 7089.89it/s] 65%|██████▍   | 259005/400000 [00:35<00:19, 7203.50it/s] 65%|██████▍   | 259729/400000 [00:35<00:19, 7068.16it/s] 65%|██████▌   | 260442/400000 [00:36<00:19, 7084.39it/s] 65%|██████▌   | 261205/400000 [00:36<00:19, 7239.24it/s] 65%|██████▌   | 261932/400000 [00:36<00:19, 7015.26it/s] 66%|██████▌   | 262637/400000 [00:36<00:19, 7014.93it/s] 66%|██████▌   | 263404/400000 [00:36<00:18, 7197.44it/s] 66%|██████▌   | 264167/400000 [00:36<00:18, 7321.63it/s] 66%|██████▌   | 264902/400000 [00:36<00:18, 7317.55it/s] 66%|██████▋   | 265636/400000 [00:36<00:18, 7201.62it/s] 67%|██████▋   | 266396/400000 [00:36<00:18, 7314.45it/s] 67%|██████▋   | 267166/400000 [00:36<00:17, 7424.59it/s] 67%|██████▋   | 267928/400000 [00:37<00:17, 7481.53it/s] 67%|██████▋   | 268678/400000 [00:37<00:17, 7354.79it/s] 67%|██████▋   | 269440/400000 [00:37<00:17, 7430.86it/s] 68%|██████▊   | 270190/400000 [00:37<00:17, 7450.13it/s] 68%|██████▊   | 270936/400000 [00:37<00:17, 7411.70it/s] 68%|██████▊   | 271678/400000 [00:37<00:17, 7393.59it/s] 68%|██████▊   | 272418/400000 [00:37<00:17, 7370.25it/s] 68%|██████▊   | 273156/400000 [00:37<00:17, 7335.40it/s] 68%|██████▊   | 273915/400000 [00:37<00:17, 7408.01it/s] 69%|██████▊   | 274657/400000 [00:38<00:16, 7409.86it/s] 69%|██████▉   | 275405/400000 [00:38<00:16, 7429.15it/s] 69%|██████▉   | 276156/400000 [00:38<00:16, 7451.43it/s] 69%|██████▉   | 276902/400000 [00:38<00:16, 7357.56it/s] 69%|██████▉   | 277639/400000 [00:38<00:16, 7294.23it/s] 70%|██████▉   | 278402/400000 [00:38<00:16, 7391.33it/s] 70%|██████▉   | 279169/400000 [00:38<00:16, 7472.65it/s] 70%|██████▉   | 279917/400000 [00:38<00:16, 7396.39it/s] 70%|███████   | 280676/400000 [00:38<00:16, 7452.43it/s] 70%|███████   | 281434/400000 [00:38<00:15, 7489.18it/s] 71%|███████   | 282185/400000 [00:39<00:15, 7494.79it/s] 71%|███████   | 282935/400000 [00:39<00:15, 7392.18it/s] 71%|███████   | 283675/400000 [00:39<00:15, 7381.12it/s] 71%|███████   | 284430/400000 [00:39<00:15, 7428.72it/s] 71%|███████▏  | 285180/400000 [00:39<00:15, 7448.97it/s] 71%|███████▏  | 285927/400000 [00:39<00:15, 7454.70it/s] 72%|███████▏  | 286673/400000 [00:39<00:15, 7448.32it/s] 72%|███████▏  | 287419/400000 [00:39<00:15, 7450.88it/s] 72%|███████▏  | 288175/400000 [00:39<00:14, 7483.04it/s] 72%|███████▏  | 288937/400000 [00:39<00:14, 7521.46it/s] 72%|███████▏  | 289694/400000 [00:40<00:14, 7529.88it/s] 73%|███████▎  | 290448/400000 [00:40<00:14, 7511.70it/s] 73%|███████▎  | 291200/400000 [00:40<00:14, 7393.96it/s] 73%|███████▎  | 291940/400000 [00:40<00:15, 7131.76it/s] 73%|███████▎  | 292656/400000 [00:40<00:15, 7060.74it/s] 73%|███████▎  | 293364/400000 [00:40<00:15, 6971.86it/s] 74%|███████▎  | 294063/400000 [00:40<00:15, 6899.41it/s] 74%|███████▎  | 294755/400000 [00:40<00:15, 6667.89it/s] 74%|███████▍  | 295512/400000 [00:40<00:15, 6913.78it/s] 74%|███████▍  | 296274/400000 [00:40<00:14, 7110.65it/s] 74%|███████▍  | 297023/400000 [00:41<00:14, 7220.00it/s] 74%|███████▍  | 297769/400000 [00:41<00:14, 7288.73it/s] 75%|███████▍  | 298501/400000 [00:41<00:14, 7239.37it/s] 75%|███████▍  | 299227/400000 [00:41<00:13, 7240.52it/s] 75%|███████▍  | 299984/400000 [00:41<00:13, 7332.61it/s] 75%|███████▌  | 300743/400000 [00:41<00:13, 7407.96it/s] 75%|███████▌  | 301485/400000 [00:41<00:13, 7151.47it/s] 76%|███████▌  | 302236/400000 [00:41<00:13, 7252.94it/s] 76%|███████▌  | 302983/400000 [00:41<00:13, 7315.84it/s] 76%|███████▌  | 303738/400000 [00:41<00:13, 7383.06it/s] 76%|███████▌  | 304500/400000 [00:42<00:12, 7451.55it/s] 76%|███████▋  | 305258/400000 [00:42<00:12, 7487.90it/s] 77%|███████▋  | 306008/400000 [00:42<00:12, 7430.40it/s] 77%|███████▋  | 306768/400000 [00:42<00:12, 7478.84it/s] 77%|███████▋  | 307517/400000 [00:42<00:12, 7459.62it/s] 77%|███████▋  | 308273/400000 [00:42<00:12, 7484.25it/s] 77%|███████▋  | 309035/400000 [00:42<00:12, 7522.19it/s] 77%|███████▋  | 309802/400000 [00:42<00:11, 7564.24it/s] 78%|███████▊  | 310559/400000 [00:42<00:11, 7563.20it/s] 78%|███████▊  | 311316/400000 [00:42<00:11, 7536.48it/s] 78%|███████▊  | 312088/400000 [00:43<00:11, 7587.98it/s] 78%|███████▊  | 312857/400000 [00:43<00:11, 7617.52it/s] 78%|███████▊  | 313623/400000 [00:43<00:11, 7627.87it/s] 79%|███████▊  | 314386/400000 [00:43<00:11, 7286.38it/s] 79%|███████▉  | 315118/400000 [00:43<00:11, 7139.87it/s] 79%|███████▉  | 315835/400000 [00:43<00:12, 6757.00it/s] 79%|███████▉  | 316518/400000 [00:43<00:12, 6689.21it/s] 79%|███████▉  | 317266/400000 [00:43<00:11, 6907.38it/s] 79%|███████▉  | 317998/400000 [00:43<00:11, 7023.42it/s] 80%|███████▉  | 318705/400000 [00:44<00:11, 6980.75it/s] 80%|███████▉  | 319420/400000 [00:44<00:11, 7030.57it/s] 80%|████████  | 320163/400000 [00:44<00:11, 7143.95it/s] 80%|████████  | 320915/400000 [00:44<00:10, 7250.57it/s] 80%|████████  | 321668/400000 [00:44<00:10, 7329.78it/s] 81%|████████  | 322439/400000 [00:44<00:10, 7439.65it/s] 81%|████████  | 323199/400000 [00:44<00:10, 7486.70it/s] 81%|████████  | 323949/400000 [00:44<00:10, 7454.79it/s] 81%|████████  | 324696/400000 [00:44<00:10, 7209.31it/s] 81%|████████▏ | 325420/400000 [00:44<00:10, 7124.29it/s] 82%|████████▏ | 326135/400000 [00:45<00:10, 7034.20it/s] 82%|████████▏ | 326863/400000 [00:45<00:10, 7104.13it/s] 82%|████████▏ | 327615/400000 [00:45<00:10, 7222.29it/s] 82%|████████▏ | 328339/400000 [00:45<00:09, 7208.30it/s] 82%|████████▏ | 329080/400000 [00:45<00:09, 7265.31it/s] 82%|████████▏ | 329843/400000 [00:45<00:09, 7370.78it/s] 83%|████████▎ | 330586/400000 [00:45<00:09, 7387.35it/s] 83%|████████▎ | 331326/400000 [00:45<00:09, 7261.75it/s] 83%|████████▎ | 332058/400000 [00:45<00:09, 7277.27it/s] 83%|████████▎ | 332787/400000 [00:45<00:09, 7084.35it/s] 83%|████████▎ | 333498/400000 [00:46<00:09, 7017.73it/s] 84%|████████▎ | 334202/400000 [00:46<00:09, 6979.41it/s] 84%|████████▎ | 334975/400000 [00:46<00:09, 7188.41it/s] 84%|████████▍ | 335697/400000 [00:46<00:08, 7174.25it/s] 84%|████████▍ | 336416/400000 [00:46<00:08, 7087.33it/s] 84%|████████▍ | 337178/400000 [00:46<00:08, 7238.17it/s] 84%|████████▍ | 337946/400000 [00:46<00:08, 7364.42it/s] 85%|████████▍ | 338718/400000 [00:46<00:08, 7465.50it/s] 85%|████████▍ | 339467/400000 [00:46<00:08, 7463.08it/s] 85%|████████▌ | 340232/400000 [00:46<00:07, 7516.13it/s] 85%|████████▌ | 340985/400000 [00:47<00:07, 7446.87it/s] 85%|████████▌ | 341731/400000 [00:47<00:08, 7143.96it/s] 86%|████████▌ | 342449/400000 [00:47<00:08, 7053.48it/s] 86%|████████▌ | 343179/400000 [00:47<00:07, 7125.21it/s] 86%|████████▌ | 343940/400000 [00:47<00:07, 7262.31it/s] 86%|████████▌ | 344692/400000 [00:47<00:07, 7336.83it/s] 86%|████████▋ | 345428/400000 [00:47<00:07, 7220.79it/s] 87%|████████▋ | 346192/400000 [00:47<00:07, 7341.25it/s] 87%|████████▋ | 346928/400000 [00:47<00:07, 7336.88it/s] 87%|████████▋ | 347684/400000 [00:48<00:07, 7399.84it/s] 87%|████████▋ | 348444/400000 [00:48<00:06, 7457.29it/s] 87%|████████▋ | 349191/400000 [00:48<00:06, 7452.51it/s] 87%|████████▋ | 349948/400000 [00:48<00:06, 7486.07it/s] 88%|████████▊ | 350702/400000 [00:48<00:06, 7500.13it/s] 88%|████████▊ | 351453/400000 [00:48<00:06, 7137.32it/s] 88%|████████▊ | 352184/400000 [00:48<00:06, 7185.28it/s] 88%|████████▊ | 352932/400000 [00:48<00:06, 7270.67it/s] 88%|████████▊ | 353696/400000 [00:48<00:06, 7360.32it/s] 89%|████████▊ | 354434/400000 [00:48<00:06, 7275.09it/s] 89%|████████▉ | 355189/400000 [00:49<00:06, 7354.67it/s] 89%|████████▉ | 355951/400000 [00:49<00:05, 7430.97it/s] 89%|████████▉ | 356696/400000 [00:49<00:05, 7392.86it/s] 89%|████████▉ | 357437/400000 [00:49<00:05, 7200.75it/s] 90%|████████▉ | 358159/400000 [00:49<00:06, 6855.90it/s] 90%|████████▉ | 358883/400000 [00:49<00:05, 6966.56it/s] 90%|████████▉ | 359631/400000 [00:49<00:05, 7112.16it/s] 90%|█████████ | 360393/400000 [00:49<00:05, 7255.30it/s] 90%|█████████ | 361147/400000 [00:49<00:05, 7338.33it/s] 90%|█████████ | 361899/400000 [00:49<00:05, 7390.53it/s] 91%|█████████ | 362640/400000 [00:50<00:05, 7084.95it/s] 91%|█████████ | 363353/400000 [00:50<00:05, 6997.50it/s] 91%|█████████ | 364056/400000 [00:50<00:05, 6864.35it/s] 91%|█████████ | 364808/400000 [00:50<00:04, 7046.27it/s] 91%|█████████▏| 365549/400000 [00:50<00:04, 7151.36it/s] 92%|█████████▏| 366267/400000 [00:50<00:04, 7140.34it/s] 92%|█████████▏| 367001/400000 [00:50<00:04, 7197.57it/s] 92%|█████████▏| 367763/400000 [00:50<00:04, 7316.90it/s] 92%|█████████▏| 368512/400000 [00:50<00:04, 7367.66it/s] 92%|█████████▏| 369256/400000 [00:50<00:04, 7387.94it/s] 92%|█████████▏| 369996/400000 [00:51<00:04, 7219.46it/s] 93%|█████████▎| 370756/400000 [00:51<00:03, 7328.01it/s] 93%|█████████▎| 371505/400000 [00:51<00:03, 7373.17it/s] 93%|█████████▎| 372269/400000 [00:51<00:03, 7451.19it/s] 93%|█████████▎| 373016/400000 [00:51<00:03, 7445.62it/s] 93%|█████████▎| 373771/400000 [00:51<00:03, 7476.32it/s] 94%|█████████▎| 374541/400000 [00:51<00:03, 7540.19it/s] 94%|█████████▍| 375296/400000 [00:51<00:03, 7355.38it/s] 94%|█████████▍| 376033/400000 [00:51<00:03, 7255.63it/s] 94%|█████████▍| 376787/400000 [00:52<00:03, 7337.82it/s] 94%|█████████▍| 377542/400000 [00:52<00:03, 7397.62it/s] 95%|█████████▍| 378283/400000 [00:52<00:02, 7349.11it/s] 95%|█████████▍| 379019/400000 [00:52<00:02, 7096.59it/s] 95%|█████████▍| 379732/400000 [00:52<00:02, 6911.24it/s] 95%|█████████▌| 380426/400000 [00:52<00:02, 6791.52it/s] 95%|█████████▌| 381108/400000 [00:52<00:02, 6779.64it/s] 95%|█████████▌| 381788/400000 [00:52<00:02, 6763.34it/s] 96%|█████████▌| 382544/400000 [00:52<00:02, 6983.39it/s] 96%|█████████▌| 383245/400000 [00:52<00:02, 6990.21it/s] 96%|█████████▌| 383946/400000 [00:53<00:02, 6900.33it/s] 96%|█████████▌| 384708/400000 [00:53<00:02, 7099.27it/s] 96%|█████████▋| 385479/400000 [00:53<00:01, 7271.07it/s] 97%|█████████▋| 386234/400000 [00:53<00:01, 7349.92it/s] 97%|█████████▋| 386980/400000 [00:53<00:01, 7380.51it/s] 97%|█████████▋| 387720/400000 [00:53<00:01, 7372.71it/s] 97%|█████████▋| 388459/400000 [00:53<00:01, 7152.25it/s] 97%|█████████▋| 389177/400000 [00:53<00:01, 7020.59it/s] 97%|█████████▋| 389886/400000 [00:53<00:01, 7038.68it/s] 98%|█████████▊| 390633/400000 [00:53<00:01, 7162.50it/s] 98%|█████████▊| 391351/400000 [00:54<00:01, 7074.66it/s] 98%|█████████▊| 392060/400000 [00:54<00:01, 6905.52it/s] 98%|█████████▊| 392753/400000 [00:54<00:01, 6801.47it/s] 98%|█████████▊| 393435/400000 [00:54<00:00, 6794.17it/s] 99%|█████████▊| 394174/400000 [00:54<00:00, 6960.61it/s] 99%|█████████▊| 394872/400000 [00:54<00:00, 6900.49it/s] 99%|█████████▉| 395564/400000 [00:54<00:00, 6856.26it/s] 99%|█████████▉| 396251/400000 [00:54<00:00, 6806.12it/s] 99%|█████████▉| 396933/400000 [00:54<00:00, 6800.63it/s] 99%|█████████▉| 397636/400000 [00:54<00:00, 6866.32it/s]100%|█████████▉| 398391/400000 [00:55<00:00, 7055.99it/s]100%|█████████▉| 399153/400000 [00:55<00:00, 7215.57it/s]100%|█████████▉| 399912/400000 [00:55<00:00, 7323.74it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7231.90it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f7afa2da518>, <torchtext.data.dataset.TabularDataset object at 0x7f7afa2da668>, <torchtext.vocab.Vocab object at 0x7f7afa2da588>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.77 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 20.60 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.38 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.38 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.46 file/s]2020-06-23 18:19:20.616345: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-23 18:19:20.620480: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-06-23 18:19:20.620747: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f8e0e6b1e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-23 18:19:20.620784: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3031040/9912422 [00:00<00:00, 30280035.53it/s]9920512it [00:00, 35442807.34it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1619081.29it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1023210.95it/s]1654784it [00:00, 12846307.32it/s]                           
0it [00:00, ?it/s]8192it [00:00, 249963.54it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
