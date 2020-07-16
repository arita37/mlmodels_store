
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe138ad8158> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe138ad8158> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe1a467f2f0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe1a467f2f0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe1c29c4e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe1c29c4e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe1519a6730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe1519a6730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe1519a6730> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:10, 140900.31it/s] 57%|█████▋    | 5619712/9912422 [00:00<00:21, 201067.71it/s]9920512it [00:00, 38460035.48it/s]                           
0it [00:00, ?it/s]32768it [00:00, 581705.40it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 466593.86it/s]1654784it [00:00, 11844263.17it/s]                         
0it [00:00, ?it/s]8192it [00:00, 187562.37it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe12f558ac8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe12f568d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe1519a6378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe1519a6378> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe1519a6378> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:00,  2.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:00,  2.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:00,  2.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:59,  2.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:59,  2.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:59,  2.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:58,  2.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:41,  3.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:41,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:41,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:41,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:40,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:40,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:40,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:40,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:39,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:28,  5.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:27,  5.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:27,  5.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:19,  7.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:19,  7.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:19,  7.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  7.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:13,  9.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:13,  9.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:13,  9.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:13,  9.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:13,  9.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12,  9.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12,  9.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:09, 13.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:09, 13.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:09, 13.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 13.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 13.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 13.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:09, 13.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 13.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 17.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 17.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 17.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 17.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 17.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 17.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 22.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 22.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 22.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 22.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 22.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 22.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 22.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 22.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 28.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 28.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 28.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 28.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 34.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 34.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 34.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 34.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 34.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 34.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 40.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 40.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 40.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 40.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 40.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 40.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 40.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 45.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 45.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 45.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 45.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 45.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 50.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 50.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 50.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 50.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 50.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 50.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 50.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 50.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 54.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 54.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 54.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 54.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 54.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 54.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 58.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 58.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 58.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 58.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 58.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 58.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 62.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 66.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 66.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 66.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 66.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 66.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 66.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 66.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 66.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 70.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 70.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 70.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 70.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 70.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 70.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 70.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 73.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 73.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 73.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 73.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.20 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 75.20 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.52 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ url]
0 examples [00:00, ? examples/s]2020-07-16 12:10:35.486701: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-16 12:10:35.501465: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-16 12:10:35.502168: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bd701ae830 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-16 12:10:35.502197: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
31 examples [00:00, 307.72 examples/s]119 examples [00:00, 381.76 examples/s]199 examples [00:00, 452.31 examples/s]292 examples [00:00, 534.33 examples/s]386 examples [00:00, 613.31 examples/s]475 examples [00:00, 674.66 examples/s]566 examples [00:00, 729.91 examples/s]656 examples [00:00, 773.15 examples/s]746 examples [00:00, 806.21 examples/s]838 examples [00:01, 835.76 examples/s]930 examples [00:01, 857.62 examples/s]1019 examples [00:01, 826.23 examples/s]1105 examples [00:01, 835.93 examples/s]1195 examples [00:01, 852.02 examples/s]1286 examples [00:01, 867.59 examples/s]1375 examples [00:01, 867.13 examples/s]1463 examples [00:01, 854.12 examples/s]1553 examples [00:01, 866.73 examples/s]1643 examples [00:01, 876.11 examples/s]1732 examples [00:02, 879.03 examples/s]1823 examples [00:02, 885.39 examples/s]1915 examples [00:02, 893.64 examples/s]2005 examples [00:02, 864.88 examples/s]2092 examples [00:02, 842.88 examples/s]2182 examples [00:02, 858.36 examples/s]2269 examples [00:02, 845.97 examples/s]2354 examples [00:02, 843.63 examples/s]2439 examples [00:02, 842.76 examples/s]2524 examples [00:02, 836.73 examples/s]2613 examples [00:03, 851.99 examples/s]2699 examples [00:03, 853.20 examples/s]2791 examples [00:03, 871.77 examples/s]2879 examples [00:03, 872.85 examples/s]2970 examples [00:03, 882.02 examples/s]3062 examples [00:03, 891.80 examples/s]3152 examples [00:03, 892.65 examples/s]3242 examples [00:03, 845.52 examples/s]3328 examples [00:03, 822.93 examples/s]3418 examples [00:04, 842.95 examples/s]3504 examples [00:04, 847.30 examples/s]3590 examples [00:04, 833.92 examples/s]3679 examples [00:04, 847.55 examples/s]3765 examples [00:04, 850.37 examples/s]3854 examples [00:04, 860.73 examples/s]3945 examples [00:04, 872.81 examples/s]4034 examples [00:04, 867.26 examples/s]4123 examples [00:04, 871.99 examples/s]4211 examples [00:04, 868.61 examples/s]4298 examples [00:05, 866.83 examples/s]4386 examples [00:05, 868.87 examples/s]4476 examples [00:05, 877.69 examples/s]4567 examples [00:05, 885.51 examples/s]4656 examples [00:05, 884.03 examples/s]4749 examples [00:05, 895.25 examples/s]4839 examples [00:05, 860.18 examples/s]4935 examples [00:05, 886.59 examples/s]5025 examples [00:05, 883.13 examples/s]5114 examples [00:05, 863.95 examples/s]5208 examples [00:06, 883.75 examples/s]5297 examples [00:06, 870.01 examples/s]5385 examples [00:06, 845.22 examples/s]5470 examples [00:06, 837.48 examples/s]5555 examples [00:06, 839.17 examples/s]5644 examples [00:06, 852.62 examples/s]5733 examples [00:06, 861.52 examples/s]5822 examples [00:06, 868.67 examples/s]5912 examples [00:06, 876.50 examples/s]6003 examples [00:06, 884.69 examples/s]6092 examples [00:07, 883.68 examples/s]6181 examples [00:07, 860.59 examples/s]6268 examples [00:07, 837.46 examples/s]6358 examples [00:07, 853.10 examples/s]6444 examples [00:07, 838.71 examples/s]6529 examples [00:07, 818.52 examples/s]6616 examples [00:07, 830.95 examples/s]6700 examples [00:07, 820.28 examples/s]6785 examples [00:07, 828.05 examples/s]6870 examples [00:08, 833.20 examples/s]6954 examples [00:08, 835.12 examples/s]7038 examples [00:08, 836.53 examples/s]7125 examples [00:08, 843.60 examples/s]7210 examples [00:08, 837.77 examples/s]7294 examples [00:08, 813.64 examples/s]7377 examples [00:08, 818.24 examples/s]7459 examples [00:08, 816.05 examples/s]7541 examples [00:08, 807.06 examples/s]7622 examples [00:08, 799.67 examples/s]7703 examples [00:09, 795.48 examples/s]7783 examples [00:09, 794.21 examples/s]7863 examples [00:09, 777.93 examples/s]7942 examples [00:09, 780.02 examples/s]8026 examples [00:09, 795.86 examples/s]8106 examples [00:09, 761.02 examples/s]8185 examples [00:09, 768.09 examples/s]8272 examples [00:09, 795.43 examples/s]8360 examples [00:09, 817.36 examples/s]8447 examples [00:09, 831.80 examples/s]8531 examples [00:10, 831.87 examples/s]8615 examples [00:10, 830.23 examples/s]8699 examples [00:10, 822.74 examples/s]8787 examples [00:10, 838.13 examples/s]8873 examples [00:10, 844.56 examples/s]8962 examples [00:10, 855.75 examples/s]9049 examples [00:10, 858.27 examples/s]9135 examples [00:10, 834.11 examples/s]9221 examples [00:10, 840.89 examples/s]9306 examples [00:11, 822.94 examples/s]9393 examples [00:11, 832.89 examples/s]9478 examples [00:11, 835.87 examples/s]9562 examples [00:11, 824.96 examples/s]9645 examples [00:11, 817.60 examples/s]9727 examples [00:11, 804.26 examples/s]9809 examples [00:11, 807.46 examples/s]9891 examples [00:11, 810.62 examples/s]9978 examples [00:11, 826.82 examples/s]10061 examples [00:11, 783.46 examples/s]10150 examples [00:12, 809.65 examples/s]10239 examples [00:12, 830.49 examples/s]10326 examples [00:12, 841.31 examples/s]10411 examples [00:12, 839.46 examples/s]10496 examples [00:12, 817.51 examples/s]10579 examples [00:12, 814.76 examples/s]10661 examples [00:12, 806.34 examples/s]10745 examples [00:12, 815.23 examples/s]10827 examples [00:12, 809.29 examples/s]10913 examples [00:12, 822.06 examples/s]10997 examples [00:13, 827.05 examples/s]11085 examples [00:13, 841.33 examples/s]11171 examples [00:13, 846.38 examples/s]11256 examples [00:13, 844.13 examples/s]11341 examples [00:13, 836.78 examples/s]11425 examples [00:13, 827.83 examples/s]11508 examples [00:13, 822.91 examples/s]11591 examples [00:13, 812.59 examples/s]11675 examples [00:13, 818.78 examples/s]11757 examples [00:13, 812.44 examples/s]11839 examples [00:14, 798.54 examples/s]11919 examples [00:14, 794.11 examples/s]12000 examples [00:14, 797.64 examples/s]12087 examples [00:14, 815.57 examples/s]12169 examples [00:14, 808.95 examples/s]12251 examples [00:14, 810.15 examples/s]12333 examples [00:14, 801.25 examples/s]12414 examples [00:14, 802.40 examples/s]12496 examples [00:14, 806.81 examples/s]12581 examples [00:15, 817.00 examples/s]12663 examples [00:15, 817.87 examples/s]12748 examples [00:15, 824.98 examples/s]12831 examples [00:15, 806.70 examples/s]12912 examples [00:15, 797.06 examples/s]12994 examples [00:15, 803.04 examples/s]13075 examples [00:15, 778.98 examples/s]13160 examples [00:15, 798.73 examples/s]13242 examples [00:15, 802.93 examples/s]13323 examples [00:15, 795.88 examples/s]13409 examples [00:16, 812.80 examples/s]13492 examples [00:16, 815.55 examples/s]13574 examples [00:16, 800.24 examples/s]13661 examples [00:16, 818.75 examples/s]13744 examples [00:16, 812.81 examples/s]13827 examples [00:16, 817.15 examples/s]13909 examples [00:16, 806.99 examples/s]13990 examples [00:16, 799.29 examples/s]14075 examples [00:16, 813.30 examples/s]14164 examples [00:16, 833.57 examples/s]14251 examples [00:17, 843.22 examples/s]14336 examples [00:17, 833.74 examples/s]14420 examples [00:17, 832.30 examples/s]14504 examples [00:17, 830.85 examples/s]14588 examples [00:17, 814.89 examples/s]14670 examples [00:17, 795.28 examples/s]14750 examples [00:17, 790.08 examples/s]14835 examples [00:17, 807.01 examples/s]14926 examples [00:17, 833.48 examples/s]15010 examples [00:17, 834.68 examples/s]15094 examples [00:18, 832.27 examples/s]15178 examples [00:18, 806.32 examples/s]15259 examples [00:18, 796.20 examples/s]15341 examples [00:18, 800.76 examples/s]15427 examples [00:18, 815.49 examples/s]15509 examples [00:18, 808.51 examples/s]15592 examples [00:18, 813.57 examples/s]15674 examples [00:18, 814.57 examples/s]15758 examples [00:18, 820.47 examples/s]15844 examples [00:19, 830.89 examples/s]15928 examples [00:19, 816.94 examples/s]16011 examples [00:19, 818.33 examples/s]16093 examples [00:19, 814.47 examples/s]16184 examples [00:19, 839.03 examples/s]16271 examples [00:19, 845.14 examples/s]16356 examples [00:19, 839.84 examples/s]16441 examples [00:19, 828.88 examples/s]16525 examples [00:19, 810.10 examples/s]16611 examples [00:19, 822.84 examples/s]16696 examples [00:20, 829.70 examples/s]16780 examples [00:20, 824.20 examples/s]16863 examples [00:20, 816.33 examples/s]16945 examples [00:20, 807.63 examples/s]17027 examples [00:20, 810.92 examples/s]17115 examples [00:20, 829.51 examples/s]17199 examples [00:20, 815.81 examples/s]17283 examples [00:20, 821.88 examples/s]17366 examples [00:20, 815.25 examples/s]17450 examples [00:20, 819.59 examples/s]17534 examples [00:21, 823.22 examples/s]17617 examples [00:21, 819.27 examples/s]17706 examples [00:21, 837.38 examples/s]17790 examples [00:21, 832.23 examples/s]17876 examples [00:21, 839.94 examples/s]17961 examples [00:21, 840.52 examples/s]18046 examples [00:21, 834.45 examples/s]18130 examples [00:21, 825.06 examples/s]18215 examples [00:21, 829.45 examples/s]18300 examples [00:21, 832.99 examples/s]18387 examples [00:22, 843.48 examples/s]18477 examples [00:22, 857.41 examples/s]18563 examples [00:22, 848.90 examples/s]18648 examples [00:22, 848.85 examples/s]18733 examples [00:22, 845.17 examples/s]18818 examples [00:22, 846.27 examples/s]18905 examples [00:22, 850.66 examples/s]18991 examples [00:22, 834.77 examples/s]19075 examples [00:22, 829.61 examples/s]19159 examples [00:23, 814.39 examples/s]19249 examples [00:23, 837.11 examples/s]19334 examples [00:23, 840.09 examples/s]19421 examples [00:23, 848.50 examples/s]19506 examples [00:23, 848.80 examples/s]19591 examples [00:23, 838.70 examples/s]19675 examples [00:23, 815.10 examples/s]19757 examples [00:23, 804.84 examples/s]19845 examples [00:23, 823.99 examples/s]19928 examples [00:23, 815.58 examples/s]20010 examples [00:24, 775.68 examples/s]20097 examples [00:24, 801.46 examples/s]20178 examples [00:24, 793.86 examples/s]20262 examples [00:24, 804.77 examples/s]20348 examples [00:24, 818.48 examples/s]20435 examples [00:24, 830.70 examples/s]20519 examples [00:24, 810.91 examples/s]20602 examples [00:24, 816.05 examples/s]20688 examples [00:24, 826.21 examples/s]20775 examples [00:24, 834.86 examples/s]20863 examples [00:25, 846.67 examples/s]20951 examples [00:25, 855.28 examples/s]21040 examples [00:25, 863.51 examples/s]21129 examples [00:25, 868.96 examples/s]21218 examples [00:25, 872.37 examples/s]21310 examples [00:25, 882.89 examples/s]21400 examples [00:25, 885.47 examples/s]21489 examples [00:25, 874.59 examples/s]21577 examples [00:25, 858.03 examples/s]21663 examples [00:25, 844.98 examples/s]21748 examples [00:26, 840.81 examples/s]21836 examples [00:26, 851.78 examples/s]21922 examples [00:26, 853.53 examples/s]22008 examples [00:26, 847.06 examples/s]22093 examples [00:26, 831.30 examples/s]22177 examples [00:26, 831.16 examples/s]22261 examples [00:26, 822.68 examples/s]22344 examples [00:26, 819.97 examples/s]22429 examples [00:26, 826.64 examples/s]22521 examples [00:27, 852.57 examples/s]22610 examples [00:27, 861.02 examples/s]22697 examples [00:27, 857.95 examples/s]22785 examples [00:27, 863.20 examples/s]22875 examples [00:27, 871.59 examples/s]22965 examples [00:27, 879.02 examples/s]23053 examples [00:27, 859.00 examples/s]23140 examples [00:27, 821.11 examples/s]23223 examples [00:27, 822.28 examples/s]23310 examples [00:27, 834.75 examples/s]23396 examples [00:28, 839.58 examples/s]23483 examples [00:28, 847.18 examples/s]23568 examples [00:28, 822.36 examples/s]23652 examples [00:28, 826.08 examples/s]23738 examples [00:28, 834.04 examples/s]23826 examples [00:28, 845.02 examples/s]23913 examples [00:28, 850.59 examples/s]24005 examples [00:28, 868.98 examples/s]24093 examples [00:28, 865.84 examples/s]24180 examples [00:28, 863.57 examples/s]24267 examples [00:29, 836.09 examples/s]24352 examples [00:29, 839.31 examples/s]24439 examples [00:29, 846.43 examples/s]24526 examples [00:29, 852.43 examples/s]24612 examples [00:29, 845.26 examples/s]24697 examples [00:29, 844.00 examples/s]24782 examples [00:29, 833.72 examples/s]24869 examples [00:29, 843.60 examples/s]24954 examples [00:29, 837.60 examples/s]25038 examples [00:29, 838.19 examples/s]25123 examples [00:30, 840.21 examples/s]25208 examples [00:30, 810.65 examples/s]25293 examples [00:30, 821.54 examples/s]25378 examples [00:30, 828.05 examples/s]25469 examples [00:30, 850.44 examples/s]25557 examples [00:30, 857.18 examples/s]25646 examples [00:30, 864.92 examples/s]25736 examples [00:30, 874.48 examples/s]25824 examples [00:30, 874.52 examples/s]25912 examples [00:31, 863.13 examples/s]25999 examples [00:31, 857.91 examples/s]26085 examples [00:31, 849.68 examples/s]26171 examples [00:31, 846.17 examples/s]26256 examples [00:31, 815.77 examples/s]26346 examples [00:31, 837.27 examples/s]26431 examples [00:31, 833.53 examples/s]26515 examples [00:31, 819.61 examples/s]26598 examples [00:31, 804.58 examples/s]26686 examples [00:31, 823.75 examples/s]26775 examples [00:32, 841.87 examples/s]26861 examples [00:32, 845.15 examples/s]26952 examples [00:32, 861.50 examples/s]27039 examples [00:32, 848.04 examples/s]27125 examples [00:32, 838.14 examples/s]27212 examples [00:32, 846.17 examples/s]27297 examples [00:32, 845.47 examples/s]27382 examples [00:32, 842.13 examples/s]27467 examples [00:32, 807.65 examples/s]27549 examples [00:32, 795.16 examples/s]27634 examples [00:33, 810.44 examples/s]27717 examples [00:33, 813.61 examples/s]27799 examples [00:33, 804.58 examples/s]27882 examples [00:33, 811.88 examples/s]27965 examples [00:33, 814.82 examples/s]28051 examples [00:33, 827.75 examples/s]28135 examples [00:33, 831.08 examples/s]28219 examples [00:33, 833.61 examples/s]28306 examples [00:33, 841.96 examples/s]28391 examples [00:34, 843.67 examples/s]28479 examples [00:34, 853.92 examples/s]28565 examples [00:34, 854.59 examples/s]28656 examples [00:34, 869.00 examples/s]28743 examples [00:34, 845.75 examples/s]28828 examples [00:34, 823.95 examples/s]28911 examples [00:34, 820.25 examples/s]28997 examples [00:34, 831.39 examples/s]29081 examples [00:34, 831.48 examples/s]29170 examples [00:34, 845.86 examples/s]29255 examples [00:35, 825.79 examples/s]29338 examples [00:35, 820.97 examples/s]29421 examples [00:35, 808.15 examples/s]29512 examples [00:35, 834.91 examples/s]29600 examples [00:35, 846.84 examples/s]29685 examples [00:35, 846.38 examples/s]29770 examples [00:35, 841.23 examples/s]29858 examples [00:35, 849.75 examples/s]29946 examples [00:35, 857.12 examples/s]30032 examples [00:35, 797.10 examples/s]30118 examples [00:36, 813.03 examples/s]30208 examples [00:36, 837.02 examples/s]30298 examples [00:36, 853.93 examples/s]30384 examples [00:36, 842.71 examples/s]30469 examples [00:36, 841.02 examples/s]30557 examples [00:36, 850.17 examples/s]30643 examples [00:36, 820.51 examples/s]30726 examples [00:36, 820.32 examples/s]30816 examples [00:36, 841.08 examples/s]30906 examples [00:37, 856.88 examples/s]30993 examples [00:37, 859.98 examples/s]31080 examples [00:37, 855.16 examples/s]31167 examples [00:37, 858.23 examples/s]31253 examples [00:37, 854.47 examples/s]31339 examples [00:37, 846.50 examples/s]31424 examples [00:37, 844.41 examples/s]31510 examples [00:37, 846.90 examples/s]31596 examples [00:37, 849.41 examples/s]31683 examples [00:37, 853.45 examples/s]31769 examples [00:38, 839.45 examples/s]31854 examples [00:38, 827.74 examples/s]31937 examples [00:38, 828.26 examples/s]32020 examples [00:38, 818.80 examples/s]32102 examples [00:38, 796.01 examples/s]32189 examples [00:38, 815.07 examples/s]32271 examples [00:38, 793.39 examples/s]32351 examples [00:38, 762.86 examples/s]32434 examples [00:38, 780.11 examples/s]32515 examples [00:38, 786.97 examples/s]32603 examples [00:39, 811.83 examples/s]32689 examples [00:39, 823.97 examples/s]32772 examples [00:39, 813.25 examples/s]32858 examples [00:39, 826.09 examples/s]32942 examples [00:39, 828.77 examples/s]33028 examples [00:39, 837.65 examples/s]33113 examples [00:39, 840.55 examples/s]33199 examples [00:39, 846.22 examples/s]33286 examples [00:39, 853.17 examples/s]33372 examples [00:39, 851.98 examples/s]33461 examples [00:40, 860.85 examples/s]33548 examples [00:40, 834.32 examples/s]33633 examples [00:40, 838.88 examples/s]33722 examples [00:40, 853.29 examples/s]33808 examples [00:40, 844.12 examples/s]33893 examples [00:40, 834.83 examples/s]33977 examples [00:40, 825.59 examples/s]34063 examples [00:40, 832.59 examples/s]34147 examples [00:40, 825.41 examples/s]34230 examples [00:41, 811.99 examples/s]34316 examples [00:41, 823.03 examples/s]34403 examples [00:41, 835.62 examples/s]34487 examples [00:41, 817.69 examples/s]34573 examples [00:41, 827.62 examples/s]34659 examples [00:41, 834.57 examples/s]34747 examples [00:41, 845.08 examples/s]34835 examples [00:41, 852.98 examples/s]34921 examples [00:41, 853.01 examples/s]35009 examples [00:41, 859.17 examples/s]35095 examples [00:42, 858.02 examples/s]35181 examples [00:42, 838.61 examples/s]35268 examples [00:42, 845.45 examples/s]35353 examples [00:42, 810.23 examples/s]35441 examples [00:42, 827.80 examples/s]35528 examples [00:42, 837.45 examples/s]35616 examples [00:42, 848.71 examples/s]35704 examples [00:42, 856.12 examples/s]35790 examples [00:42, 838.16 examples/s]35875 examples [00:42, 840.14 examples/s]35960 examples [00:43, 818.43 examples/s]36047 examples [00:43, 832.68 examples/s]36133 examples [00:43, 840.55 examples/s]36220 examples [00:43, 848.78 examples/s]36306 examples [00:43, 850.88 examples/s]36392 examples [00:43, 846.29 examples/s]36477 examples [00:43, 845.87 examples/s]36563 examples [00:43, 848.55 examples/s]36648 examples [00:43, 841.47 examples/s]36740 examples [00:43, 861.39 examples/s]36828 examples [00:44, 865.30 examples/s]36915 examples [00:44, 865.54 examples/s]37002 examples [00:44, 859.06 examples/s]37088 examples [00:44, 852.54 examples/s]37174 examples [00:44, 822.21 examples/s]37257 examples [00:44, 818.57 examples/s]37340 examples [00:44, 806.11 examples/s]37421 examples [00:44, 805.57 examples/s]37509 examples [00:44, 825.77 examples/s]37594 examples [00:45, 831.34 examples/s]37678 examples [00:45, 830.44 examples/s]37762 examples [00:45, 811.99 examples/s]37848 examples [00:45, 821.44 examples/s]37933 examples [00:45, 829.12 examples/s]38017 examples [00:45, 812.50 examples/s]38099 examples [00:45, 788.28 examples/s]38186 examples [00:45, 809.94 examples/s]38274 examples [00:45, 828.21 examples/s]38358 examples [00:45, 817.99 examples/s]38441 examples [00:46, 819.38 examples/s]38525 examples [00:46, 823.64 examples/s]38614 examples [00:46, 840.99 examples/s]38703 examples [00:46, 854.60 examples/s]38791 examples [00:46, 859.76 examples/s]38878 examples [00:46, 859.97 examples/s]38965 examples [00:46, 853.39 examples/s]39054 examples [00:46, 862.70 examples/s]39141 examples [00:46, 857.51 examples/s]39227 examples [00:46, 855.97 examples/s]39313 examples [00:47, 833.70 examples/s]39397 examples [00:47, 824.79 examples/s]39480 examples [00:47, 826.26 examples/s]39569 examples [00:47, 842.92 examples/s]39659 examples [00:47, 856.98 examples/s]39746 examples [00:47, 858.82 examples/s]39832 examples [00:47, 829.36 examples/s]39922 examples [00:47, 848.02 examples/s]40008 examples [00:47, 799.22 examples/s]40089 examples [00:48, 801.75 examples/s]40175 examples [00:48, 816.82 examples/s]40263 examples [00:48, 833.22 examples/s]40348 examples [00:48, 835.61 examples/s]40433 examples [00:48, 838.05 examples/s]40518 examples [00:48, 841.15 examples/s]40603 examples [00:48, 839.27 examples/s]40688 examples [00:48, 835.64 examples/s]40774 examples [00:48, 842.51 examples/s]40861 examples [00:48, 849.49 examples/s]40947 examples [00:49, 844.94 examples/s]41032 examples [00:49, 832.90 examples/s]41116 examples [00:49, 825.01 examples/s]41199 examples [00:49, 811.94 examples/s]41284 examples [00:49, 821.32 examples/s]41371 examples [00:49, 834.23 examples/s]41455 examples [00:49, 821.43 examples/s]41540 examples [00:49, 827.76 examples/s]41623 examples [00:49, 825.76 examples/s]41708 examples [00:49, 831.40 examples/s]41796 examples [00:50, 843.21 examples/s]41881 examples [00:50, 834.21 examples/s]41966 examples [00:50, 835.99 examples/s]42056 examples [00:50, 853.26 examples/s]42147 examples [00:50, 868.64 examples/s]42235 examples [00:50, 853.97 examples/s]42321 examples [00:50, 853.10 examples/s]42410 examples [00:50, 863.67 examples/s]42501 examples [00:50, 873.49 examples/s]42589 examples [00:50, 844.66 examples/s]42674 examples [00:51, 838.73 examples/s]42762 examples [00:51, 848.32 examples/s]42848 examples [00:51, 845.55 examples/s]42935 examples [00:51, 848.89 examples/s]43023 examples [00:51, 855.38 examples/s]43111 examples [00:51, 861.21 examples/s]43198 examples [00:51, 852.16 examples/s]43284 examples [00:51, 842.36 examples/s]43375 examples [00:51, 860.30 examples/s]43462 examples [00:52, 857.18 examples/s]43549 examples [00:52, 860.53 examples/s]43636 examples [00:52, 854.01 examples/s]43725 examples [00:52, 862.72 examples/s]43814 examples [00:52, 868.72 examples/s]43902 examples [00:52, 870.96 examples/s]43990 examples [00:52, 865.40 examples/s]44081 examples [00:52, 875.40 examples/s]44169 examples [00:52, 869.64 examples/s]44257 examples [00:52, 840.62 examples/s]44343 examples [00:53, 844.17 examples/s]44429 examples [00:53, 846.62 examples/s]44514 examples [00:53, 846.96 examples/s]44604 examples [00:53, 861.46 examples/s]44695 examples [00:53, 872.83 examples/s]44783 examples [00:53, 861.77 examples/s]44870 examples [00:53, 859.56 examples/s]44957 examples [00:53, 855.86 examples/s]45046 examples [00:53, 865.75 examples/s]45137 examples [00:53, 875.50 examples/s]45225 examples [00:54, 867.91 examples/s]45315 examples [00:54, 875.35 examples/s]45403 examples [00:54, 865.58 examples/s]45492 examples [00:54, 870.66 examples/s]45580 examples [00:54, 829.25 examples/s]45667 examples [00:54, 840.58 examples/s]45752 examples [00:54, 811.74 examples/s]45835 examples [00:54, 815.69 examples/s]45917 examples [00:54, 803.83 examples/s]46006 examples [00:54, 827.03 examples/s]46092 examples [00:55, 834.93 examples/s]46176 examples [00:55, 832.20 examples/s]46261 examples [00:55, 835.89 examples/s]46351 examples [00:55, 854.14 examples/s]46437 examples [00:55, 823.97 examples/s]46520 examples [00:55, 823.63 examples/s]46603 examples [00:55, 816.65 examples/s]46685 examples [00:55, 814.69 examples/s]46768 examples [00:55, 816.50 examples/s]46850 examples [00:56, 816.73 examples/s]46937 examples [00:56, 829.64 examples/s]47024 examples [00:56, 839.75 examples/s]47109 examples [00:56, 840.36 examples/s]47194 examples [00:56, 837.84 examples/s]47278 examples [00:56, 835.40 examples/s]47362 examples [00:56, 826.51 examples/s]47445 examples [00:56, 774.53 examples/s]47526 examples [00:56, 783.19 examples/s]47606 examples [00:56, 786.02 examples/s]47691 examples [00:57, 802.56 examples/s]47780 examples [00:57, 826.91 examples/s]47864 examples [00:57, 776.23 examples/s]47943 examples [00:57, 758.40 examples/s]48028 examples [00:57, 783.72 examples/s]48112 examples [00:57, 798.55 examples/s]48193 examples [00:57, 793.13 examples/s]48275 examples [00:57, 800.18 examples/s]48363 examples [00:57, 822.22 examples/s]48451 examples [00:57, 836.11 examples/s]48535 examples [00:58, 829.70 examples/s]48619 examples [00:58, 819.70 examples/s]48702 examples [00:58, 808.15 examples/s]48787 examples [00:58, 819.93 examples/s]48874 examples [00:58, 832.18 examples/s]48958 examples [00:58, 821.90 examples/s]49044 examples [00:58, 830.92 examples/s]49128 examples [00:58, 831.22 examples/s]49212 examples [00:58, 816.98 examples/s]49299 examples [00:59, 831.53 examples/s]49387 examples [00:59, 843.83 examples/s]49478 examples [00:59, 859.89 examples/s]49565 examples [00:59, 858.42 examples/s]49655 examples [00:59, 869.97 examples/s]49744 examples [00:59, 874.99 examples/s]49832 examples [00:59, 867.77 examples/s]49919 examples [00:59, 863.16 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6027/50000 [00:00<00:00, 60258.42 examples/s] 34%|███▍      | 17235/50000 [00:00<00:00, 69962.55 examples/s] 57%|█████▋    | 28625/50000 [00:00<00:00, 79117.62 examples/s] 80%|███████▉  | 39969/50000 [00:00<00:00, 87015.65 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 699.57 examples/s]157 examples [00:00, 741.50 examples/s]245 examples [00:00, 776.90 examples/s]334 examples [00:00, 805.12 examples/s]426 examples [00:00, 834.34 examples/s]518 examples [00:00, 857.51 examples/s]611 examples [00:00, 876.61 examples/s]705 examples [00:00, 893.89 examples/s]797 examples [00:00, 901.14 examples/s]886 examples [00:01, 895.70 examples/s]975 examples [00:01, 891.27 examples/s]1063 examples [00:01, 887.13 examples/s]1154 examples [00:01, 890.85 examples/s]1243 examples [00:01, 887.49 examples/s]1333 examples [00:01, 890.39 examples/s]1422 examples [00:01, 875.53 examples/s]1510 examples [00:01, 862.79 examples/s]1597 examples [00:01, 857.20 examples/s]1687 examples [00:01, 867.32 examples/s]1774 examples [00:02, 840.10 examples/s]1859 examples [00:02, 836.05 examples/s]1943 examples [00:02, 829.83 examples/s]2028 examples [00:02, 832.99 examples/s]2112 examples [00:02, 829.22 examples/s]2195 examples [00:02, 823.94 examples/s]2278 examples [00:02, 817.86 examples/s]2365 examples [00:02, 831.85 examples/s]2449 examples [00:02, 831.05 examples/s]2534 examples [00:02, 836.09 examples/s]2623 examples [00:03, 849.97 examples/s]2709 examples [00:03, 846.57 examples/s]2798 examples [00:03, 859.11 examples/s]2887 examples [00:03, 868.02 examples/s]2974 examples [00:03, 842.67 examples/s]3060 examples [00:03, 847.62 examples/s]3145 examples [00:03, 844.61 examples/s]3238 examples [00:03, 867.03 examples/s]3329 examples [00:03, 879.44 examples/s]3418 examples [00:03, 880.39 examples/s]3511 examples [00:04, 892.96 examples/s]3605 examples [00:04, 904.14 examples/s]3696 examples [00:04, 899.68 examples/s]3787 examples [00:04, 897.87 examples/s]3877 examples [00:04, 881.93 examples/s]3972 examples [00:04, 898.62 examples/s]4067 examples [00:04, 912.60 examples/s]4159 examples [00:04, 889.90 examples/s]4249 examples [00:04, 864.72 examples/s]4340 examples [00:05, 876.60 examples/s]4428 examples [00:05, 873.72 examples/s]4516 examples [00:05, 855.57 examples/s]4602 examples [00:05, 853.01 examples/s]4688 examples [00:05, 844.77 examples/s]4774 examples [00:05, 849.09 examples/s]4865 examples [00:05, 865.28 examples/s]4956 examples [00:05, 877.39 examples/s]5046 examples [00:05, 883.14 examples/s]5137 examples [00:05, 889.48 examples/s]5227 examples [00:06, 884.30 examples/s]5316 examples [00:06, 882.03 examples/s]5405 examples [00:06, 875.57 examples/s]5493 examples [00:06, 857.46 examples/s]5583 examples [00:06, 869.46 examples/s]5671 examples [00:06, 859.99 examples/s]5758 examples [00:06, 831.94 examples/s]5847 examples [00:06, 846.19 examples/s]5937 examples [00:06, 859.66 examples/s]6024 examples [00:06, 839.57 examples/s]6109 examples [00:07, 839.62 examples/s]6194 examples [00:07, 819.48 examples/s]6277 examples [00:07, 800.07 examples/s]6360 examples [00:07, 807.64 examples/s]6448 examples [00:07, 827.87 examples/s]6532 examples [00:07, 809.45 examples/s]6619 examples [00:07, 825.42 examples/s]6703 examples [00:07, 828.19 examples/s]6787 examples [00:07, 796.72 examples/s]6875 examples [00:08, 819.74 examples/s]6960 examples [00:08, 827.22 examples/s]7044 examples [00:08, 790.54 examples/s]7131 examples [00:08, 812.60 examples/s]7216 examples [00:08, 823.20 examples/s]7299 examples [00:08, 811.98 examples/s]7387 examples [00:08, 829.85 examples/s]7472 examples [00:08, 834.59 examples/s]7565 examples [00:08, 858.26 examples/s]7652 examples [00:08, 853.68 examples/s]7741 examples [00:09, 863.78 examples/s]7829 examples [00:09, 866.72 examples/s]7922 examples [00:09, 883.68 examples/s]8011 examples [00:09, 877.89 examples/s]8106 examples [00:09, 897.66 examples/s]8197 examples [00:09, 900.09 examples/s]8288 examples [00:09, 889.27 examples/s]8378 examples [00:09, 871.83 examples/s]8466 examples [00:09, 857.43 examples/s]8558 examples [00:09, 875.13 examples/s]8654 examples [00:10, 897.08 examples/s]8744 examples [00:10, 897.08 examples/s]8836 examples [00:10, 902.21 examples/s]8927 examples [00:10, 901.37 examples/s]9019 examples [00:10, 906.07 examples/s]9113 examples [00:10, 914.81 examples/s]9205 examples [00:10, 895.02 examples/s]9295 examples [00:10, 896.43 examples/s]9385 examples [00:10, 883.42 examples/s]9476 examples [00:10, 888.49 examples/s]9565 examples [00:11, 873.46 examples/s]9653 examples [00:11, 846.57 examples/s]9747 examples [00:11, 869.95 examples/s]9835 examples [00:11, 871.95 examples/s]9924 examples [00:11, 876.27 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteX0Q5YT/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteX0Q5YT/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe1519a6730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe1519a6730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe1519a6730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe0dad23da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe0dadd1278>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe1519a6730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe1519a6730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe1519a6730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe12f585048>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe12f585080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe0c941d400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe0c941d400> 

  function with postional parmater data_info <function split_train_valid at 0x7fe0c941d400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe0c941d510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe0c941d510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe0c941d510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=b05864becd770d869b493a22a2869e44f03f1f17860aced3ef79977c15617338
  Stored in directory: /tmp/pip-ephem-wheel-cache-m9jrs71b/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip: 0.00B [01:11, ?B/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json [Errno 104] Connection reset by peer

  




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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 298, in compute
    out_tmp = preprocessor_func(data_info=self.data_info, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 241, in create_tabular_dataset
    TEXT.build_vocab(tabular_train, vectors=pretrained_emb)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/data/field.py", line 309, in build_vocab
    self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 101, in __init__
    self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 182, in load_vectors
    vectors[idx] = pretrained_aliases[vector](**kwargs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 484, in __init__
    super(GloVe, self).__init__(name, url=url, **kwargs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 323, in __init__
    self.cache(name, cache, url=url, max_vectors=max_vectors)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 358, in cache
    urlretrieve(url, dest, reporthook=reporthook(t))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 248, in urlretrieve
    with contextlib.closing(urlopen(url, data)) as fp:
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 223, in urlopen
    return opener.open(url, data, timeout)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 532, in open
    response = meth(req, response)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 642, in http_response
    'http', request, response, code, msg, hdrs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 564, in error
    result = self._call_chain(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 504, in _call_chain
    result = func(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 756, in http_error_302
    return self.parent.open(new, timeout=req.timeout)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 532, in open
    response = meth(req, response)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 642, in http_response
    'http', request, response, code, msg, hdrs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 564, in error
    result = self._call_chain(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 504, in _call_chain
    result = func(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 756, in http_error_302
    return self.parent.open(new, timeout=req.timeout)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 526, in open
    response = self._open(req, data)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 544, in _open
    '_open', req)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 504, in _call_chain
    result = func(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 1377, in http_open
    return self.do_open(http.client.HTTPConnection, req)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 1352, in do_open
    r = h.getresponse()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/http/client.py", line 1364, in getresponse
    response.begin()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/http/client.py", line 307, in begin
    version, status, reason = self._read_status()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/http/client.py", line 268, in _read_status
    line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/socket.py", line 586, in readinto
    return self._sock.recv_into(b)
ConnectionResetError: [Errno 104] Connection reset by peer
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.87 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.87 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  5.87 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.07 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.95 file/s]2020-07-16 12:13:21.792930: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-16 12:13:21.798579: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-16 12:13:21.798918: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56473a442120 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-16 12:13:21.798956: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:00, 162344.17it/s] 65%|██████▌   | 6463488/9912422 [00:00<00:14, 231668.98it/s]9920512it [00:00, 42023490.50it/s]                           
0it [00:00, ?it/s]32768it [00:00, 440017.27it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 145603.22it/s]1654784it [00:00, 10764896.70it/s]                         
0it [00:00, ?it/s]8192it [00:00, 122157.97it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
