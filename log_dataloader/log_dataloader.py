
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe7c6df7a60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe7c6df7a60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe831a62510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe831a62510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe850cb1ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe850cb1ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe7ded96950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe7ded96950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe7ded96950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 44%|████▍     | 4407296/9912422 [00:00<00:00, 43606300.36it/s]9920512it [00:00, 34449403.54it/s]                             
0it [00:00, ?it/s]32768it [00:00, 749280.12it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 481550.31it/s]1654784it [00:00, 11738455.73it/s]                         
0it [00:00, ?it/s]8192it [00:00, 127388.98it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe7dce70198>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe7dd305898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe7ded96598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe7ded96598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe7ded96598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:16,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:15,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:14,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:14,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:50,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:50,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:49,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:35,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:34,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:34,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:15,  7.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 10.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 10.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 10.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 25.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 25.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 31.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 31.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 38.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 38.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 38.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 46.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 46.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 46.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 46.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 53.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 53.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 53.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 53.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 59.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 59.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 59.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 59.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 59.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 59.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 59.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 59.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 59.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 67.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 67.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 67.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 67.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 67.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 67.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 67.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 68.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 68.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 68.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 68.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 71.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.61s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.61s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.61s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.34 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.61s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 71.34 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.72 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ url]
0 examples [00:00, ? examples/s]2020-07-20 18:09:15.739519: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-20 18:09:15.788140: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-20 18:09:15.788639: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5643ff938870 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-20 18:09:15.788667: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  8.06 examples/s]94 examples [00:00, 11.47 examples/s]186 examples [00:00, 16.30 examples/s]277 examples [00:00, 23.11 examples/s]370 examples [00:00, 32.66 examples/s]459 examples [00:00, 45.94 examples/s]551 examples [00:00, 64.24 examples/s]643 examples [00:00, 89.09 examples/s]734 examples [00:00, 122.14 examples/s]826 examples [00:01, 165.09 examples/s]919 examples [00:01, 219.13 examples/s]1014 examples [00:01, 284.79 examples/s]1106 examples [00:01, 359.02 examples/s]1198 examples [00:01, 437.32 examples/s]1291 examples [00:01, 519.45 examples/s]1386 examples [00:01, 600.39 examples/s]1479 examples [00:01, 639.61 examples/s]1567 examples [00:01, 677.55 examples/s]1652 examples [00:01, 717.35 examples/s]1737 examples [00:02, 733.04 examples/s]1820 examples [00:02, 754.11 examples/s]1908 examples [00:02, 786.51 examples/s]1997 examples [00:02, 812.96 examples/s]2084 examples [00:02, 829.00 examples/s]2171 examples [00:02, 839.04 examples/s]2257 examples [00:02, 839.06 examples/s]2346 examples [00:02, 852.91 examples/s]2433 examples [00:02, 849.24 examples/s]2520 examples [00:02, 853.81 examples/s]2610 examples [00:03, 866.21 examples/s]2697 examples [00:03, 848.76 examples/s]2783 examples [00:03, 831.60 examples/s]2875 examples [00:03, 854.16 examples/s]2967 examples [00:03, 869.83 examples/s]3061 examples [00:03, 887.92 examples/s]3153 examples [00:03, 896.92 examples/s]3243 examples [00:03, 894.22 examples/s]3334 examples [00:03, 896.07 examples/s]3424 examples [00:04, 891.16 examples/s]3515 examples [00:04, 894.84 examples/s]3605 examples [00:04, 892.41 examples/s]3695 examples [00:04, 849.76 examples/s]3783 examples [00:04, 854.85 examples/s]3872 examples [00:04, 865.03 examples/s]3963 examples [00:04, 876.00 examples/s]4053 examples [00:04, 880.51 examples/s]4145 examples [00:04, 889.77 examples/s]4235 examples [00:04, 883.02 examples/s]4324 examples [00:05, 878.77 examples/s]4412 examples [00:05, 859.37 examples/s]4510 examples [00:05, 889.90 examples/s]4602 examples [00:05, 896.46 examples/s]4695 examples [00:05, 904.29 examples/s]4789 examples [00:05, 912.20 examples/s]4888 examples [00:05, 933.31 examples/s]4987 examples [00:05, 948.04 examples/s]5083 examples [00:05, 929.83 examples/s]5177 examples [00:05, 924.97 examples/s]5270 examples [00:06, 918.69 examples/s]5363 examples [00:06, 919.82 examples/s]5456 examples [00:06, 921.53 examples/s]5549 examples [00:06, 921.94 examples/s]5642 examples [00:06, 903.87 examples/s]5733 examples [00:06, 898.77 examples/s]5823 examples [00:06, 887.23 examples/s]5912 examples [00:06, 863.27 examples/s]6001 examples [00:06, 868.69 examples/s]6089 examples [00:07, 855.52 examples/s]6175 examples [00:07, 824.30 examples/s]6266 examples [00:07, 847.97 examples/s]6363 examples [00:07, 880.05 examples/s]6452 examples [00:07, 881.36 examples/s]6541 examples [00:07, 881.84 examples/s]6637 examples [00:07, 902.85 examples/s]6728 examples [00:07, 897.13 examples/s]6820 examples [00:07, 902.54 examples/s]6912 examples [00:07, 906.18 examples/s]7006 examples [00:08, 914.86 examples/s]7098 examples [00:08, 912.15 examples/s]7192 examples [00:08, 917.79 examples/s]7286 examples [00:08, 922.56 examples/s]7379 examples [00:08, 885.68 examples/s]7472 examples [00:08, 897.75 examples/s]7567 examples [00:08, 910.39 examples/s]7661 examples [00:08, 919.06 examples/s]7754 examples [00:08, 920.42 examples/s]7848 examples [00:08, 925.89 examples/s]7942 examples [00:09, 927.75 examples/s]8038 examples [00:09, 934.71 examples/s]8132 examples [00:09, 925.19 examples/s]8225 examples [00:09, 906.88 examples/s]8317 examples [00:09, 910.44 examples/s]8409 examples [00:09, 906.40 examples/s]8504 examples [00:09, 916.43 examples/s]8599 examples [00:09, 924.84 examples/s]8694 examples [00:09, 929.80 examples/s]8788 examples [00:09, 922.84 examples/s]8881 examples [00:10, 914.69 examples/s]8973 examples [00:10, 895.82 examples/s]9065 examples [00:10, 901.97 examples/s]9156 examples [00:10, 903.63 examples/s]9250 examples [00:10, 911.16 examples/s]9342 examples [00:10, 896.52 examples/s]9433 examples [00:10, 898.65 examples/s]9523 examples [00:10, 887.58 examples/s]9612 examples [00:10, 876.33 examples/s]9700 examples [00:11, 841.46 examples/s]9794 examples [00:11, 868.29 examples/s]9886 examples [00:11, 881.22 examples/s]9978 examples [00:11, 890.93 examples/s]10068 examples [00:11, 838.87 examples/s]10158 examples [00:11, 855.03 examples/s]10252 examples [00:11, 878.86 examples/s]10343 examples [00:11, 887.91 examples/s]10433 examples [00:11, 865.82 examples/s]10521 examples [00:11, 868.17 examples/s]10609 examples [00:12, 869.72 examples/s]10699 examples [00:12, 875.82 examples/s]10789 examples [00:12, 880.45 examples/s]10878 examples [00:12, 876.92 examples/s]10966 examples [00:12, 870.01 examples/s]11056 examples [00:12, 876.50 examples/s]11147 examples [00:12, 885.09 examples/s]11236 examples [00:12, 881.14 examples/s]11328 examples [00:12, 892.02 examples/s]11420 examples [00:12, 897.64 examples/s]11510 examples [00:13, 897.25 examples/s]11601 examples [00:13, 899.88 examples/s]11692 examples [00:13, 902.54 examples/s]11783 examples [00:13, 902.43 examples/s]11875 examples [00:13, 906.41 examples/s]11966 examples [00:13, 874.40 examples/s]12059 examples [00:13, 889.34 examples/s]12149 examples [00:13, 886.46 examples/s]12242 examples [00:13, 896.97 examples/s]12332 examples [00:13, 890.52 examples/s]12422 examples [00:14, 886.76 examples/s]12511 examples [00:14, 883.21 examples/s]12600 examples [00:14, 874.45 examples/s]12688 examples [00:14, 844.72 examples/s]12776 examples [00:14, 852.23 examples/s]12864 examples [00:14, 858.21 examples/s]12954 examples [00:14, 867.79 examples/s]13041 examples [00:14, 865.75 examples/s]13130 examples [00:14, 870.10 examples/s]13220 examples [00:15, 877.38 examples/s]13308 examples [00:15, 873.24 examples/s]13396 examples [00:15, 874.56 examples/s]13485 examples [00:15, 876.64 examples/s]13573 examples [00:15, 870.51 examples/s]13661 examples [00:15, 866.34 examples/s]13748 examples [00:15, 857.47 examples/s]13834 examples [00:15, 855.63 examples/s]13922 examples [00:15, 860.69 examples/s]14009 examples [00:15, 862.54 examples/s]14096 examples [00:16, 857.12 examples/s]14182 examples [00:16, 852.50 examples/s]14268 examples [00:16, 822.90 examples/s]14356 examples [00:16, 838.26 examples/s]14444 examples [00:16, 849.54 examples/s]14533 examples [00:16, 860.80 examples/s]14620 examples [00:16, 848.52 examples/s]14710 examples [00:16, 862.11 examples/s]14799 examples [00:16, 867.83 examples/s]14886 examples [00:16, 868.03 examples/s]14975 examples [00:17, 872.70 examples/s]15066 examples [00:17, 880.96 examples/s]15156 examples [00:17, 884.93 examples/s]15248 examples [00:17, 894.44 examples/s]15339 examples [00:17, 897.06 examples/s]15430 examples [00:17, 898.88 examples/s]15520 examples [00:17, 895.51 examples/s]15610 examples [00:17, 886.79 examples/s]15699 examples [00:17, 882.56 examples/s]15788 examples [00:17, 879.61 examples/s]15878 examples [00:18, 884.49 examples/s]15968 examples [00:18, 886.67 examples/s]16058 examples [00:18, 889.68 examples/s]16149 examples [00:18, 892.89 examples/s]16240 examples [00:18, 894.95 examples/s]16330 examples [00:18, 887.31 examples/s]16419 examples [00:18, 887.68 examples/s]16511 examples [00:18, 896.79 examples/s]16601 examples [00:18, 879.27 examples/s]16692 examples [00:18, 887.65 examples/s]16781 examples [00:19, 886.39 examples/s]16871 examples [00:19, 889.05 examples/s]16961 examples [00:19, 892.27 examples/s]17053 examples [00:19, 898.10 examples/s]17146 examples [00:19, 905.12 examples/s]17242 examples [00:19, 917.81 examples/s]17335 examples [00:19, 921.33 examples/s]17428 examples [00:19, 923.37 examples/s]17521 examples [00:19, 905.09 examples/s]17612 examples [00:19, 886.41 examples/s]17703 examples [00:20, 891.20 examples/s]17793 examples [00:20, 887.29 examples/s]17885 examples [00:20, 895.32 examples/s]17975 examples [00:20, 895.17 examples/s]18068 examples [00:20, 902.99 examples/s]18164 examples [00:20, 918.45 examples/s]18256 examples [00:20, 916.18 examples/s]18352 examples [00:20, 926.12 examples/s]18446 examples [00:20, 928.49 examples/s]18539 examples [00:21, 923.88 examples/s]18632 examples [00:21, 923.05 examples/s]18725 examples [00:21, 918.33 examples/s]18817 examples [00:21, 912.30 examples/s]18909 examples [00:21, 867.09 examples/s]19001 examples [00:21, 882.03 examples/s]19092 examples [00:21, 888.86 examples/s]19186 examples [00:21, 900.80 examples/s]19278 examples [00:21, 888.93 examples/s]19368 examples [00:21, 879.11 examples/s]19457 examples [00:22, 876.68 examples/s]19549 examples [00:22, 886.64 examples/s]19638 examples [00:22, 881.39 examples/s]19731 examples [00:22, 893.43 examples/s]19821 examples [00:22, 893.59 examples/s]19911 examples [00:22, 893.57 examples/s]20001 examples [00:22, 841.60 examples/s]20089 examples [00:22, 850.79 examples/s]20185 examples [00:22, 878.30 examples/s]20274 examples [00:22, 867.41 examples/s]20370 examples [00:23, 891.19 examples/s]20467 examples [00:23, 912.95 examples/s]20559 examples [00:23, 906.67 examples/s]20653 examples [00:23, 914.80 examples/s]20745 examples [00:23, 914.29 examples/s]20837 examples [00:23, 913.49 examples/s]20929 examples [00:23, 900.07 examples/s]21020 examples [00:23, 899.34 examples/s]21111 examples [00:23, 900.07 examples/s]21202 examples [00:24, 883.68 examples/s]21291 examples [00:24, 883.76 examples/s]21380 examples [00:24, 884.12 examples/s]21469 examples [00:24, 878.98 examples/s]21557 examples [00:24, 849.02 examples/s]21643 examples [00:24, 845.61 examples/s]21731 examples [00:24, 852.02 examples/s]21820 examples [00:24, 862.53 examples/s]21907 examples [00:24, 852.48 examples/s]21995 examples [00:24, 859.55 examples/s]22082 examples [00:25, 851.02 examples/s]22170 examples [00:25, 858.14 examples/s]22256 examples [00:25, 856.83 examples/s]22348 examples [00:25, 874.65 examples/s]22439 examples [00:25, 882.69 examples/s]22531 examples [00:25, 892.81 examples/s]22621 examples [00:25, 892.68 examples/s]22711 examples [00:25, 878.90 examples/s]22799 examples [00:25, 857.47 examples/s]22885 examples [00:25, 849.39 examples/s]22971 examples [00:26, 781.84 examples/s]23058 examples [00:26, 804.90 examples/s]23143 examples [00:26, 817.28 examples/s]23232 examples [00:26, 836.75 examples/s]23326 examples [00:26, 864.54 examples/s]23415 examples [00:26, 870.62 examples/s]23503 examples [00:26, 853.91 examples/s]23590 examples [00:26, 856.96 examples/s]23676 examples [00:26, 848.11 examples/s]23769 examples [00:27, 868.44 examples/s]23862 examples [00:27, 885.50 examples/s]23951 examples [00:27, 875.90 examples/s]24045 examples [00:27, 891.72 examples/s]24135 examples [00:27, 887.45 examples/s]24228 examples [00:27, 898.76 examples/s]24322 examples [00:27, 908.09 examples/s]24413 examples [00:27, 895.41 examples/s]24503 examples [00:27, 849.12 examples/s]24594 examples [00:27, 866.07 examples/s]24687 examples [00:28, 882.13 examples/s]24783 examples [00:28, 902.64 examples/s]24876 examples [00:28, 909.21 examples/s]24971 examples [00:28, 919.60 examples/s]25064 examples [00:28, 917.01 examples/s]25161 examples [00:28, 931.67 examples/s]25255 examples [00:28, 923.41 examples/s]25348 examples [00:28, 921.92 examples/s]25441 examples [00:28, 911.74 examples/s]25533 examples [00:28, 903.61 examples/s]25626 examples [00:29, 910.29 examples/s]25719 examples [00:29, 914.65 examples/s]25811 examples [00:29, 887.61 examples/s]25902 examples [00:29, 891.72 examples/s]25992 examples [00:29, 880.98 examples/s]26082 examples [00:29, 885.97 examples/s]26177 examples [00:29, 901.94 examples/s]26269 examples [00:29, 906.47 examples/s]26361 examples [00:29, 908.01 examples/s]26456 examples [00:29, 918.77 examples/s]26550 examples [00:30, 924.57 examples/s]26643 examples [00:30, 923.67 examples/s]26738 examples [00:30, 930.49 examples/s]26832 examples [00:30, 927.84 examples/s]26926 examples [00:30, 928.94 examples/s]27019 examples [00:30, 912.39 examples/s]27115 examples [00:30, 925.27 examples/s]27213 examples [00:30, 938.87 examples/s]27308 examples [00:30, 908.41 examples/s]27403 examples [00:30, 920.43 examples/s]27496 examples [00:31, 917.21 examples/s]27590 examples [00:31, 923.08 examples/s]27683 examples [00:31, 918.14 examples/s]27775 examples [00:31, 910.37 examples/s]27873 examples [00:31, 930.12 examples/s]27972 examples [00:31, 946.23 examples/s]28067 examples [00:31, 935.16 examples/s]28161 examples [00:31, 925.28 examples/s]28254 examples [00:31, 899.59 examples/s]28345 examples [00:32, 898.54 examples/s]28436 examples [00:32, 892.34 examples/s]28526 examples [00:32, 889.31 examples/s]28616 examples [00:32, 884.83 examples/s]28705 examples [00:32, 883.61 examples/s]28794 examples [00:32, 881.90 examples/s]28883 examples [00:32, 864.65 examples/s]28972 examples [00:32, 869.76 examples/s]29060 examples [00:32, 856.73 examples/s]29149 examples [00:32, 864.09 examples/s]29242 examples [00:33, 881.46 examples/s]29333 examples [00:33, 886.52 examples/s]29422 examples [00:33, 864.97 examples/s]29511 examples [00:33, 871.13 examples/s]29603 examples [00:33, 884.35 examples/s]29692 examples [00:33, 877.53 examples/s]29781 examples [00:33, 879.14 examples/s]29869 examples [00:33, 877.86 examples/s]29961 examples [00:33, 888.15 examples/s]30050 examples [00:33, 844.32 examples/s]30144 examples [00:34, 870.86 examples/s]30241 examples [00:34, 896.09 examples/s]30332 examples [00:34, 874.62 examples/s]30420 examples [00:34, 827.60 examples/s]30511 examples [00:34, 849.07 examples/s]30602 examples [00:34, 865.07 examples/s]30690 examples [00:34, 849.50 examples/s]30779 examples [00:34, 858.58 examples/s]30866 examples [00:34, 860.71 examples/s]30955 examples [00:35, 868.27 examples/s]31047 examples [00:35, 882.55 examples/s]31143 examples [00:35, 902.75 examples/s]31238 examples [00:35, 914.73 examples/s]31330 examples [00:35, 906.26 examples/s]31421 examples [00:35, 903.74 examples/s]31512 examples [00:35, 884.50 examples/s]31601 examples [00:35, 883.86 examples/s]31691 examples [00:35, 886.77 examples/s]31780 examples [00:35, 861.18 examples/s]31872 examples [00:36, 877.90 examples/s]31963 examples [00:36, 886.80 examples/s]32054 examples [00:36, 892.32 examples/s]32145 examples [00:36, 896.20 examples/s]32235 examples [00:36, 875.39 examples/s]32323 examples [00:36, 865.74 examples/s]32410 examples [00:36, 858.59 examples/s]32497 examples [00:36, 859.57 examples/s]32586 examples [00:36, 868.35 examples/s]32673 examples [00:36, 838.02 examples/s]32764 examples [00:37, 855.96 examples/s]32856 examples [00:37, 872.14 examples/s]32945 examples [00:37, 876.48 examples/s]33038 examples [00:37, 889.30 examples/s]33131 examples [00:37, 900.96 examples/s]33226 examples [00:37, 914.13 examples/s]33319 examples [00:37, 917.37 examples/s]33411 examples [00:37, 916.90 examples/s]33504 examples [00:37, 918.04 examples/s]33596 examples [00:37, 910.61 examples/s]33688 examples [00:38, 912.97 examples/s]33780 examples [00:38, 914.14 examples/s]33872 examples [00:38, 910.58 examples/s]33967 examples [00:38, 919.39 examples/s]34060 examples [00:38, 922.54 examples/s]34155 examples [00:38, 929.91 examples/s]34249 examples [00:38, 931.21 examples/s]34343 examples [00:38, 931.02 examples/s]34437 examples [00:38, 933.34 examples/s]34531 examples [00:39, 916.70 examples/s]34624 examples [00:39, 919.32 examples/s]34716 examples [00:39, 913.66 examples/s]34808 examples [00:39, 910.74 examples/s]34900 examples [00:39, 862.47 examples/s]34987 examples [00:39, 815.22 examples/s]35083 examples [00:39, 852.01 examples/s]35174 examples [00:39, 866.93 examples/s]35267 examples [00:39, 882.63 examples/s]35357 examples [00:39, 887.14 examples/s]35447 examples [00:40, 887.76 examples/s]35537 examples [00:40, 891.36 examples/s]35627 examples [00:40, 891.57 examples/s]35719 examples [00:40, 897.34 examples/s]35812 examples [00:40, 904.85 examples/s]35903 examples [00:40, 889.06 examples/s]35994 examples [00:40, 894.82 examples/s]36084 examples [00:40, 891.34 examples/s]36178 examples [00:40, 902.70 examples/s]36273 examples [00:40, 914.11 examples/s]36365 examples [00:41, 892.56 examples/s]36455 examples [00:41, 842.97 examples/s]36546 examples [00:41, 859.47 examples/s]36636 examples [00:41, 870.43 examples/s]36725 examples [00:41, 875.80 examples/s]36820 examples [00:41, 894.22 examples/s]36910 examples [00:41, 890.92 examples/s]37000 examples [00:41, 886.32 examples/s]37089 examples [00:41, 874.64 examples/s]37177 examples [00:42, 871.69 examples/s]37267 examples [00:42, 878.91 examples/s]37356 examples [00:42, 881.47 examples/s]37448 examples [00:42, 891.34 examples/s]37538 examples [00:42, 876.03 examples/s]37626 examples [00:42, 876.70 examples/s]37716 examples [00:42, 883.56 examples/s]37805 examples [00:42, 873.40 examples/s]37897 examples [00:42, 885.45 examples/s]37988 examples [00:42, 890.18 examples/s]38079 examples [00:43, 895.63 examples/s]38169 examples [00:43, 893.14 examples/s]38260 examples [00:43, 897.13 examples/s]38350 examples [00:43, 896.73 examples/s]38440 examples [00:43, 889.00 examples/s]38529 examples [00:43, 879.70 examples/s]38618 examples [00:43, 860.67 examples/s]38705 examples [00:43, 861.32 examples/s]38794 examples [00:43, 867.02 examples/s]38881 examples [00:43, 867.24 examples/s]38968 examples [00:44, 866.42 examples/s]39055 examples [00:44, 825.47 examples/s]39148 examples [00:44, 852.89 examples/s]39234 examples [00:44, 847.01 examples/s]39324 examples [00:44, 861.66 examples/s]39417 examples [00:44, 878.35 examples/s]39509 examples [00:44, 888.05 examples/s]39599 examples [00:44, 858.68 examples/s]39692 examples [00:44, 878.21 examples/s]39787 examples [00:44, 897.27 examples/s]39882 examples [00:45, 911.41 examples/s]39974 examples [00:45, 905.71 examples/s]40065 examples [00:45, 836.85 examples/s]40161 examples [00:45, 868.64 examples/s]40254 examples [00:45, 885.26 examples/s]40345 examples [00:45, 891.88 examples/s]40437 examples [00:45, 898.64 examples/s]40528 examples [00:45, 884.84 examples/s]40617 examples [00:45, 885.87 examples/s]40708 examples [00:46, 892.68 examples/s]40798 examples [00:46, 893.14 examples/s]40888 examples [00:46, 888.10 examples/s]40978 examples [00:46, 891.01 examples/s]41068 examples [00:46, 887.78 examples/s]41158 examples [00:46, 888.44 examples/s]41251 examples [00:46, 898.33 examples/s]41342 examples [00:46, 898.78 examples/s]41435 examples [00:46, 907.22 examples/s]41526 examples [00:46, 899.35 examples/s]41616 examples [00:47, 898.21 examples/s]41711 examples [00:47, 910.70 examples/s]41803 examples [00:47, 910.76 examples/s]41895 examples [00:47, 883.89 examples/s]41984 examples [00:47, 882.03 examples/s]42077 examples [00:47, 894.76 examples/s]42175 examples [00:47, 918.29 examples/s]42268 examples [00:47, 912.81 examples/s]42360 examples [00:47, 897.63 examples/s]42450 examples [00:47, 887.61 examples/s]42543 examples [00:48, 899.89 examples/s]42640 examples [00:48, 919.60 examples/s]42733 examples [00:48, 905.44 examples/s]42825 examples [00:48, 908.23 examples/s]42916 examples [00:48, 902.79 examples/s]43007 examples [00:48, 904.71 examples/s]43098 examples [00:48, 903.16 examples/s]43189 examples [00:48, 900.69 examples/s]43286 examples [00:48, 920.25 examples/s]43379 examples [00:48, 917.96 examples/s]43473 examples [00:49, 922.59 examples/s]43566 examples [00:49, 919.63 examples/s]43659 examples [00:49, 913.00 examples/s]43751 examples [00:49, 890.82 examples/s]43841 examples [00:49, 875.96 examples/s]43930 examples [00:49, 879.45 examples/s]44027 examples [00:49, 902.76 examples/s]44122 examples [00:49, 915.10 examples/s]44214 examples [00:49, 879.96 examples/s]44309 examples [00:50, 897.48 examples/s]44404 examples [00:50, 910.56 examples/s]44496 examples [00:50, 912.51 examples/s]44589 examples [00:50, 915.11 examples/s]44681 examples [00:50, 914.70 examples/s]44775 examples [00:50, 919.76 examples/s]44869 examples [00:50, 923.79 examples/s]44963 examples [00:50, 925.66 examples/s]45056 examples [00:50, 850.06 examples/s]45148 examples [00:50, 868.10 examples/s]45245 examples [00:51, 894.01 examples/s]45336 examples [00:51, 897.16 examples/s]45427 examples [00:51, 885.93 examples/s]45517 examples [00:51, 884.44 examples/s]45608 examples [00:51, 889.76 examples/s]45699 examples [00:51, 893.83 examples/s]45791 examples [00:51, 899.06 examples/s]45882 examples [00:51, 897.97 examples/s]45972 examples [00:51, 894.75 examples/s]46062 examples [00:51, 891.03 examples/s]46152 examples [00:52, 874.13 examples/s]46245 examples [00:52, 888.29 examples/s]46338 examples [00:52, 899.75 examples/s]46429 examples [00:52, 888.80 examples/s]46518 examples [00:52, 837.04 examples/s]46608 examples [00:52, 853.15 examples/s]46701 examples [00:52, 874.49 examples/s]46791 examples [00:52, 881.80 examples/s]46880 examples [00:52, 872.70 examples/s]46968 examples [00:53, 871.07 examples/s]47060 examples [00:53, 882.71 examples/s]47149 examples [00:53, 884.45 examples/s]47238 examples [00:53, 882.00 examples/s]47328 examples [00:53, 886.13 examples/s]47421 examples [00:53, 896.33 examples/s]47514 examples [00:53, 903.53 examples/s]47605 examples [00:53, 886.83 examples/s]47694 examples [00:53, 871.67 examples/s]47782 examples [00:53, 865.03 examples/s]47873 examples [00:54, 875.32 examples/s]47967 examples [00:54, 891.22 examples/s]48057 examples [00:54, 890.60 examples/s]48149 examples [00:54, 899.03 examples/s]48239 examples [00:54, 877.45 examples/s]48327 examples [00:54, 875.62 examples/s]48418 examples [00:54, 885.37 examples/s]48507 examples [00:54, 877.54 examples/s]48598 examples [00:54, 885.76 examples/s]48689 examples [00:54, 890.46 examples/s]48780 examples [00:55, 894.37 examples/s]48872 examples [00:55, 901.65 examples/s]48968 examples [00:55, 916.55 examples/s]49060 examples [00:55, 914.25 examples/s]49152 examples [00:55, 886.27 examples/s]49243 examples [00:55, 891.51 examples/s]49337 examples [00:55, 900.75 examples/s]49433 examples [00:55, 915.24 examples/s]49527 examples [00:55, 921.03 examples/s]49620 examples [00:55, 922.40 examples/s]49716 examples [00:56, 931.17 examples/s]49810 examples [00:56, 921.44 examples/s]49903 examples [00:56, 922.56 examples/s]49996 examples [00:56, 911.14 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6119/50000 [00:00<00:00, 61182.91 examples/s] 35%|███▌      | 17518/50000 [00:00<00:00, 71057.18 examples/s] 58%|█████▊    | 29014/50000 [00:00<00:00, 80251.08 examples/s] 79%|███████▉  | 39439/50000 [00:00<00:00, 86203.56 examples/s]                                                               0 examples [00:00, ? examples/s]74 examples [00:00, 734.72 examples/s]170 examples [00:00, 786.40 examples/s]263 examples [00:00, 824.07 examples/s]362 examples [00:00, 866.99 examples/s]457 examples [00:00, 889.40 examples/s]553 examples [00:00, 907.11 examples/s]652 examples [00:00, 929.07 examples/s]750 examples [00:00, 941.62 examples/s]842 examples [00:00, 934.05 examples/s]933 examples [00:01, 910.83 examples/s]1024 examples [00:01, 908.12 examples/s]1114 examples [00:01, 904.56 examples/s]1210 examples [00:01, 919.76 examples/s]1304 examples [00:01, 924.41 examples/s]1397 examples [00:01, 915.67 examples/s]1489 examples [00:01, 906.89 examples/s]1584 examples [00:01, 918.09 examples/s]1679 examples [00:01, 925.20 examples/s]1774 examples [00:01, 930.27 examples/s]1868 examples [00:02, 925.54 examples/s]1961 examples [00:02, 922.21 examples/s]2056 examples [00:02, 928.31 examples/s]2151 examples [00:02, 934.07 examples/s]2245 examples [00:02, 935.71 examples/s]2339 examples [00:02, 914.30 examples/s]2432 examples [00:02, 917.64 examples/s]2524 examples [00:02, 915.67 examples/s]2616 examples [00:02, 915.08 examples/s]2709 examples [00:02, 919.02 examples/s]2802 examples [00:03, 920.72 examples/s]2895 examples [00:03, 920.20 examples/s]2990 examples [00:03, 928.22 examples/s]3085 examples [00:03, 934.59 examples/s]3179 examples [00:03, 935.81 examples/s]3273 examples [00:03, 930.59 examples/s]3367 examples [00:03, 909.91 examples/s]3459 examples [00:03, 902.59 examples/s]3550 examples [00:03, 880.44 examples/s]3641 examples [00:03, 886.47 examples/s]3733 examples [00:04, 894.81 examples/s]3828 examples [00:04, 908.72 examples/s]3920 examples [00:04, 910.56 examples/s]4015 examples [00:04, 921.29 examples/s]4108 examples [00:04, 915.19 examples/s]4200 examples [00:04, 902.63 examples/s]4291 examples [00:04, 890.29 examples/s]4383 examples [00:04, 896.98 examples/s]4473 examples [00:04, 885.65 examples/s]4567 examples [00:04, 900.38 examples/s]4665 examples [00:05, 919.00 examples/s]4758 examples [00:05, 920.18 examples/s]4851 examples [00:05, 914.73 examples/s]4943 examples [00:05, 879.82 examples/s]5037 examples [00:05, 894.97 examples/s]5127 examples [00:05, 884.34 examples/s]5224 examples [00:05, 905.94 examples/s]5320 examples [00:05, 920.25 examples/s]5416 examples [00:05, 930.73 examples/s]5510 examples [00:06, 931.26 examples/s]5604 examples [00:06, 922.06 examples/s]5698 examples [00:06, 926.53 examples/s]5791 examples [00:06, 909.87 examples/s]5884 examples [00:06, 915.24 examples/s]5976 examples [00:06, 914.25 examples/s]6068 examples [00:06, 902.93 examples/s]6159 examples [00:06, 904.92 examples/s]6255 examples [00:06, 920.56 examples/s]6349 examples [00:06, 924.33 examples/s]6442 examples [00:07, 924.39 examples/s]6542 examples [00:07, 943.43 examples/s]6640 examples [00:07, 952.05 examples/s]6736 examples [00:07, 948.38 examples/s]6831 examples [00:07, 933.63 examples/s]6925 examples [00:07, 921.24 examples/s]7018 examples [00:07, 889.03 examples/s]7108 examples [00:07, 884.70 examples/s]7198 examples [00:07, 886.99 examples/s]7287 examples [00:07, 819.23 examples/s]7382 examples [00:08, 852.71 examples/s]7475 examples [00:08, 874.10 examples/s]7572 examples [00:08, 899.24 examples/s]7668 examples [00:08, 911.99 examples/s]7762 examples [00:08, 917.44 examples/s]7857 examples [00:08, 926.65 examples/s]7951 examples [00:08, 924.87 examples/s]8048 examples [00:08, 934.75 examples/s]8145 examples [00:08, 943.56 examples/s]8240 examples [00:09, 931.73 examples/s]8334 examples [00:09, 926.09 examples/s]8427 examples [00:09, 924.22 examples/s]8523 examples [00:09, 933.58 examples/s]8618 examples [00:09, 937.90 examples/s]8712 examples [00:09, 934.91 examples/s]8811 examples [00:09, 949.11 examples/s]8906 examples [00:09, 948.30 examples/s]9001 examples [00:09, 946.56 examples/s]9096 examples [00:09, 929.97 examples/s]9190 examples [00:10, 928.93 examples/s]9288 examples [00:10, 941.80 examples/s]9383 examples [00:10, 940.61 examples/s]9478 examples [00:10, 929.74 examples/s]9572 examples [00:10, 920.43 examples/s]9668 examples [00:10, 929.78 examples/s]9766 examples [00:10, 941.70 examples/s]9861 examples [00:10, 937.27 examples/s]9956 examples [00:10, 938.34 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLHUZEJ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLHUZEJ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe7ded96950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe7ded96950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe7ded96950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe76910fe48>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe7dce70ba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe7ded96950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe7ded96950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe7ded96950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe7dd3050f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe7dd305518>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe763090ea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe763090ea0> 

  function with postional parmater data_info <function split_train_valid at 0x7fe763090ea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe76308f048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe76308f048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe76308f048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=aa934d498d39e6f07d031719dfe1fa49affc27139da393cc593d2c7946945c59
  Stored in directory: /tmp/pip-ephem-wheel-cache-5upraktg/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:02:32, 11.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:58:07, 16.0kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<10:31:33, 22.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 811k/862M [00:01<7:22:30, 32.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.77M/862M [00:01<5:09:52, 46.3kB/s].vector_cache/glove.6B.zip:   1%|          | 4.53M/862M [00:01<3:36:22, 66.1kB/s].vector_cache/glove.6B.zip:   1%|          | 6.18M/862M [00:01<2:32:15, 93.7kB/s].vector_cache/glove.6B.zip:   1%|          | 10.7M/862M [00:01<1:46:06, 134kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 14.2M/862M [00:01<1:14:05, 191kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.3M/862M [00:01<51:43, 272kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 22.9M/862M [00:02<36:06, 387kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:02<25:17, 551kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.3M/862M [00:02<17:43, 782kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.8M/862M [00:02<12:40, 1.09MB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.7M/862M [00:02<09:03, 1.52MB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.2M/862M [00:02<06:26, 2.14MB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.7M/862M [00:02<04:39, 2.94MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:02<03:23, 4.02MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.7M/862M [00:02<02:29, 5.45MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.9M/862M [00:03<01:52, 7.25MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:03<01:35, 8.51MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:05<03:01, 4.43MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:05<05:25, 2.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<04:37, 2.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:05<03:25, 3.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:07<10:54, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:07<09:17, 1.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:07<06:54, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:09<07:32, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:09<08:33, 1.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<06:48, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:09<04:56, 2.68MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:11<11:01, 1.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:11<09:17, 1.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<06:51, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:13<07:29, 1.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:13<06:52, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:13<05:09, 2.54MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:15<06:16, 2.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:15<06:00, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:15<04:36, 2.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:17<05:52, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:17<05:44, 2.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<04:24, 2.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:17<03:15, 3.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:19<34:01, 381kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:19<25:24, 510kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:19<18:06, 714kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:19<12:48, 1.01MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:21<45:32, 283kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:21<33:25, 385kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:21<23:44, 541kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<16:44, 766kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:23<1:35:41, 134kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:23<1:08:30, 187kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:23<48:14, 265kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:25<36:16, 351kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:25<26:43, 477kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:25<19:01, 669kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<13:29, 941kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<17:01, 745kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<13:26, 943kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<09:46, 1.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<09:26, 1.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<09:43, 1.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<07:29, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<05:28, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:10, 1.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<06:34, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<04:58, 2.51MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:01, 2.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<07:18, 1.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:52, 2.12MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<04:17, 2.89MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:52, 1.14MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<09:08, 1.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<06:46, 1.82MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:14, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<06:34, 1.87MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<04:58, 2.47MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:59, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<07:14, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:49, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:39<04:13, 2.89MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:47, 1.56MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<07:00, 1.74MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:13, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<03:49, 3.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<19:25, 624kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<15:04, 804kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<10:53, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:05, 1.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<10:01, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<07:44, 1.55MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:34, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<10:46, 1.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:00, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:36, 1.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<04:48, 2.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<12:22, 963kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<10:08, 1.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<07:27, 1.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:37, 1.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:14, 1.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<06:29, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<04:41, 2.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:22, 1.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:16, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<05:23, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<03:54, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<29:46, 393kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<23:46, 492kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<17:20, 674kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<12:16, 949kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<15:59, 728kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<12:35, 924kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<09:05, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:28, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<2:23:03, 80.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<1:41:32, 114kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<1:11:14, 162kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<49:49, 231kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<2:13:43, 86.0kB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<1:36:32, 119kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<1:08:13, 168kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<47:45, 240kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<40:40, 281kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<29:52, 382kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<21:13, 537kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<17:07, 663kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<14:38, 775kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<10:56, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:45, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<12:36, 896kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<10:12, 1.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<07:28, 1.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:30, 1.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:32, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:51, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:31, 3.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<53:52, 207kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<40:20, 276kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<28:53, 386kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<20:16, 547kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<20:32, 540kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<15:43, 704kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<11:19, 976kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<10:09, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<09:52, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:34, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:27, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:29, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:35, 1.66MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:54, 2.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:39, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:20, 2.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<04:02, 2.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<02:59, 3.62MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:38, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<07:23, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:26, 1.98MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:00, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:32, 1.94MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<04:12, 2.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:06, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:57, 1.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:42, 2.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:26, 3.09MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:26<06:39, 1.59MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:51, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:23, 2.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:20, 1.97MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:41, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:21, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:00, 2.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<02:57, 3.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<12:18, 849kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<09:54, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:14, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<07:10, 1.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:18, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:43, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<05:24, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<06:18, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:03, 2.04MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<03:41, 2.78MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<09:01, 1.14MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<07:34, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:33, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:02, 2.52MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<09:20, 1.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<09:03, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:52, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<05:00, 2.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:02, 1.67MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:27, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:06, 2.45MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:01, 3.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<08:24, 1.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<07:07, 1.41MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:14, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:42, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<06:27, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:07, 1.94MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:42, 2.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<08:45, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:13, 1.37MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:17, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<05:48, 1.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:16, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:59, 2.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<04:46, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:44, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:33, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:18, 2.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<06:52, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<06:00, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:29, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:05, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:56, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<04:40, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:25, 2.80MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<05:28, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<04:53, 1.95MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<03:39, 2.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<04:35, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<05:32, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:23, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:11, 2.95MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<05:56, 1.58MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:18, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:59, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<04:42, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<05:34, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:28, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:15, 2.84MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<08:11, 1.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<06:45, 1.37MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:57, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:27, 1.69MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<06:04, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<04:49, 1.91MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:30, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<08:11, 1.11MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<06:51, 1.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:03, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<05:22, 1.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<06:00, 1.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:42, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:24, 2.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<05:41, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:05, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:49, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<04:29, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<05:20, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:16, 2.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:06, 2.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<07:49, 1.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<06:34, 1.35MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:49, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<05:10, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<05:46, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<04:35, 1.91MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:19, 2.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<07:38, 1.14MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<06:23, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:41, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<05:03, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<05:29, 1.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:15, 2.03MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:09, 2.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<04:23, 1.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:02, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:01, 2.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<03:56, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:49, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<02:53, 2.94MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<03:45, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:39, 2.31MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<02:48, 2.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:40, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<04:41, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:48, 2.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:46, 3.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<07:02, 1.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<05:58, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:23, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<03:10, 2.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<11:01, 747kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<08:34, 959kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<06:19, 1.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:31, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<08:03, 1.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<07:41, 1.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:52, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<04:13, 1.92MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<08:06, 999kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<06:27, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:46, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:25, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<08:57, 897kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<08:14, 974kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<06:13, 1.29MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:27, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:59, 995kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<06:33, 1.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:49, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:58, 1.59MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<04:26, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:20, 2.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:55, 1.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<04:49, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:51, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:48, 2.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<06:53, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<05:43, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:10, 1.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:34, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<04:09, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:06, 2.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:43, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:32, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<02:40, 2.83MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:24, 2.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<04:15, 1.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:26, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<02:30, 2.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<06:23, 1.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:10, 1.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:08, 1.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:59, 2.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<06:53, 1.07MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<05:38, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:06, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:58, 2.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<09:44, 753kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<08:36, 851kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<06:28, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:37, 1.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<07:50, 926kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<06:19, 1.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:36, 1.57MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:46, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<05:10, 1.39MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<04:03, 1.77MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:57, 2.42MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:14, 1.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:47, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<02:49, 2.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:30, 2.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:10, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<02:26, 2.88MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<01:48, 3.88MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:26, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:34, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:33, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<03:24, 2.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:27, 2.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<09:18, 743kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<07:21, 940kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<05:20, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<05:07, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<05:15, 1.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<04:03, 1.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:54, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:58, 1.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:07, 1.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:04, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:14, 3.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<07:36, 883kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:02, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:40, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:20, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:26, 1.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:34, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:41, 1.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:29, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:31, 2.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:29, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:40, 1.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:26, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:30, 2.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<05:51, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:42, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:39, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<02:36, 2.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<08:37, 746kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<07:36, 845kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:38, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<04:03, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:40, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:03, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:01, 2.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:23, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:48, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:57, 2.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:09, 2.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:06, 1.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:34, 1.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:38, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:10, 1.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:44, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:59, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:10, 2.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:00, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:16, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:08, 1.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:15, 2.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<31:28, 191kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<22:44, 264kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<16:01, 374kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<11:15, 530kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<12:09, 489kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<09:13, 644kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<06:37, 894kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:48, 1.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:22, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:05, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<02:55, 2.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<05:09, 1.13MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:19, 1.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:11, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:23, 1.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:04, 1.86MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:19, 2.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:46, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:38, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:00, 2.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:32, 2.20MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:27, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<01:51, 3.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<02:25, 2.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<03:03, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:28, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<01:47, 3.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<04:23, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:44, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:46, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:02, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:26, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:44, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:58, 2.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:26, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:46, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:00, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:23, 1.55MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:38, 1.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:54, 2.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:38, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:10, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:22, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:41, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<03:08, 1.62MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:30, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:49, 2.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<04:24, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:41, 1.36MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:43, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:54, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:16, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:35, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:52, 2.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<04:07, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:23, 1.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:30, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<01:48, 2.68MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<13:20, 362kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<09:55, 487kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<07:03, 681kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<05:52, 811kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<04:41, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:24, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:19, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:29, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:41, 1.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:57, 2.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<02:42, 1.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:24, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<01:48, 2.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:14, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:45, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:09, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:35, 2.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:21, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:12, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:39, 2.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:03, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:31, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:59, 2.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:27, 3.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:34, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:18, 1.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:44, 2.48MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:05, 2.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:30, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:58, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:27, 2.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<02:09, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:01, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:32, 2.71MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<01:55, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:22, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<01:54, 2.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:22, 2.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<03:29, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<02:56, 1.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:09, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:32, 2.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<07:34, 528kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<06:11, 646kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<04:31, 883kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<03:10, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<04:38, 848kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<03:41, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:40, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:41, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:21, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:45, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<01:59, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:19, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:49, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:20, 2.79MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:51, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:45, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:19, 2.80MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<00:57, 3.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<1:49:59, 33.3kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<1:17:16, 47.3kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<53:50, 67.4kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<38:03, 94.3kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<27:02, 132kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<18:54, 188kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<13:48, 255kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<10:31, 334kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<07:31, 465kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<05:15, 659kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<05:20, 645kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<04:09, 828kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:59, 1.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:45, 1.22MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:53<02:48, 1.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<02:08, 1.58MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:31, 2.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:40, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<02:16, 1.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:40, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:49, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:40, 1.94MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:15, 2.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:31, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:50, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:28, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:04, 2.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:43, 1.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<02:17, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:40, 1.83MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:46, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:37, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:12, 2.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:26, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:44, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:22, 2.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<00:59, 2.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:34, 1.84MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:27, 1.99MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:05, 2.65MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<00:46, 3.62MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<28:46, 98.2kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<20:51, 135kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<14:43, 191kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<10:11, 272kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<08:51, 311kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<06:31, 421kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<04:36, 591kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<03:43, 720kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<03:16, 818kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:27, 1.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:43, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:42, 965kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:12, 1.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:36, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:37, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:43, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:20, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:57, 2.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<03:26, 719kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:38, 937kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:00, 1.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:25, 1.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:34, 933kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:06, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:36, 1.50MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:08, 2.07MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:39, 1.40MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:41, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:18, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:55, 2.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:23, 670kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:37, 864kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:52, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:46, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:30, 1.46MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:05, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:47, 2.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:56, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:54, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:26, 1.46MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:02, 2.00MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:13, 1.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:06, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:49, 2.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:58, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:55, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:41, 2.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:30, 3.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:34, 1.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:20, 1.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:59, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:03, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:11, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:55, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:39, 2.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:13, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:04, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:47, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:53, 1.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:50, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:37, 2.72MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:45, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:56, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:44, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:32, 2.96MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:46, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:42, 2.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:32, 2.88MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:41, 2.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:51, 1.76MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:41, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:29, 2.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:15, 1.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:02, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:45, 1.86MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:32, 2.56MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<02:02, 669kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:35, 855kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:08, 1.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:02, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:03, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:48, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:34, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:06, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:55, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:40, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:41, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:45, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:35, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:25, 2.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:34, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:31, 2.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:23, 2.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:28, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:34, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:27, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:19, 3.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:33, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:30, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:22, 2.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:25, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:31, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:16, 2.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:33, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:29, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:21, 2.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:23, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:25, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:19, 2.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:13, 3.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:26, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:22, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:16, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:18, 1.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:21, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:17, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:11, 2.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:28, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:23, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:16, 1.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:16, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:14, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:10, 2.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:11, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:07, 2.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:10, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:05, 2.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:13, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:11, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:07, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 2.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:06, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 1.81MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 1.69MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.87MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.48MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<22:33:36,  4.93it/s]  0%|          | 718/400000 [00:00<15:46:06,  7.03it/s]  0%|          | 1452/400000 [00:00<11:01:19, 10.04it/s]  1%|          | 2183/400000 [00:00<7:42:21, 14.34it/s]   1%|          | 2944/400000 [00:00<5:23:17, 20.47it/s]  1%|          | 3711/400000 [00:00<3:46:07, 29.21it/s]  1%|          | 4493/400000 [00:00<2:38:13, 41.66it/s]  1%|▏         | 5244/400000 [00:00<1:50:48, 59.37it/s]  1%|▏         | 5981/400000 [00:01<1:17:41, 84.53it/s]  2%|▏         | 6727/400000 [00:01<54:32, 120.17it/s]   2%|▏         | 7450/400000 [00:01<38:22, 170.46it/s]  2%|▏         | 8215/400000 [00:01<27:04, 241.20it/s]  2%|▏         | 8971/400000 [00:01<19:10, 339.93it/s]  2%|▏         | 9730/400000 [00:01<13:39, 476.46it/s]  3%|▎         | 10476/400000 [00:01<09:48, 661.58it/s]  3%|▎         | 11233/400000 [00:01<07:06, 910.98it/s]  3%|▎         | 11995/400000 [00:01<05:13, 1237.94it/s]  3%|▎         | 12743/400000 [00:01<03:54, 1649.34it/s]  3%|▎         | 13496/400000 [00:02<02:59, 2153.95it/s]  4%|▎         | 14247/400000 [00:02<02:20, 2740.12it/s]  4%|▍         | 15013/400000 [00:02<01:53, 3393.44it/s]  4%|▍         | 15772/400000 [00:02<01:34, 4068.06it/s]  4%|▍         | 16531/400000 [00:02<01:21, 4724.95it/s]  4%|▍         | 17298/400000 [00:02<01:11, 5338.75it/s]  5%|▍         | 18057/400000 [00:02<01:05, 5821.41it/s]  5%|▍         | 18822/400000 [00:02<01:00, 6269.93it/s]  5%|▍         | 19579/400000 [00:02<00:58, 6555.54it/s]  5%|▌         | 20329/400000 [00:02<00:56, 6732.10it/s]  5%|▌         | 21080/400000 [00:03<00:54, 6946.76it/s]  5%|▌         | 21852/400000 [00:03<00:52, 7161.32it/s]  6%|▌         | 22616/400000 [00:03<00:51, 7298.20it/s]  6%|▌         | 23394/400000 [00:03<00:50, 7433.85it/s]  6%|▌         | 24156/400000 [00:03<00:50, 7467.45it/s]  6%|▌         | 24916/400000 [00:03<00:50, 7358.32it/s]  6%|▋         | 25705/400000 [00:03<00:49, 7510.00it/s]  7%|▋         | 26483/400000 [00:03<00:49, 7588.09it/s]  7%|▋         | 27248/400000 [00:03<00:49, 7605.55it/s]  7%|▋         | 28013/400000 [00:03<00:48, 7611.62it/s]  7%|▋         | 28792/400000 [00:04<00:48, 7661.36it/s]  7%|▋         | 29561/400000 [00:04<00:48, 7651.60it/s]  8%|▊         | 30348/400000 [00:04<00:47, 7714.50it/s]  8%|▊         | 31121/400000 [00:04<00:48, 7667.58it/s]  8%|▊         | 31889/400000 [00:04<00:48, 7540.96it/s]  8%|▊         | 32645/400000 [00:04<00:49, 7432.75it/s]  8%|▊         | 33390/400000 [00:04<00:50, 7299.50it/s]  9%|▊         | 34122/400000 [00:04<00:50, 7302.51it/s]  9%|▊         | 34863/400000 [00:04<00:49, 7332.48it/s]  9%|▉         | 35597/400000 [00:04<00:50, 7253.94it/s]  9%|▉         | 36342/400000 [00:05<00:49, 7310.99it/s]  9%|▉         | 37081/400000 [00:05<00:49, 7333.31it/s]  9%|▉         | 37832/400000 [00:05<00:49, 7383.61it/s] 10%|▉         | 38580/400000 [00:05<00:48, 7410.73it/s] 10%|▉         | 39328/400000 [00:05<00:48, 7431.04it/s] 10%|█         | 40072/400000 [00:05<00:48, 7407.09it/s] 10%|█         | 40837/400000 [00:05<00:48, 7478.33it/s] 10%|█         | 41586/400000 [00:05<00:48, 7451.25it/s] 11%|█         | 42345/400000 [00:05<00:47, 7491.83it/s] 11%|█         | 43095/400000 [00:05<00:47, 7483.95it/s] 11%|█         | 43844/400000 [00:06<00:47, 7461.94it/s] 11%|█         | 44591/400000 [00:06<00:47, 7429.35it/s] 11%|█▏        | 45337/400000 [00:06<00:47, 7436.50it/s] 12%|█▏        | 46081/400000 [00:06<00:48, 7369.63it/s] 12%|█▏        | 46843/400000 [00:06<00:47, 7441.17it/s] 12%|█▏        | 47588/400000 [00:06<00:47, 7438.74it/s] 12%|█▏        | 48338/400000 [00:06<00:47, 7454.19it/s] 12%|█▏        | 49084/400000 [00:06<00:47, 7405.82it/s] 12%|█▏        | 49849/400000 [00:06<00:46, 7474.91it/s] 13%|█▎        | 50615/400000 [00:06<00:46, 7528.76it/s] 13%|█▎        | 51369/400000 [00:07<00:46, 7478.19it/s] 13%|█▎        | 52137/400000 [00:07<00:46, 7537.54it/s] 13%|█▎        | 52906/400000 [00:07<00:45, 7581.43it/s] 13%|█▎        | 53690/400000 [00:07<00:45, 7654.65it/s] 14%|█▎        | 54456/400000 [00:07<00:45, 7632.33it/s] 14%|█▍        | 55220/400000 [00:07<00:45, 7628.75it/s] 14%|█▍        | 55984/400000 [00:07<00:45, 7611.66it/s] 14%|█▍        | 56750/400000 [00:07<00:45, 7624.40it/s] 14%|█▍        | 57513/400000 [00:07<00:46, 7339.32it/s] 15%|█▍        | 58250/400000 [00:07<00:47, 7224.66it/s] 15%|█▍        | 58997/400000 [00:08<00:46, 7295.16it/s] 15%|█▍        | 59756/400000 [00:08<00:46, 7379.67it/s] 15%|█▌        | 60517/400000 [00:08<00:45, 7446.79it/s] 15%|█▌        | 61263/400000 [00:08<00:45, 7439.49it/s] 16%|█▌        | 62010/400000 [00:08<00:45, 7447.25it/s] 16%|█▌        | 62768/400000 [00:08<00:45, 7486.24it/s] 16%|█▌        | 63518/400000 [00:08<00:45, 7473.85it/s] 16%|█▌        | 64266/400000 [00:08<00:45, 7460.73it/s] 16%|█▋        | 65013/400000 [00:08<00:45, 7384.85it/s] 16%|█▋        | 65752/400000 [00:09<00:47, 7102.60it/s] 17%|█▋        | 66495/400000 [00:09<00:46, 7197.21it/s] 17%|█▋        | 67264/400000 [00:09<00:45, 7336.59it/s] 17%|█▋        | 68028/400000 [00:09<00:44, 7423.50it/s] 17%|█▋        | 68822/400000 [00:09<00:43, 7568.88it/s] 17%|█▋        | 69581/400000 [00:09<00:43, 7572.45it/s] 18%|█▊        | 70340/400000 [00:09<00:44, 7446.95it/s] 18%|█▊        | 71102/400000 [00:09<00:43, 7495.42it/s] 18%|█▊        | 71867/400000 [00:09<00:43, 7539.23it/s] 18%|█▊        | 72622/400000 [00:09<00:43, 7502.27it/s] 18%|█▊        | 73373/400000 [00:10<00:44, 7369.95it/s] 19%|█▊        | 74149/400000 [00:10<00:43, 7479.68it/s] 19%|█▊        | 74911/400000 [00:10<00:43, 7520.72it/s] 19%|█▉        | 75666/400000 [00:10<00:43, 7526.73it/s] 19%|█▉        | 76434/400000 [00:10<00:42, 7571.62it/s] 19%|█▉        | 77192/400000 [00:10<00:43, 7460.02it/s] 19%|█▉        | 77954/400000 [00:10<00:42, 7507.12it/s] 20%|█▉        | 78716/400000 [00:10<00:42, 7538.91it/s] 20%|█▉        | 79471/400000 [00:10<00:43, 7449.75it/s] 20%|██        | 80237/400000 [00:10<00:42, 7509.30it/s] 20%|██        | 80989/400000 [00:11<00:42, 7483.28it/s] 20%|██        | 81743/400000 [00:11<00:42, 7497.98it/s] 21%|██        | 82494/400000 [00:11<00:42, 7499.16it/s] 21%|██        | 83245/400000 [00:11<00:42, 7459.53it/s] 21%|██        | 83999/400000 [00:11<00:42, 7481.26it/s] 21%|██        | 84748/400000 [00:11<00:42, 7475.01it/s] 21%|██▏       | 85505/400000 [00:11<00:41, 7502.81it/s] 22%|██▏       | 86256/400000 [00:11<00:42, 7467.63it/s] 22%|██▏       | 87003/400000 [00:11<00:42, 7425.44it/s] 22%|██▏       | 87755/400000 [00:11<00:41, 7451.69it/s] 22%|██▏       | 88512/400000 [00:12<00:41, 7485.72it/s] 22%|██▏       | 89265/400000 [00:12<00:41, 7495.65it/s] 23%|██▎       | 90020/400000 [00:12<00:41, 7509.79it/s] 23%|██▎       | 90772/400000 [00:12<00:41, 7404.25it/s] 23%|██▎       | 91513/400000 [00:12<00:42, 7210.51it/s] 23%|██▎       | 92236/400000 [00:12<00:43, 7113.68it/s] 23%|██▎       | 92994/400000 [00:12<00:42, 7245.27it/s] 23%|██▎       | 93749/400000 [00:12<00:41, 7332.39it/s] 24%|██▎       | 94500/400000 [00:12<00:41, 7382.32it/s] 24%|██▍       | 95240/400000 [00:12<00:41, 7378.08it/s] 24%|██▍       | 95979/400000 [00:13<00:41, 7340.32it/s] 24%|██▍       | 96745/400000 [00:13<00:40, 7433.10it/s] 24%|██▍       | 97522/400000 [00:13<00:40, 7531.01it/s] 25%|██▍       | 98300/400000 [00:13<00:39, 7603.75it/s] 25%|██▍       | 99062/400000 [00:13<00:39, 7600.00it/s] 25%|██▍       | 99823/400000 [00:13<00:39, 7585.05it/s] 25%|██▌       | 100624/400000 [00:13<00:38, 7707.11it/s] 25%|██▌       | 101396/400000 [00:13<00:39, 7499.90it/s] 26%|██▌       | 102164/400000 [00:13<00:39, 7550.88it/s] 26%|██▌       | 102921/400000 [00:13<00:39, 7500.77it/s] 26%|██▌       | 103674/400000 [00:14<00:39, 7507.48it/s] 26%|██▌       | 104438/400000 [00:14<00:39, 7545.63it/s] 26%|██▋       | 105201/400000 [00:14<00:38, 7568.69it/s] 26%|██▋       | 105959/400000 [00:14<00:38, 7560.30it/s] 27%|██▋       | 106738/400000 [00:14<00:38, 7626.71it/s] 27%|██▋       | 107502/400000 [00:14<00:39, 7486.16it/s] 27%|██▋       | 108252/400000 [00:14<00:39, 7445.45it/s] 27%|██▋       | 109008/400000 [00:14<00:38, 7477.55it/s] 27%|██▋       | 109757/400000 [00:14<00:39, 7442.09it/s] 28%|██▊       | 110502/400000 [00:14<00:39, 7335.78it/s] 28%|██▊       | 111248/400000 [00:15<00:39, 7370.45it/s] 28%|██▊       | 112004/400000 [00:15<00:38, 7425.93it/s] 28%|██▊       | 112774/400000 [00:15<00:38, 7503.17it/s] 28%|██▊       | 113535/400000 [00:15<00:38, 7533.48it/s] 29%|██▊       | 114313/400000 [00:15<00:37, 7605.53it/s] 29%|██▉       | 115074/400000 [00:15<00:37, 7559.50it/s] 29%|██▉       | 115833/400000 [00:15<00:37, 7567.19it/s] 29%|██▉       | 116590/400000 [00:15<00:38, 7400.59it/s] 29%|██▉       | 117349/400000 [00:15<00:37, 7454.52it/s] 30%|██▉       | 118111/400000 [00:16<00:37, 7500.95it/s] 30%|██▉       | 118862/400000 [00:16<00:37, 7458.79it/s] 30%|██▉       | 119613/400000 [00:16<00:37, 7472.42it/s] 30%|███       | 120372/400000 [00:16<00:37, 7505.81it/s] 30%|███       | 121123/400000 [00:16<00:37, 7434.24it/s] 30%|███       | 121887/400000 [00:16<00:37, 7494.26it/s] 31%|███       | 122661/400000 [00:16<00:36, 7565.21it/s] 31%|███       | 123431/400000 [00:16<00:36, 7603.94it/s] 31%|███       | 124207/400000 [00:16<00:36, 7649.53it/s] 31%|███       | 124979/400000 [00:16<00:35, 7667.78it/s] 31%|███▏      | 125751/400000 [00:17<00:35, 7683.34it/s] 32%|███▏      | 126520/400000 [00:17<00:36, 7595.80it/s] 32%|███▏      | 127280/400000 [00:17<00:36, 7550.45it/s] 32%|███▏      | 128061/400000 [00:17<00:35, 7625.20it/s] 32%|███▏      | 128838/400000 [00:17<00:35, 7666.06it/s] 32%|███▏      | 129617/400000 [00:17<00:35, 7701.33it/s] 33%|███▎      | 130388/400000 [00:17<00:35, 7660.87it/s] 33%|███▎      | 131155/400000 [00:17<00:35, 7599.00it/s] 33%|███▎      | 131917/400000 [00:17<00:35, 7603.00it/s] 33%|███▎      | 132680/400000 [00:17<00:35, 7609.06it/s] 33%|███▎      | 133442/400000 [00:18<00:35, 7580.59it/s] 34%|███▎      | 134201/400000 [00:18<00:35, 7541.65it/s] 34%|███▎      | 134956/400000 [00:18<00:35, 7466.11it/s] 34%|███▍      | 135709/400000 [00:18<00:35, 7485.15it/s] 34%|███▍      | 136475/400000 [00:18<00:34, 7535.22it/s] 34%|███▍      | 137247/400000 [00:18<00:34, 7589.68it/s] 35%|███▍      | 138011/400000 [00:18<00:34, 7603.20it/s] 35%|███▍      | 138779/400000 [00:18<00:34, 7623.52it/s] 35%|███▍      | 139542/400000 [00:18<00:34, 7563.71it/s] 35%|███▌      | 140317/400000 [00:18<00:34, 7617.04it/s] 35%|███▌      | 141084/400000 [00:19<00:33, 7630.86it/s] 35%|███▌      | 141877/400000 [00:19<00:33, 7716.92it/s] 36%|███▌      | 142659/400000 [00:19<00:33, 7743.95it/s] 36%|███▌      | 143437/400000 [00:19<00:33, 7752.39it/s] 36%|███▌      | 144213/400000 [00:19<00:33, 7746.60it/s] 36%|███▌      | 144988/400000 [00:19<00:32, 7745.77it/s] 36%|███▋      | 145763/400000 [00:19<00:33, 7687.82it/s] 37%|███▋      | 146532/400000 [00:19<00:33, 7602.39it/s] 37%|███▋      | 147293/400000 [00:19<00:33, 7521.22it/s] 37%|███▋      | 148046/400000 [00:19<00:33, 7504.49it/s] 37%|███▋      | 148797/400000 [00:20<00:33, 7464.97it/s] 37%|███▋      | 149557/400000 [00:20<00:33, 7501.86it/s] 38%|███▊      | 150308/400000 [00:20<00:34, 7326.64it/s] 38%|███▊      | 151066/400000 [00:20<00:33, 7398.53it/s] 38%|███▊      | 151831/400000 [00:20<00:33, 7471.14it/s] 38%|███▊      | 152588/400000 [00:20<00:32, 7498.73it/s] 38%|███▊      | 153349/400000 [00:20<00:32, 7527.80it/s] 39%|███▊      | 154103/400000 [00:20<00:32, 7495.81it/s] 39%|███▊      | 154853/400000 [00:20<00:32, 7459.42it/s] 39%|███▉      | 155600/400000 [00:20<00:32, 7422.03it/s] 39%|███▉      | 156343/400000 [00:21<00:32, 7421.45it/s] 39%|███▉      | 157086/400000 [00:21<00:32, 7365.88it/s] 39%|███▉      | 157829/400000 [00:21<00:32, 7383.09it/s] 40%|███▉      | 158586/400000 [00:21<00:32, 7437.73it/s] 40%|███▉      | 159340/400000 [00:21<00:32, 7465.59it/s] 40%|████      | 160103/400000 [00:21<00:31, 7514.08it/s] 40%|████      | 160872/400000 [00:21<00:31, 7563.14it/s] 40%|████      | 161631/400000 [00:21<00:31, 7570.22it/s] 41%|████      | 162398/400000 [00:21<00:31, 7599.22it/s] 41%|████      | 163163/400000 [00:21<00:31, 7613.42it/s] 41%|████      | 163925/400000 [00:22<00:31, 7590.70it/s] 41%|████      | 164685/400000 [00:22<00:31, 7555.14it/s] 41%|████▏     | 165445/400000 [00:22<00:30, 7567.09it/s] 42%|████▏     | 166215/400000 [00:22<00:30, 7604.94it/s] 42%|████▏     | 166992/400000 [00:22<00:30, 7652.55it/s] 42%|████▏     | 167758/400000 [00:22<00:30, 7618.99it/s] 42%|████▏     | 168539/400000 [00:22<00:30, 7675.30it/s] 42%|████▏     | 169307/400000 [00:22<00:30, 7597.05it/s] 43%|████▎     | 170073/400000 [00:22<00:30, 7615.69it/s] 43%|████▎     | 170848/400000 [00:22<00:29, 7655.17it/s] 43%|████▎     | 171614/400000 [00:23<00:30, 7570.44it/s] 43%|████▎     | 172372/400000 [00:23<00:30, 7490.81it/s] 43%|████▎     | 173126/400000 [00:23<00:30, 7503.26it/s] 43%|████▎     | 173877/400000 [00:23<00:30, 7449.69it/s] 44%|████▎     | 174627/400000 [00:23<00:30, 7462.37it/s] 44%|████▍     | 175396/400000 [00:23<00:29, 7528.33it/s] 44%|████▍     | 176177/400000 [00:23<00:29, 7609.67it/s] 44%|████▍     | 176945/400000 [00:23<00:29, 7628.27it/s] 44%|████▍     | 177722/400000 [00:23<00:28, 7667.84it/s] 45%|████▍     | 178490/400000 [00:23<00:28, 7648.19it/s] 45%|████▍     | 179256/400000 [00:24<00:28, 7637.11it/s] 45%|████▌     | 180020/400000 [00:24<00:28, 7636.68it/s] 45%|████▌     | 180784/400000 [00:24<00:29, 7541.80it/s] 45%|████▌     | 181539/400000 [00:24<00:29, 7525.73it/s] 46%|████▌     | 182305/400000 [00:24<00:28, 7564.20it/s] 46%|████▌     | 183068/400000 [00:24<00:28, 7581.22it/s] 46%|████▌     | 183827/400000 [00:24<00:28, 7488.87it/s] 46%|████▌     | 184577/400000 [00:24<00:29, 7242.96it/s] 46%|████▋     | 185304/400000 [00:24<00:29, 7204.36it/s] 47%|████▋     | 186026/400000 [00:25<00:30, 6974.14it/s] 47%|████▋     | 186757/400000 [00:25<00:30, 7069.90it/s] 47%|████▋     | 187516/400000 [00:25<00:29, 7216.20it/s] 47%|████▋     | 188259/400000 [00:25<00:29, 7277.01it/s] 47%|████▋     | 189023/400000 [00:25<00:28, 7380.03it/s] 47%|████▋     | 189786/400000 [00:25<00:28, 7452.37it/s] 48%|████▊     | 190533/400000 [00:25<00:28, 7422.08it/s] 48%|████▊     | 191290/400000 [00:25<00:27, 7465.47it/s] 48%|████▊     | 192038/400000 [00:25<00:28, 7407.54it/s] 48%|████▊     | 192785/400000 [00:25<00:27, 7426.00it/s] 48%|████▊     | 193547/400000 [00:26<00:27, 7482.21it/s] 49%|████▊     | 194298/400000 [00:26<00:27, 7489.34it/s] 49%|████▉     | 195048/400000 [00:26<00:27, 7481.49it/s] 49%|████▉     | 195797/400000 [00:26<00:27, 7449.40it/s] 49%|████▉     | 196543/400000 [00:26<00:27, 7337.20it/s] 49%|████▉     | 197278/400000 [00:26<00:28, 7036.04it/s] 50%|████▉     | 198011/400000 [00:26<00:28, 7119.40it/s] 50%|████▉     | 198759/400000 [00:26<00:27, 7223.25it/s] 50%|████▉     | 199484/400000 [00:26<00:27, 7208.98it/s] 50%|█████     | 200241/400000 [00:26<00:27, 7313.42it/s] 50%|█████     | 200979/400000 [00:27<00:27, 7331.87it/s] 50%|█████     | 201739/400000 [00:27<00:26, 7408.65it/s] 51%|█████     | 202481/400000 [00:27<00:26, 7371.66it/s] 51%|█████     | 203234/400000 [00:27<00:26, 7418.34it/s] 51%|█████     | 203985/400000 [00:27<00:26, 7445.04it/s] 51%|█████     | 204741/400000 [00:27<00:26, 7478.21it/s] 51%|█████▏    | 205505/400000 [00:27<00:25, 7521.82it/s] 52%|█████▏    | 206258/400000 [00:27<00:25, 7499.58it/s] 52%|█████▏    | 207013/400000 [00:27<00:25, 7512.88it/s] 52%|█████▏    | 207767/400000 [00:27<00:25, 7519.78it/s] 52%|█████▏    | 208520/400000 [00:28<00:25, 7485.74it/s] 52%|█████▏    | 209269/400000 [00:28<00:26, 7314.07it/s] 53%|█████▎    | 210026/400000 [00:28<00:25, 7387.64it/s] 53%|█████▎    | 210776/400000 [00:28<00:25, 7420.01it/s] 53%|█████▎    | 211556/400000 [00:28<00:25, 7529.35it/s] 53%|█████▎    | 212310/400000 [00:28<00:25, 7481.50it/s] 53%|█████▎    | 213070/400000 [00:28<00:24, 7516.24it/s] 53%|█████▎    | 213824/400000 [00:28<00:24, 7522.36it/s] 54%|█████▎    | 214583/400000 [00:28<00:24, 7541.56it/s] 54%|█████▍    | 215340/400000 [00:28<00:24, 7549.28it/s] 54%|█████▍    | 216096/400000 [00:29<00:24, 7519.08it/s] 54%|█████▍    | 216849/400000 [00:29<00:24, 7499.96it/s] 54%|█████▍    | 217611/400000 [00:29<00:24, 7534.18it/s] 55%|█████▍    | 218376/400000 [00:29<00:23, 7567.98it/s] 55%|█████▍    | 219155/400000 [00:29<00:23, 7632.35it/s] 55%|█████▍    | 219919/400000 [00:29<00:23, 7610.44it/s] 55%|█████▌    | 220684/400000 [00:29<00:23, 7620.60it/s] 55%|█████▌    | 221447/400000 [00:29<00:23, 7586.74it/s] 56%|█████▌    | 222206/400000 [00:29<00:23, 7474.57it/s] 56%|█████▌    | 222954/400000 [00:29<00:24, 7317.19it/s] 56%|█████▌    | 223707/400000 [00:30<00:23, 7379.08it/s] 56%|█████▌    | 224466/400000 [00:30<00:23, 7440.75it/s] 56%|█████▋    | 225220/400000 [00:30<00:23, 7469.98it/s] 56%|█████▋    | 225968/400000 [00:30<00:23, 7443.76it/s] 57%|█████▋    | 226750/400000 [00:30<00:22, 7552.38it/s] 57%|█████▋    | 227516/400000 [00:30<00:22, 7582.17it/s] 57%|█████▋    | 228305/400000 [00:30<00:22, 7671.92it/s] 57%|█████▋    | 229073/400000 [00:30<00:22, 7653.56it/s] 57%|█████▋    | 229849/400000 [00:30<00:22, 7684.47it/s] 58%|█████▊    | 230618/400000 [00:30<00:22, 7663.49it/s] 58%|█████▊    | 231385/400000 [00:31<00:22, 7533.16it/s] 58%|█████▊    | 232149/400000 [00:31<00:22, 7564.88it/s] 58%|█████▊    | 232906/400000 [00:31<00:22, 7554.81it/s] 58%|█████▊    | 233662/400000 [00:31<00:22, 7525.31it/s] 59%|█████▊    | 234418/400000 [00:31<00:21, 7535.02it/s] 59%|█████▉    | 235172/400000 [00:31<00:22, 7469.59it/s] 59%|█████▉    | 235941/400000 [00:31<00:21, 7533.94it/s] 59%|█████▉    | 236734/400000 [00:31<00:21, 7646.16it/s] 59%|█████▉    | 237500/400000 [00:31<00:21, 7517.89it/s] 60%|█████▉    | 238267/400000 [00:31<00:21, 7561.28it/s] 60%|█████▉    | 239026/400000 [00:32<00:21, 7569.14it/s] 60%|█████▉    | 239786/400000 [00:32<00:21, 7577.01it/s] 60%|██████    | 240569/400000 [00:32<00:20, 7649.06it/s] 60%|██████    | 241335/400000 [00:32<00:20, 7630.63it/s] 61%|██████    | 242118/400000 [00:32<00:20, 7687.08it/s] 61%|██████    | 242888/400000 [00:32<00:20, 7638.79it/s] 61%|██████    | 243656/400000 [00:32<00:20, 7648.48it/s] 61%|██████    | 244422/400000 [00:32<00:20, 7580.89it/s] 61%|██████▏   | 245186/400000 [00:32<00:20, 7594.44it/s] 61%|██████▏   | 245946/400000 [00:33<00:20, 7554.55it/s] 62%|██████▏   | 246702/400000 [00:33<00:20, 7532.46it/s] 62%|██████▏   | 247456/400000 [00:33<00:20, 7510.06it/s] 62%|██████▏   | 248209/400000 [00:33<00:20, 7514.93it/s] 62%|██████▏   | 248966/400000 [00:33<00:20, 7531.29it/s] 62%|██████▏   | 249730/400000 [00:33<00:19, 7561.13it/s] 63%|██████▎   | 250487/400000 [00:33<00:19, 7541.30it/s] 63%|██████▎   | 251274/400000 [00:33<00:19, 7636.17it/s] 63%|██████▎   | 252057/400000 [00:33<00:19, 7693.10it/s] 63%|██████▎   | 252827/400000 [00:33<00:19, 7618.57it/s] 63%|██████▎   | 253590/400000 [00:34<00:19, 7613.64it/s] 64%|██████▎   | 254354/400000 [00:34<00:19, 7620.42it/s] 64%|██████▍   | 255118/400000 [00:34<00:19, 7624.54it/s] 64%|██████▍   | 255902/400000 [00:34<00:18, 7685.50it/s] 64%|██████▍   | 256671/400000 [00:34<00:18, 7666.51it/s] 64%|██████▍   | 257445/400000 [00:34<00:18, 7688.12it/s] 65%|██████▍   | 258214/400000 [00:34<00:18, 7656.32it/s] 65%|██████▍   | 258980/400000 [00:34<00:18, 7596.93it/s] 65%|██████▍   | 259740/400000 [00:34<00:18, 7483.80it/s] 65%|██████▌   | 260508/400000 [00:34<00:18, 7540.11it/s] 65%|██████▌   | 261263/400000 [00:35<00:18, 7495.50it/s] 66%|██████▌   | 262067/400000 [00:35<00:18, 7648.64it/s] 66%|██████▌   | 262856/400000 [00:35<00:17, 7717.67it/s] 66%|██████▌   | 263629/400000 [00:35<00:17, 7642.19it/s] 66%|██████▌   | 264416/400000 [00:35<00:17, 7707.77it/s] 66%|██████▋   | 265188/400000 [00:35<00:17, 7564.87it/s] 66%|██████▋   | 265946/400000 [00:35<00:17, 7556.07it/s] 67%|██████▋   | 266706/400000 [00:35<00:17, 7567.20it/s] 67%|██████▋   | 267493/400000 [00:35<00:17, 7655.46it/s] 67%|██████▋   | 268273/400000 [00:35<00:17, 7696.31it/s] 67%|██████▋   | 269044/400000 [00:36<00:17, 7493.86it/s] 67%|██████▋   | 269820/400000 [00:36<00:17, 7569.52it/s] 68%|██████▊   | 270603/400000 [00:36<00:16, 7643.94it/s] 68%|██████▊   | 271370/400000 [00:36<00:16, 7650.30it/s] 68%|██████▊   | 272136/400000 [00:36<00:16, 7587.22it/s] 68%|██████▊   | 272896/400000 [00:36<00:16, 7514.33it/s] 68%|██████▊   | 273697/400000 [00:36<00:16, 7654.29it/s] 69%|██████▊   | 274464/400000 [00:36<00:16, 7655.94it/s] 69%|██████▉   | 275231/400000 [00:36<00:16, 7646.82it/s] 69%|██████▉   | 276006/400000 [00:36<00:16, 7674.86it/s] 69%|██████▉   | 276783/400000 [00:37<00:16, 7701.03it/s] 69%|██████▉   | 277589/400000 [00:37<00:15, 7805.14it/s] 70%|██████▉   | 278388/400000 [00:37<00:15, 7857.38it/s] 70%|██████▉   | 279175/400000 [00:37<00:15, 7835.79it/s] 70%|██████▉   | 279959/400000 [00:37<00:15, 7835.75it/s] 70%|███████   | 280743/400000 [00:37<00:15, 7699.12it/s] 70%|███████   | 281526/400000 [00:37<00:15, 7737.49it/s] 71%|███████   | 282303/400000 [00:37<00:15, 7746.17it/s] 71%|███████   | 283078/400000 [00:37<00:15, 7741.78it/s] 71%|███████   | 283853/400000 [00:37<00:15, 7696.81it/s] 71%|███████   | 284623/400000 [00:38<00:15, 7595.21it/s] 71%|███████▏  | 285383/400000 [00:38<00:15, 7567.68it/s] 72%|███████▏  | 286168/400000 [00:38<00:14, 7649.51it/s] 72%|███████▏  | 286934/400000 [00:38<00:14, 7633.81it/s] 72%|███████▏  | 287700/400000 [00:38<00:14, 7640.09it/s] 72%|███████▏  | 288465/400000 [00:38<00:14, 7586.74it/s] 72%|███████▏  | 289239/400000 [00:38<00:14, 7629.79it/s] 73%|███████▎  | 290018/400000 [00:38<00:14, 7676.64it/s] 73%|███████▎  | 290786/400000 [00:38<00:14, 7660.21it/s] 73%|███████▎  | 291553/400000 [00:38<00:14, 7614.39it/s] 73%|███████▎  | 292315/400000 [00:39<00:14, 7418.55it/s] 73%|███████▎  | 293078/400000 [00:39<00:14, 7479.40it/s] 73%|███████▎  | 293840/400000 [00:39<00:14, 7518.94it/s] 74%|███████▎  | 294604/400000 [00:39<00:13, 7554.72it/s] 74%|███████▍  | 295375/400000 [00:39<00:13, 7598.76it/s] 74%|███████▍  | 296136/400000 [00:39<00:13, 7527.67it/s] 74%|███████▍  | 296914/400000 [00:39<00:13, 7600.35it/s] 74%|███████▍  | 297675/400000 [00:39<00:13, 7461.25it/s] 75%|███████▍  | 298423/400000 [00:39<00:13, 7448.86it/s] 75%|███████▍  | 299169/400000 [00:39<00:13, 7446.42it/s] 75%|███████▍  | 299929/400000 [00:40<00:13, 7488.90it/s] 75%|███████▌  | 300690/400000 [00:40<00:13, 7522.47it/s] 75%|███████▌  | 301486/400000 [00:40<00:12, 7648.43it/s] 76%|███████▌  | 302252/400000 [00:40<00:12, 7627.39it/s] 76%|███████▌  | 303032/400000 [00:40<00:12, 7675.97it/s] 76%|███████▌  | 303801/400000 [00:40<00:12, 7582.02it/s] 76%|███████▌  | 304577/400000 [00:40<00:12, 7633.84it/s] 76%|███████▋  | 305341/400000 [00:40<00:12, 7599.73it/s] 77%|███████▋  | 306102/400000 [00:40<00:12, 7599.58it/s] 77%|███████▋  | 306863/400000 [00:41<00:12, 7505.31it/s] 77%|███████▋  | 307614/400000 [00:41<00:12, 7471.44it/s] 77%|███████▋  | 308362/400000 [00:41<00:12, 7449.34it/s] 77%|███████▋  | 309128/400000 [00:41<00:12, 7510.15it/s] 77%|███████▋  | 309880/400000 [00:41<00:12, 7504.33it/s] 78%|███████▊  | 310631/400000 [00:41<00:12, 7336.42it/s] 78%|███████▊  | 311368/400000 [00:41<00:12, 7346.26it/s] 78%|███████▊  | 312143/400000 [00:41<00:11, 7461.59it/s] 78%|███████▊  | 312916/400000 [00:41<00:11, 7537.75it/s] 78%|███████▊  | 313694/400000 [00:41<00:11, 7607.68it/s] 79%|███████▊  | 314456/400000 [00:42<00:11, 7577.80it/s] 79%|███████▉  | 315217/400000 [00:42<00:11, 7586.26it/s] 79%|███████▉  | 315982/400000 [00:42<00:11, 7605.27it/s] 79%|███████▉  | 316743/400000 [00:42<00:10, 7591.77it/s] 79%|███████▉  | 317503/400000 [00:42<00:11, 7460.41it/s] 80%|███████▉  | 318250/400000 [00:42<00:11, 7416.31it/s] 80%|███████▉  | 318993/400000 [00:42<00:10, 7406.21it/s] 80%|███████▉  | 319784/400000 [00:42<00:10, 7548.38it/s] 80%|████████  | 320572/400000 [00:42<00:10, 7641.18it/s] 80%|████████  | 321343/400000 [00:42<00:10, 7659.61it/s] 81%|████████  | 322110/400000 [00:43<00:10, 7646.09it/s] 81%|████████  | 322876/400000 [00:43<00:10, 7496.55it/s] 81%|████████  | 323627/400000 [00:43<00:10, 7362.99it/s] 81%|████████  | 324370/400000 [00:43<00:10, 7382.71it/s] 81%|████████▏ | 325131/400000 [00:43<00:10, 7448.11it/s] 81%|████████▏ | 325877/400000 [00:43<00:09, 7430.21it/s] 82%|████████▏ | 326621/400000 [00:43<00:09, 7401.94it/s] 82%|████████▏ | 327362/400000 [00:43<00:09, 7382.98it/s] 82%|████████▏ | 328101/400000 [00:43<00:09, 7327.10it/s] 82%|████████▏ | 328871/400000 [00:43<00:09, 7434.40it/s] 82%|████████▏ | 329619/400000 [00:44<00:09, 7446.57it/s] 83%|████████▎ | 330375/400000 [00:44<00:09, 7478.68it/s] 83%|████████▎ | 331124/400000 [00:44<00:09, 7456.15it/s] 83%|████████▎ | 331880/400000 [00:44<00:09, 7485.16it/s] 83%|████████▎ | 332634/400000 [00:44<00:08, 7499.46it/s] 83%|████████▎ | 333385/400000 [00:44<00:08, 7501.67it/s] 84%|████████▎ | 334136/400000 [00:44<00:08, 7455.67it/s] 84%|████████▎ | 334882/400000 [00:44<00:08, 7416.59it/s] 84%|████████▍ | 335624/400000 [00:44<00:08, 7342.05it/s] 84%|████████▍ | 336359/400000 [00:44<00:08, 7299.72it/s] 84%|████████▍ | 337090/400000 [00:45<00:08, 7245.38it/s] 84%|████████▍ | 337815/400000 [00:45<00:08, 7169.55it/s] 85%|████████▍ | 338533/400000 [00:45<00:08, 7163.05it/s] 85%|████████▍ | 339250/400000 [00:45<00:08, 7158.03it/s] 85%|████████▌ | 340001/400000 [00:45<00:08, 7258.03it/s] 85%|████████▌ | 340736/400000 [00:45<00:08, 7284.44it/s] 85%|████████▌ | 341481/400000 [00:45<00:07, 7331.59it/s] 86%|████████▌ | 342239/400000 [00:45<00:07, 7402.40it/s] 86%|████████▌ | 343024/400000 [00:45<00:07, 7529.94it/s] 86%|████████▌ | 343809/400000 [00:45<00:07, 7621.00it/s] 86%|████████▌ | 344572/400000 [00:46<00:07, 7314.05it/s] 86%|████████▋ | 345335/400000 [00:46<00:07, 7405.46it/s] 87%|████████▋ | 346131/400000 [00:46<00:07, 7563.53it/s] 87%|████████▋ | 346896/400000 [00:46<00:06, 7588.74it/s] 87%|████████▋ | 347657/400000 [00:46<00:07, 7382.25it/s] 87%|████████▋ | 348398/400000 [00:46<00:07, 7331.32it/s] 87%|████████▋ | 349152/400000 [00:46<00:06, 7390.52it/s] 87%|████████▋ | 349898/400000 [00:46<00:06, 7408.45it/s] 88%|████████▊ | 350669/400000 [00:46<00:06, 7494.55it/s] 88%|████████▊ | 351436/400000 [00:46<00:06, 7545.18it/s] 88%|████████▊ | 352192/400000 [00:47<00:06, 7444.55it/s] 88%|████████▊ | 352938/400000 [00:47<00:06, 7391.27it/s] 88%|████████▊ | 353721/400000 [00:47<00:06, 7515.87it/s] 89%|████████▊ | 354515/400000 [00:47<00:05, 7636.87it/s] 89%|████████▉ | 355294/400000 [00:47<00:05, 7681.22it/s] 89%|████████▉ | 356065/400000 [00:47<00:05, 7689.35it/s] 89%|████████▉ | 356835/400000 [00:47<00:05, 7665.11it/s] 89%|████████▉ | 357615/400000 [00:47<00:05, 7705.08it/s] 90%|████████▉ | 358389/400000 [00:47<00:05, 7714.43it/s] 90%|████████▉ | 359161/400000 [00:48<00:05, 7640.90it/s] 90%|████████▉ | 359927/400000 [00:48<00:05, 7645.14it/s] 90%|█████████ | 360692/400000 [00:48<00:05, 7583.16it/s] 90%|█████████ | 361474/400000 [00:48<00:05, 7650.16it/s] 91%|█████████ | 362240/400000 [00:48<00:04, 7610.14it/s] 91%|█████████ | 363014/400000 [00:48<00:04, 7646.76it/s] 91%|█████████ | 363780/400000 [00:48<00:04, 7650.46it/s] 91%|█████████ | 364546/400000 [00:48<00:04, 7629.85it/s] 91%|█████████▏| 365310/400000 [00:48<00:04, 7626.84it/s] 92%|█████████▏| 366093/400000 [00:48<00:04, 7684.76it/s] 92%|█████████▏| 366870/400000 [00:49<00:04, 7708.81it/s] 92%|█████████▏| 367642/400000 [00:49<00:04, 7698.40it/s] 92%|█████████▏| 368412/400000 [00:49<00:04, 7564.79it/s] 92%|█████████▏| 369191/400000 [00:49<00:04, 7628.82it/s] 92%|█████████▏| 369968/400000 [00:49<00:03, 7668.97it/s] 93%|█████████▎| 370736/400000 [00:49<00:03, 7636.70it/s] 93%|█████████▎| 371501/400000 [00:49<00:03, 7592.17it/s] 93%|█████████▎| 372261/400000 [00:49<00:03, 7582.18it/s] 93%|█████████▎| 373020/400000 [00:49<00:03, 7584.49it/s] 93%|█████████▎| 373779/400000 [00:49<00:03, 7492.10it/s] 94%|█████████▎| 374529/400000 [00:50<00:03, 7376.78it/s] 94%|█████████▍| 375304/400000 [00:50<00:03, 7482.62it/s] 94%|█████████▍| 376063/400000 [00:50<00:03, 7513.93it/s] 94%|█████████▍| 376837/400000 [00:50<00:03, 7578.86it/s] 94%|█████████▍| 377596/400000 [00:50<00:02, 7532.18it/s] 95%|█████████▍| 378350/400000 [00:50<00:02, 7475.91it/s] 95%|█████████▍| 379113/400000 [00:50<00:02, 7519.79it/s] 95%|█████████▍| 379868/400000 [00:50<00:02, 7527.57it/s] 95%|█████████▌| 380631/400000 [00:50<00:02, 7556.21it/s] 95%|█████████▌| 381419/400000 [00:50<00:02, 7649.67it/s] 96%|█████████▌| 382193/400000 [00:51<00:02, 7674.99it/s] 96%|█████████▌| 382961/400000 [00:51<00:02, 7530.92it/s] 96%|█████████▌| 383717/400000 [00:51<00:02, 7536.39it/s] 96%|█████████▌| 384487/400000 [00:51<00:02, 7583.76it/s] 96%|█████████▋| 385266/400000 [00:51<00:01, 7643.47it/s] 97%|█████████▋| 386031/400000 [00:51<00:01, 7494.37it/s] 97%|█████████▋| 386782/400000 [00:51<00:01, 7431.94it/s] 97%|█████████▋| 387526/400000 [00:51<00:01, 7394.83it/s] 97%|█████████▋| 388290/400000 [00:51<00:01, 7464.69it/s] 97%|█████████▋| 389038/400000 [00:51<00:01, 7413.09it/s] 97%|█████████▋| 389811/400000 [00:52<00:01, 7502.89it/s] 98%|█████████▊| 390569/400000 [00:52<00:01, 7524.67it/s] 98%|█████████▊| 391322/400000 [00:52<00:01, 7448.53it/s] 98%|█████████▊| 392076/400000 [00:52<00:01, 7473.27it/s] 98%|█████████▊| 392855/400000 [00:52<00:00, 7563.72it/s] 98%|█████████▊| 393622/400000 [00:52<00:00, 7595.00it/s] 99%|█████████▊| 394410/400000 [00:52<00:00, 7677.00it/s] 99%|█████████▉| 395179/400000 [00:52<00:00, 7620.23it/s] 99%|█████████▉| 395942/400000 [00:52<00:00, 7604.47it/s] 99%|█████████▉| 396703/400000 [00:52<00:00, 7600.79it/s] 99%|█████████▉| 397469/400000 [00:53<00:00, 7616.63it/s]100%|█████████▉| 398242/400000 [00:53<00:00, 7650.04it/s]100%|█████████▉| 399008/400000 [00:53<00:00, 7538.35it/s]100%|█████████▉| 399764/400000 [00:53<00:00, 7541.85it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7490.34it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe763174cc0>, <torchtext.data.dataset.TabularDataset object at 0x7fe763174e10>, <torchtext.vocab.Vocab object at 0x7fe763174d30>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.81 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.81 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.81 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.89 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.89 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.15 file/s]2020-07-20 18:18:43.176540: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-20 18:18:43.180886: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-20 18:18:43.181056: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e8d5f98dd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-20 18:18:43.181074: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████▏     | 4104192/9912422 [00:00<00:00, 40694229.79it/s]9920512it [00:00, 34559303.36it/s]                             
0it [00:00, ?it/s]32768it [00:00, 594883.69it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 143466.56it/s]1654784it [00:00, 10450718.68it/s]                         
0it [00:00, ?it/s]8192it [00:00, 135909.70it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
