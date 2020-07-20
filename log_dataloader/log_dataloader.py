
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f11aff56a60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f11aff56a60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f121abc1510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f121abc1510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1239e10ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1239e10ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f11c7ef4950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f11c7ef4950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f11c7ef4950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3809280/9912422 [00:00<00:00, 37871528.56it/s]9920512it [00:00, 34672477.82it/s]                             
0it [00:00, ?it/s]32768it [00:00, 552685.04it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 461127.17it/s]1654784it [00:00, 11494405.96it/s]                         
0it [00:00, ?it/s]8192it [00:00, 139153.89it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f11c644a5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f11c6465898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f11c7ef4598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f11c7ef4598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f11c7ef4598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:10,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:10,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:09,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:09,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:49,  3.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:49,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:48,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:48,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:48,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:47,  3.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:00<00:34,  4.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:34,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:33,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:33,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:33,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:33,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:33,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:23,  6.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:23,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:23,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:23,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:23,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:23,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:16,  8.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:16,  8.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:16,  8.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:16,  8.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:16,  8.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:12, 11.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:12, 11.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:11, 11.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:11, 11.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:11, 11.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:11, 11.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:08, 14.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:06, 18.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 22.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:05, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:05, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 27.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 27.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 27.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:03, 27.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 27.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 27.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 27.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 31.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 31.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 31.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 31.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 31.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 31.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 31.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 36.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 36.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 36.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 36.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 36.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 36.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 36.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 39.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 39.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 39.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 39.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 39.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 39.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 43.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 43.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 44.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 44.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 44.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 44.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 44.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 48.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 48.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 48.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 48.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 48.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 48.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 48.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 46.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 46.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 46.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 46.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 46.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 46.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 46.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 46.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 50.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 48.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 48.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 48.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 48.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 48.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 48.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 48.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 48.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 51.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 51.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 51.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 51.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 51.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 51.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 51.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 50.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 50.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 50.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 50.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 50.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 50.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 50.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 51.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 51.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 51.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 51.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 51.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 51.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 51.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 51.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 51.45 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 51.45 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 51.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 51.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 52.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 52.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 52.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 52.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 52.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 52.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 52.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 52.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 52.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 52.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 52.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 52.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 52.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 52.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 51.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 51.61 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 51.61 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 51.61 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.57s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.57s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 51.61 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.57s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 51.61 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.57s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 51.61 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.81 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ url]
0 examples [00:00, ? examples/s]2020-07-20 00:08:11.937899: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-20 00:08:11.987644: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-20 00:08:11.987824: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5593fb959db0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-20 00:08:11.987841: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.16 examples/s]119 examples [00:00,  3.09 examples/s]253 examples [00:00,  4.41 examples/s]387 examples [00:00,  6.29 examples/s]516 examples [00:00,  8.97 examples/s]648 examples [00:00, 12.77 examples/s]774 examples [00:01, 18.17 examples/s]908 examples [00:01, 25.81 examples/s]1038 examples [00:01, 36.55 examples/s]1167 examples [00:01, 51.59 examples/s]1292 examples [00:01, 72.42 examples/s]1417 examples [00:01, 100.90 examples/s]1555 examples [00:01, 139.77 examples/s]1694 examples [00:01, 191.38 examples/s]1826 examples [00:01, 255.65 examples/s]1954 examples [00:01, 336.07 examples/s]2081 examples [00:02, 428.67 examples/s]2207 examples [00:02, 534.23 examples/s]2333 examples [00:02, 643.62 examples/s]2458 examples [00:02, 753.13 examples/s]2583 examples [00:02, 842.74 examples/s]2706 examples [00:02, 870.12 examples/s]2820 examples [00:02, 920.50 examples/s]2934 examples [00:02, 975.20 examples/s]3054 examples [00:02, 1032.04 examples/s]3169 examples [00:03, 1054.38 examples/s]3295 examples [00:03, 1106.80 examples/s]3413 examples [00:03, 1125.16 examples/s]3534 examples [00:03, 1149.30 examples/s]3666 examples [00:03, 1194.65 examples/s]3803 examples [00:03, 1239.42 examples/s]3935 examples [00:03, 1260.55 examples/s]4072 examples [00:03, 1291.16 examples/s]4213 examples [00:03, 1322.32 examples/s]4347 examples [00:03, 1323.71 examples/s]4487 examples [00:04, 1344.93 examples/s]4623 examples [00:04, 1298.12 examples/s]4754 examples [00:04, 1275.17 examples/s]4883 examples [00:04, 1270.34 examples/s]5020 examples [00:04, 1297.98 examples/s]5151 examples [00:04, 1284.37 examples/s]5288 examples [00:04, 1308.84 examples/s]5423 examples [00:04, 1320.59 examples/s]5564 examples [00:04, 1345.08 examples/s]5704 examples [00:04, 1361.00 examples/s]5841 examples [00:05, 1344.47 examples/s]5976 examples [00:05, 1322.48 examples/s]6115 examples [00:05, 1341.37 examples/s]6251 examples [00:05, 1344.20 examples/s]6387 examples [00:05, 1347.19 examples/s]6522 examples [00:05, 1324.61 examples/s]6655 examples [00:05, 1309.18 examples/s]6787 examples [00:05, 1259.66 examples/s]6916 examples [00:05, 1263.57 examples/s]7053 examples [00:06, 1293.45 examples/s]7183 examples [00:06, 1281.24 examples/s]7316 examples [00:06, 1293.18 examples/s]7446 examples [00:06, 1293.54 examples/s]7578 examples [00:06, 1297.86 examples/s]7708 examples [00:06, 1272.39 examples/s]7838 examples [00:06, 1277.81 examples/s]7966 examples [00:06, 1273.12 examples/s]8100 examples [00:06, 1292.00 examples/s]8230 examples [00:06, 1276.53 examples/s]8358 examples [00:07, 1266.58 examples/s]8485 examples [00:07, 1225.07 examples/s]8608 examples [00:07, 1215.63 examples/s]8731 examples [00:07, 1217.58 examples/s]8853 examples [00:07, 1200.03 examples/s]8974 examples [00:07, 1197.51 examples/s]9095 examples [00:07, 1200.45 examples/s]9220 examples [00:07, 1212.50 examples/s]9342 examples [00:07, 1208.94 examples/s]9463 examples [00:07, 1200.75 examples/s]9589 examples [00:08, 1215.69 examples/s]9711 examples [00:08, 1172.03 examples/s]9831 examples [00:08, 1177.57 examples/s]9955 examples [00:08, 1194.20 examples/s]10075 examples [00:08, 1115.60 examples/s]10188 examples [00:08, 1119.58 examples/s]10303 examples [00:08, 1113.96 examples/s]10424 examples [00:08, 1139.33 examples/s]10544 examples [00:08, 1155.05 examples/s]10660 examples [00:09, 1153.67 examples/s]10781 examples [00:09, 1168.59 examples/s]10899 examples [00:09, 1167.10 examples/s]11024 examples [00:09, 1188.48 examples/s]11149 examples [00:09, 1204.95 examples/s]11270 examples [00:09, 1202.71 examples/s]11395 examples [00:09, 1215.39 examples/s]11517 examples [00:09, 1204.70 examples/s]11638 examples [00:09, 1205.66 examples/s]11761 examples [00:09, 1212.71 examples/s]11884 examples [00:10, 1217.23 examples/s]12006 examples [00:10, 1162.43 examples/s]12123 examples [00:10, 1155.90 examples/s]12244 examples [00:10, 1170.56 examples/s]12366 examples [00:10, 1182.44 examples/s]12485 examples [00:10, 1174.27 examples/s]12606 examples [00:10, 1182.49 examples/s]12725 examples [00:10, 1160.80 examples/s]12849 examples [00:10, 1183.47 examples/s]12972 examples [00:10, 1194.97 examples/s]13094 examples [00:11, 1201.46 examples/s]13215 examples [00:11, 1193.91 examples/s]13335 examples [00:11, 1165.50 examples/s]13456 examples [00:11, 1177.03 examples/s]13583 examples [00:11, 1200.96 examples/s]13704 examples [00:11, 1196.67 examples/s]13824 examples [00:11, 1194.33 examples/s]13944 examples [00:11, 1175.10 examples/s]14071 examples [00:11, 1200.00 examples/s]14197 examples [00:11, 1215.54 examples/s]14319 examples [00:12, 1170.40 examples/s]14441 examples [00:12, 1184.55 examples/s]14560 examples [00:12, 1150.91 examples/s]14683 examples [00:12, 1173.37 examples/s]14806 examples [00:12, 1187.55 examples/s]14926 examples [00:12, 1171.30 examples/s]15045 examples [00:12, 1176.47 examples/s]15163 examples [00:12, 1164.46 examples/s]15284 examples [00:12, 1175.40 examples/s]15412 examples [00:13, 1202.83 examples/s]15538 examples [00:13, 1218.93 examples/s]15661 examples [00:13, 1194.53 examples/s]15781 examples [00:13, 1186.75 examples/s]15906 examples [00:13, 1203.30 examples/s]16030 examples [00:13, 1212.36 examples/s]16152 examples [00:13, 1204.18 examples/s]16273 examples [00:13, 1198.31 examples/s]16393 examples [00:13, 1189.94 examples/s]16514 examples [00:13, 1194.16 examples/s]16637 examples [00:14, 1202.76 examples/s]16761 examples [00:14, 1211.39 examples/s]16883 examples [00:14, 1196.31 examples/s]17004 examples [00:14, 1197.96 examples/s]17124 examples [00:14, 1196.35 examples/s]17244 examples [00:14, 1193.93 examples/s]17367 examples [00:14, 1203.73 examples/s]17488 examples [00:14, 1191.63 examples/s]17610 examples [00:14, 1198.10 examples/s]17738 examples [00:14, 1219.68 examples/s]17861 examples [00:15, 1221.29 examples/s]17986 examples [00:15, 1228.89 examples/s]18109 examples [00:15, 1217.21 examples/s]18231 examples [00:15, 1214.53 examples/s]18356 examples [00:15, 1223.88 examples/s]18479 examples [00:15, 1200.49 examples/s]18600 examples [00:15, 1143.78 examples/s]18720 examples [00:15, 1159.03 examples/s]18837 examples [00:15, 1153.51 examples/s]18953 examples [00:15, 1152.02 examples/s]19074 examples [00:16, 1167.38 examples/s]19198 examples [00:16, 1188.02 examples/s]19318 examples [00:16, 1126.41 examples/s]19440 examples [00:16, 1150.69 examples/s]19562 examples [00:16, 1167.39 examples/s]19680 examples [00:16, 1162.41 examples/s]19804 examples [00:16, 1182.87 examples/s]19925 examples [00:16, 1190.13 examples/s]20045 examples [00:16, 1131.03 examples/s]20168 examples [00:17, 1157.95 examples/s]20293 examples [00:17, 1183.95 examples/s]20413 examples [00:17, 1187.76 examples/s]20533 examples [00:17, 1182.05 examples/s]20656 examples [00:17, 1194.09 examples/s]20780 examples [00:17, 1205.54 examples/s]20905 examples [00:17, 1218.26 examples/s]21028 examples [00:17, 1212.20 examples/s]21150 examples [00:17, 1185.43 examples/s]21269 examples [00:17, 1168.38 examples/s]21394 examples [00:18, 1191.59 examples/s]21516 examples [00:18, 1199.97 examples/s]21639 examples [00:18, 1208.06 examples/s]21760 examples [00:18, 1197.72 examples/s]21884 examples [00:18, 1208.24 examples/s]22008 examples [00:18, 1216.96 examples/s]22131 examples [00:18, 1220.47 examples/s]22254 examples [00:18, 1220.74 examples/s]22377 examples [00:18, 1199.34 examples/s]22498 examples [00:18, 1201.79 examples/s]22619 examples [00:19, 1198.62 examples/s]22745 examples [00:19, 1214.79 examples/s]22871 examples [00:19, 1226.39 examples/s]22994 examples [00:19, 1225.80 examples/s]23117 examples [00:19, 1222.96 examples/s]23243 examples [00:19, 1232.79 examples/s]23369 examples [00:19, 1240.78 examples/s]23494 examples [00:19, 1219.75 examples/s]23617 examples [00:19, 1218.15 examples/s]23743 examples [00:19, 1228.39 examples/s]23871 examples [00:20, 1240.70 examples/s]23996 examples [00:20, 1233.47 examples/s]24120 examples [00:20, 1230.69 examples/s]24244 examples [00:20, 1225.76 examples/s]24367 examples [00:20, 1204.63 examples/s]24488 examples [00:20, 1200.69 examples/s]24611 examples [00:20, 1206.99 examples/s]24732 examples [00:20, 1189.15 examples/s]24853 examples [00:20, 1193.54 examples/s]24978 examples [00:20, 1207.59 examples/s]25102 examples [00:21, 1217.08 examples/s]25226 examples [00:21, 1223.13 examples/s]25350 examples [00:21, 1227.69 examples/s]25473 examples [00:21, 1222.91 examples/s]25596 examples [00:21, 1220.17 examples/s]25719 examples [00:21, 1219.03 examples/s]25841 examples [00:21, 1197.78 examples/s]25965 examples [00:21, 1208.14 examples/s]26088 examples [00:21, 1213.26 examples/s]26210 examples [00:21, 1187.92 examples/s]26334 examples [00:22, 1201.94 examples/s]26456 examples [00:22, 1206.20 examples/s]26581 examples [00:22, 1217.84 examples/s]26703 examples [00:22, 1211.94 examples/s]26825 examples [00:22, 1195.31 examples/s]26952 examples [00:22, 1215.02 examples/s]27077 examples [00:22, 1224.75 examples/s]27203 examples [00:22, 1233.32 examples/s]27327 examples [00:22, 1232.43 examples/s]27451 examples [00:23, 1223.00 examples/s]27574 examples [00:23, 1225.02 examples/s]27699 examples [00:23, 1231.89 examples/s]27824 examples [00:23, 1236.42 examples/s]27948 examples [00:23, 1232.69 examples/s]28072 examples [00:23, 1206.10 examples/s]28193 examples [00:23, 1203.09 examples/s]28316 examples [00:23, 1208.37 examples/s]28437 examples [00:23, 1202.26 examples/s]28558 examples [00:23, 1181.90 examples/s]28677 examples [00:24, 1164.86 examples/s]28794 examples [00:24, 1155.56 examples/s]28917 examples [00:24, 1175.06 examples/s]29053 examples [00:24, 1222.52 examples/s]29186 examples [00:24, 1251.87 examples/s]29320 examples [00:24, 1274.67 examples/s]29457 examples [00:24, 1300.93 examples/s]29595 examples [00:24, 1321.33 examples/s]29728 examples [00:24, 1314.05 examples/s]29860 examples [00:24, 1307.20 examples/s]29991 examples [00:25, 1304.83 examples/s]30122 examples [00:25, 1230.22 examples/s]30252 examples [00:25, 1249.92 examples/s]30378 examples [00:25, 1251.24 examples/s]30504 examples [00:25, 1243.19 examples/s]30637 examples [00:25, 1265.66 examples/s]30773 examples [00:25, 1291.47 examples/s]30907 examples [00:25, 1302.62 examples/s]31039 examples [00:25, 1307.77 examples/s]31173 examples [00:25, 1314.95 examples/s]31308 examples [00:26, 1323.25 examples/s]31444 examples [00:26, 1333.51 examples/s]31578 examples [00:26, 1311.36 examples/s]31710 examples [00:26, 1307.52 examples/s]31841 examples [00:26, 1267.07 examples/s]31971 examples [00:26, 1275.69 examples/s]32105 examples [00:26, 1293.48 examples/s]32235 examples [00:26, 1283.19 examples/s]32368 examples [00:26, 1294.97 examples/s]32498 examples [00:26, 1296.33 examples/s]32628 examples [00:27, 1290.31 examples/s]32764 examples [00:27, 1308.17 examples/s]32904 examples [00:27, 1332.15 examples/s]33040 examples [00:27, 1339.72 examples/s]33175 examples [00:27, 1316.82 examples/s]33309 examples [00:27, 1323.31 examples/s]33448 examples [00:27, 1340.42 examples/s]33587 examples [00:27, 1352.80 examples/s]33723 examples [00:27, 1275.64 examples/s]33854 examples [00:28, 1284.61 examples/s]33985 examples [00:28, 1290.70 examples/s]34115 examples [00:28, 1287.97 examples/s]34249 examples [00:28, 1300.91 examples/s]34380 examples [00:28, 1268.41 examples/s]34508 examples [00:28, 1260.56 examples/s]34635 examples [00:28, 1258.18 examples/s]34764 examples [00:28, 1266.57 examples/s]34894 examples [00:28, 1273.17 examples/s]35022 examples [00:28, 1245.60 examples/s]35148 examples [00:29, 1248.81 examples/s]35280 examples [00:29, 1269.28 examples/s]35408 examples [00:29, 1253.76 examples/s]35534 examples [00:29, 1249.24 examples/s]35666 examples [00:29, 1269.63 examples/s]35795 examples [00:29, 1274.20 examples/s]35934 examples [00:29, 1304.81 examples/s]36065 examples [00:29, 1305.41 examples/s]36196 examples [00:29, 1281.37 examples/s]36325 examples [00:29, 1271.67 examples/s]36453 examples [00:30, 1238.93 examples/s]36590 examples [00:30, 1275.31 examples/s]36728 examples [00:30, 1302.57 examples/s]36859 examples [00:30, 1303.49 examples/s]36991 examples [00:30, 1306.89 examples/s]37122 examples [00:30, 1297.41 examples/s]37255 examples [00:30, 1303.97 examples/s]37387 examples [00:30, 1306.53 examples/s]37521 examples [00:30, 1314.33 examples/s]37653 examples [00:30, 1314.33 examples/s]37785 examples [00:31, 1311.35 examples/s]37917 examples [00:31, 1291.05 examples/s]38047 examples [00:31, 1258.29 examples/s]38174 examples [00:31, 1258.91 examples/s]38301 examples [00:31, 1234.84 examples/s]38425 examples [00:31, 1177.81 examples/s]38544 examples [00:31, 1175.97 examples/s]38667 examples [00:31, 1191.52 examples/s]38787 examples [00:31, 1129.80 examples/s]38901 examples [00:32, 1086.11 examples/s]39011 examples [00:32, 1088.02 examples/s]39121 examples [00:32, 1055.88 examples/s]39244 examples [00:32, 1102.57 examples/s]39377 examples [00:32, 1162.03 examples/s]39495 examples [00:32, 1160.43 examples/s]39615 examples [00:32, 1169.56 examples/s]39747 examples [00:32, 1209.01 examples/s]39883 examples [00:32, 1248.32 examples/s]40009 examples [00:32, 1192.11 examples/s]40141 examples [00:33, 1227.32 examples/s]40267 examples [00:33, 1234.95 examples/s]40404 examples [00:33, 1271.23 examples/s]40532 examples [00:33, 1262.20 examples/s]40663 examples [00:33, 1273.16 examples/s]40791 examples [00:33, 1274.48 examples/s]40923 examples [00:33, 1286.25 examples/s]41052 examples [00:33, 1276.23 examples/s]41180 examples [00:33, 1265.70 examples/s]41307 examples [00:34, 1257.36 examples/s]41433 examples [00:34, 1241.24 examples/s]41558 examples [00:34, 1219.93 examples/s]41684 examples [00:34, 1229.04 examples/s]41808 examples [00:34, 1223.31 examples/s]41931 examples [00:34, 1169.47 examples/s]42049 examples [00:34, 1162.13 examples/s]42172 examples [00:34, 1179.78 examples/s]42291 examples [00:34, 1176.73 examples/s]42417 examples [00:34, 1198.14 examples/s]42542 examples [00:35, 1210.56 examples/s]42664 examples [00:35, 1205.73 examples/s]42786 examples [00:35, 1208.69 examples/s]42912 examples [00:35, 1222.23 examples/s]43037 examples [00:35, 1230.01 examples/s]43161 examples [00:35, 1225.17 examples/s]43284 examples [00:35, 1223.38 examples/s]43407 examples [00:35, 1217.10 examples/s]43529 examples [00:35, 1200.30 examples/s]43654 examples [00:35, 1214.52 examples/s]43782 examples [00:36, 1231.67 examples/s]43906 examples [00:36, 1221.36 examples/s]44029 examples [00:36, 1205.17 examples/s]44152 examples [00:36, 1212.26 examples/s]44274 examples [00:36, 1197.43 examples/s]44399 examples [00:36, 1211.49 examples/s]44521 examples [00:36, 1209.52 examples/s]44646 examples [00:36, 1219.79 examples/s]44772 examples [00:36, 1229.62 examples/s]44898 examples [00:36, 1237.88 examples/s]45022 examples [00:37, 1235.05 examples/s]45146 examples [00:37, 1194.13 examples/s]45266 examples [00:37, 1195.22 examples/s]45391 examples [00:37, 1208.35 examples/s]45514 examples [00:37, 1212.92 examples/s]45639 examples [00:37, 1222.03 examples/s]45762 examples [00:37, 1213.49 examples/s]45887 examples [00:37, 1221.25 examples/s]46010 examples [00:37, 1219.43 examples/s]46132 examples [00:38, 1213.88 examples/s]46254 examples [00:38, 1204.56 examples/s]46375 examples [00:38, 1204.60 examples/s]46496 examples [00:38, 1187.13 examples/s]46615 examples [00:38, 1157.06 examples/s]46734 examples [00:38, 1164.38 examples/s]46851 examples [00:38, 1157.45 examples/s]46972 examples [00:38, 1171.67 examples/s]47091 examples [00:38, 1174.71 examples/s]47215 examples [00:38, 1192.46 examples/s]47338 examples [00:39, 1202.43 examples/s]47462 examples [00:39, 1212.07 examples/s]47584 examples [00:39, 1205.82 examples/s]47707 examples [00:39, 1211.61 examples/s]47830 examples [00:39, 1214.82 examples/s]47953 examples [00:39, 1218.29 examples/s]48075 examples [00:39, 1213.13 examples/s]48197 examples [00:39, 1197.20 examples/s]48317 examples [00:39, 1191.27 examples/s]48442 examples [00:39, 1206.20 examples/s]48563 examples [00:40, 1207.31 examples/s]48684 examples [00:40, 1181.56 examples/s]48803 examples [00:40, 1162.26 examples/s]48926 examples [00:40, 1180.27 examples/s]49049 examples [00:40, 1192.47 examples/s]49172 examples [00:40, 1202.70 examples/s]49293 examples [00:40, 1191.16 examples/s]49413 examples [00:40, 1171.23 examples/s]49531 examples [00:40, 1172.60 examples/s]49660 examples [00:40, 1204.61 examples/s]49781 examples [00:41, 1193.97 examples/s]49918 examples [00:41, 1239.96 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 9208/50000 [00:00<00:00, 92077.23 examples/s] 51%|█████     | 25461/50000 [00:00<00:00, 105840.32 examples/s] 85%|████████▍ | 42399/50000 [00:00<00:00, 119261.40 examples/s]                                                                0 examples [00:00, ? examples/s]113 examples [00:00, 1121.12 examples/s]250 examples [00:00, 1173.94 examples/s]392 examples [00:00, 1237.92 examples/s]534 examples [00:00, 1285.60 examples/s]672 examples [00:00, 1310.40 examples/s]805 examples [00:00, 1311.69 examples/s]939 examples [00:00, 1319.48 examples/s]1077 examples [00:00, 1334.68 examples/s]1211 examples [00:00, 1335.50 examples/s]1349 examples [00:01, 1348.33 examples/s]1489 examples [00:01, 1361.41 examples/s]1624 examples [00:01, 1354.90 examples/s]1759 examples [00:01, 1346.46 examples/s]1896 examples [00:01, 1353.28 examples/s]2038 examples [00:01, 1372.20 examples/s]2175 examples [00:01, 1345.96 examples/s]2310 examples [00:01, 1331.92 examples/s]2450 examples [00:01, 1350.64 examples/s]2590 examples [00:01, 1364.00 examples/s]2729 examples [00:02, 1369.85 examples/s]2867 examples [00:02, 1350.26 examples/s]3003 examples [00:02, 1345.52 examples/s]3138 examples [00:02, 1312.74 examples/s]3270 examples [00:02, 1298.18 examples/s]3401 examples [00:02, 1267.40 examples/s]3529 examples [00:02, 1249.48 examples/s]3655 examples [00:02, 1248.71 examples/s]3781 examples [00:02, 1248.10 examples/s]3906 examples [00:02, 1241.00 examples/s]4031 examples [00:03, 1221.99 examples/s]4161 examples [00:03, 1242.96 examples/s]4287 examples [00:03, 1245.16 examples/s]4412 examples [00:03, 1244.57 examples/s]4545 examples [00:03, 1268.04 examples/s]4685 examples [00:03, 1303.07 examples/s]4817 examples [00:03, 1306.51 examples/s]4953 examples [00:03, 1321.56 examples/s]5091 examples [00:03, 1337.82 examples/s]5225 examples [00:03, 1298.53 examples/s]5362 examples [00:04, 1317.18 examples/s]5495 examples [00:04, 1316.04 examples/s]5629 examples [00:04, 1319.85 examples/s]5769 examples [00:04, 1340.61 examples/s]5904 examples [00:04, 1326.60 examples/s]6039 examples [00:04, 1330.05 examples/s]6173 examples [00:04, 1279.10 examples/s]6303 examples [00:04, 1285.15 examples/s]6441 examples [00:04, 1309.80 examples/s]6573 examples [00:05, 1295.05 examples/s]6706 examples [00:05, 1304.84 examples/s]6838 examples [00:05, 1309.08 examples/s]6976 examples [00:05, 1329.13 examples/s]7113 examples [00:05, 1340.91 examples/s]7248 examples [00:05, 1321.51 examples/s]7381 examples [00:05, 1318.40 examples/s]7513 examples [00:05, 1308.51 examples/s]7651 examples [00:05, 1327.65 examples/s]7784 examples [00:05, 1319.63 examples/s]7917 examples [00:06, 1301.68 examples/s]8048 examples [00:06, 1298.47 examples/s]8179 examples [00:06, 1299.64 examples/s]8313 examples [00:06, 1310.06 examples/s]8452 examples [00:06, 1331.84 examples/s]8587 examples [00:06, 1335.33 examples/s]8721 examples [00:06, 1323.19 examples/s]8854 examples [00:06, 1306.93 examples/s]8989 examples [00:06, 1316.55 examples/s]9121 examples [00:06, 1224.10 examples/s]9255 examples [00:07, 1255.77 examples/s]9396 examples [00:07, 1296.03 examples/s]9528 examples [00:07, 1299.85 examples/s]9659 examples [00:07, 1295.79 examples/s]9797 examples [00:07, 1318.01 examples/s]9937 examples [00:07, 1339.20 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4CI306/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4CI306/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f11c7ef4950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f11c7ef4950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f11c7ef4950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f123341bf98>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f11862a6400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f11c7ef4950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f11c7ef4950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f11c7ef4950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f11862a6ef0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f11c64654a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f11408eaea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f11408eaea0> 

  function with postional parmater data_info <function split_train_valid at 0x7f11408eaea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f11408e8048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f11408e8048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f11408e8048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=9ea80af45faf7d4ceddab8d4485bb12369e35f93940da1fa5318566a749cc102
  Stored in directory: /tmp/pip-ephem-wheel-cache-weskg898/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:27:32, 13.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:09:09, 18.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:15:37, 25.9kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:29:28, 36.9kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:31:58, 52.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.55M/862M [00:01<3:09:08, 75.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:12:00, 107kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.6M/862M [00:01<1:31:53, 153kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.9M/862M [00:01<1:04:00, 218kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:01<44:45, 311kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.9M/862M [00:01<31:11, 443kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:01<21:54, 629kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.0M/862M [00:02<15:21, 893kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.2M/862M [00:02<10:47, 1.26MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.1M/862M [00:02<07:37, 1.78MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<05:32, 2.44MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<05:46, 2.33MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<05:52, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<04:33, 2.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<05:44, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<05:21, 2.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:06<04:09, 3.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:06<03:04, 4.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<35:26, 375kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<26:12, 507kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:08<18:39, 711kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<16:01, 826kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<12:31, 1.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:10<09:04, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<09:27, 1.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<09:17, 1.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<07:09, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<07:07, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<06:20, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<04:45, 2.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<06:21, 2.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<07:05, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<05:37, 2.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:16<04:03, 3.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<12:34:33, 17.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<8:49:14, 24.5kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:18<6:10:01, 34.9kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<4:21:19, 49.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<3:05:29, 69.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<2:10:23, 98.8kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:20<1:31:04, 141kB/s] .vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<2:48:34, 76.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<1:59:14, 107kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<1:23:37, 153kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<1:01:21, 208kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<45:38, 279kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<32:28, 392kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<22:48, 557kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<25:34, 496kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<18:59, 667kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<13:39, 928kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<09:42, 1.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<38:11, 330kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<27:59, 451kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<19:52, 633kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<16:50, 745kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<14:19, 876kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<10:33, 1.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<07:34, 1.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<09:51, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:10, 1.53MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:58, 2.08MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<07:05, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<07:28, 1.66MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:51, 2.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<06:05, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:32, 2.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:08, 2.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:47, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:43, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:07, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<04:44, 2.58MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:28, 3.51MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<10:30, 1.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<08:36, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:19, 1.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:14, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:31, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<05:53, 2.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:18, 2.80MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<07:16, 1.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:35, 1.83MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<04:59, 2.41MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:53, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:27, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<04:05, 2.92MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:27, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:14, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<04:53, 2.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<03:35, 3.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<07:09, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:14, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<04:37, 2.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:59, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:34, 1.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<05:11, 2.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:31, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:04, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<03:49, 3.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:23, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:07, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:52, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:16, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<04:53, 2.37MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<03:40, 3.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:15, 2.19MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:00, 1.92MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:42, 2.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<03:25, 3.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<09:50, 1.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:02, 1.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<05:54, 1.93MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:48, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:54, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<04:24, 2.57MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<05:45, 1.96MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:19, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:59, 2.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:18, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:06, 1.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:46, 2.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<03:27, 3.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<09:50, 1.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<08:02, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<05:51, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:42, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:57, 1.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:25, 2.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:34, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:52, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<03:39, 3.01MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<02:43, 4.03MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<45:31, 241kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<34:04, 322kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<24:18, 451kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<17:07, 638kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<16:55, 644kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<12:59, 839kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<09:19, 1.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<06:38, 1.63MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<44:53, 241kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<32:30, 333kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<22:58, 470kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<18:34, 579kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<15:10, 709kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<11:09, 963kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<09:30, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<07:44, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:38, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:26, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:39, 1.59MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:07, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:41, 2.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<11:30, 918kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<09:08, 1.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<06:39, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<07:06, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<07:06, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:26, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:54, 2.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<10:46, 968kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<08:35, 1.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:13, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:47, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:52, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<05:14, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:48, 2.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<07:36, 1.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:23, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<04:43, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:41, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<06:05, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:46, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:26, 2.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<9:49:07, 17.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<6:53:08, 24.5kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<4:48:43, 35.0kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<3:23:45, 49.5kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<2:23:33, 70.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<1:40:29, 100kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<1:12:27, 138kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<51:42, 193kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<36:21, 274kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<27:42, 359kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<20:23, 487kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<14:29, 684kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<12:26, 793kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<09:42, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<07:01, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:13, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:02, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:25, 1.81MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:21, 1.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:45, 2.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:33, 2.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:45, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:29, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:23, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<03:10, 3.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<11:19, 848kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<08:57, 1.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<06:28, 1.48MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:37, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<06:46, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:11, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<03:44, 2.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:12, 1.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:05, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:30, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:12, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:44, 1.64MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:32, 2.06MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<03:17, 2.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<13:39, 683kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<10:33, 882kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<07:35, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:20, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:10, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:32, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<03:58, 2.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<13:42, 670kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<10:27, 877kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<07:33, 1.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<05:24, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<13:49, 659kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<10:31, 865kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<07:36, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<07:16, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:05, 1.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:28, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:04, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:32, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:17, 2.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:09, 2.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:01, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:29, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:21, 2.64MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:16, 2.06MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:56, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:56, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<02:51, 3.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<11:22, 770kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<08:54, 983kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<06:27, 1.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<06:25, 1.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<06:24, 1.36MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:57, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<03:34, 2.41MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<12:44, 677kB/s] .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<09:54, 870kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<07:23, 1.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<05:20, 1.61MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:58, 1.43MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:17, 1.61MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:56, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:25, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:03, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:03, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<02:56, 2.87MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<07:19, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<06:02, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:26, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:56, 1.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:17, 1.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:09, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<02:59, 2.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<10:25, 794kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<09:13, 897kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<06:54, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<04:54, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<12:54, 636kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<09:56, 825kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<07:09, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:46, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:32, 1.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:57, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<03:34, 2.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:27, 1.48MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:41, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:29, 2.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:11, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:42, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:39, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:43, 2.92MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:56, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:38, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<02:45, 2.87MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:38, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:24, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<02:35, 3.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:31, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:11, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:20, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:46<02:25, 3.18MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<13:18, 580kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<10:08, 761kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<07:16, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:44, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:32, 1.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:02, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:29, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:49, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:47, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:43, 2.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<10:06, 743kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<07:52, 952kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:40, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:36, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:35, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:14, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<03:03, 2.42MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:37, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:44, 1.56MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:30, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:02, 1.81MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:27, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:31, 2.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<02:33, 2.84MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<10:22, 697kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<08:03, 898kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<05:48, 1.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<05:37, 1.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:41, 1.53MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:25, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:59, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:22, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:27, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:29, 2.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<10:15, 685kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:56, 884kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<05:41, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:31, 1.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:23, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:09, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<02:59, 2.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<10:27, 659kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<08:04, 853kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<05:48, 1.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:32, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:38, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:25, 1.98MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:51, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:26, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:35, 2.59MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<03:16, 2.04MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:01, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:16, 2.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:02, 2.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:34, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:48, 2.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:02, 3.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<04:22, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:45, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:47, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:21, 1.93MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:02, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:17, 2.82MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:59, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:30, 1.82MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:48, 2.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:01, 3.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<07:21, 860kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:49, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:13, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:23, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:26, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:25, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:26, 2.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<08:14, 752kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<06:26, 960kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:38, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:34, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:52, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:50, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:19, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:40, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:51, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:04, 2.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:52, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:21, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:29, 2.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:01, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:22, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:37, 2.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<01:53, 3.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:33, 892kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<05:12, 1.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:45, 1.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:55, 1.47MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:01, 1.43MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:07, 1.84MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:14, 2.55MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<07:41, 742kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<06:00, 949kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:20, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:15, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:13, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:15, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:19, 2.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<07:35, 733kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<05:55, 938kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<04:17, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:11, 1.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:08, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:11, 1.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:17, 2.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<07:25, 731kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<05:46, 938kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:10, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:05, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:02, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:04, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:12, 2.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:56, 1.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:18, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:25, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:53, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:07, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:27, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:45, 2.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<12:20, 418kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<09:11, 560kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<06:31, 786kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<05:39, 899kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<05:04, 1.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:49, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:43, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<07:59, 628kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<06:08, 816kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<04:24, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:09, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:32, 1.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:44, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:59, 2.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:46, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:33, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:55, 2.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:18, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:40, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:08, 2.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<01:32, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:35, 847kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:25, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<03:12, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:15, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:45, 1.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:02, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:28, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:42, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:05, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:30, 3.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:18, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:48, 1.61MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:04, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:25, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:38, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:04, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<01:29, 2.94MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<14:24, 305kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<10:33, 416kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<07:26, 586kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<06:07, 706kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<05:11, 831kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:49, 1.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<02:42, 1.58MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:46, 1.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:06, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:16, 1.85MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:29, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:12, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:37, 2.55MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:11, 3.47MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<25:39, 160kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<18:52, 218kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<13:23, 306kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<09:19, 435kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<10:50, 373kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<08:01, 504kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<05:41, 706kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:49, 826kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:14, 936kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:08, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<02:14, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:11, 1.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:38, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:55, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:13, 1.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:24, 1.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:53, 2.03MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:39<01:21, 2.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<05:31, 682kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<04:16, 880kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<03:04, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:56, 1.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:51, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:11, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:33, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<08:32, 425kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<06:21, 571kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<04:30, 798kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:55, 907kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:32, 1.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:39, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:53, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<05:20, 655kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:07, 846kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:57, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:47, 1.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:40, 1.28MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:01, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:26, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:32, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:04, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:36, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<01:09, 2.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:25, 742kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:26, 951kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:29, 1.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:25, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:01, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:29, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:45, 1.79MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:30, 2.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:06, 2.79MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<00:49, 3.71MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:58, 774kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:27, 889kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:35, 1.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:48, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:33, 659kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:30, 855kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:31, 1.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:23, 1.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:19, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:46, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<01:15, 2.28MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<04:21, 657kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<03:21, 851kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:24, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:16, 1.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:53, 1.48MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:22, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:34, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:42, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:19, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<00:57, 2.80MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:31, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:21, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:59, 2.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:15, 2.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:25, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:07, 2.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<00:48, 3.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<08:16, 305kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<06:03, 415kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<04:15, 585kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<03:28, 706kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:56, 831kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:09, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:31, 1.57MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:05, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:43, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:14, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:22, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:28, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:07, 2.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:49, 2.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:17, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:08, 1.95MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:50, 2.62MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:03, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:12, 1.81MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<00:57, 2.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:40, 3.13MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<04:59, 423kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<03:40, 572kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:35, 800kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:14, 908kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:47, 1.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:17, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:19, 1.49MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:20, 1.47MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:02, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:43, 2.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<03:30, 543kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<02:39, 713kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:53, 991kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:41, 1.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:35, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:12, 1.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:50, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<02:28, 711kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:55, 915kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:22, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:18, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:17, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:58, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:41, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:05, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:56, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:41, 2.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:48, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:54, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:42, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:30, 2.99MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:02, 1.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:51, 1.71MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:38, 2.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:27, 3.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:52, 753kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:37, 869kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:12, 1.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:49, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:51, 723kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:26, 929kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:01, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:58, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:58, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:45, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:33, 2.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:23, 3.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<03:50, 313kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<02:50, 423kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<01:58, 594kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:33, 726kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:13, 918kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:52, 1.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:48, 1.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:49, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:37, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:26, 2.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:59, 1.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:47, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:34, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:35, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:36, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:28, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:19, 2.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:02, 822kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:48, 1.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:34, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:34, 1.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:33, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:25, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:17, 2.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:36, 1.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:30, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:21, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:22, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:20, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:14, 2.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:17, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:15, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:11, 2.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:14, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:16, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:13, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:08, 3.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:34, 770kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:26, 986kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:18, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:16, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:16, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:12, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:07, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:14, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:11, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:07, 2.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:03, 2.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:00, 2.93MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:02, 757kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 965kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.28MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<114:29:31,  1.03s/it]  0%|          | 1015/400000 [00:01<79:56:40,  1.39it/s]  1%|          | 2093/400000 [00:01<55:48:47,  1.98it/s]  1%|          | 3037/400000 [00:01<38:58:47,  2.83it/s]  1%|          | 4080/400000 [00:01<27:13:02,  4.04it/s]  1%|▏         | 5108/400000 [00:01<19:00:21,  5.77it/s]  2%|▏         | 6124/400000 [00:01<13:16:23,  8.24it/s]  2%|▏         | 7179/400000 [00:01<9:16:09, 11.77it/s]   2%|▏         | 8218/400000 [00:01<6:28:28, 16.81it/s]  2%|▏         | 9299/400000 [00:01<4:31:21, 24.00it/s]  3%|▎         | 10373/400000 [00:02<3:09:36, 34.25it/s]  3%|▎         | 11399/400000 [00:02<2:12:34, 48.85it/s]  3%|▎         | 12416/400000 [00:02<1:32:46, 69.63it/s]  3%|▎         | 13430/400000 [00:02<1:04:57, 99.18it/s]  4%|▎         | 14545/400000 [00:02<45:30, 141.15it/s]   4%|▍         | 15624/400000 [00:02<31:56, 200.52it/s]  4%|▍         | 16671/400000 [00:02<22:29, 284.10it/s]  4%|▍         | 17741/400000 [00:02<15:52, 401.29it/s]  5%|▍         | 18793/400000 [00:02<11:15, 563.94it/s]  5%|▍         | 19841/400000 [00:02<08:03, 786.12it/s]  5%|▌         | 20871/400000 [00:03<05:48, 1087.44it/s]  5%|▌         | 21898/400000 [00:03<04:14, 1485.88it/s]  6%|▌         | 22967/400000 [00:03<03:08, 2003.31it/s]  6%|▌         | 24006/400000 [00:03<02:23, 2626.16it/s]  6%|▋         | 25025/400000 [00:03<01:50, 3378.19it/s]  7%|▋         | 26041/400000 [00:03<01:28, 4218.22it/s]  7%|▋         | 27126/400000 [00:03<01:12, 5165.19it/s]  7%|▋         | 28235/400000 [00:03<01:00, 6150.45it/s]  7%|▋         | 29311/400000 [00:03<00:52, 7056.80it/s]  8%|▊         | 30374/400000 [00:03<00:47, 7824.46it/s]  8%|▊         | 31444/400000 [00:04<00:43, 8510.09it/s]  8%|▊         | 32546/400000 [00:04<00:40, 9133.99it/s]  8%|▊         | 33658/400000 [00:04<00:37, 9650.10it/s]  9%|▊         | 34744/400000 [00:04<00:36, 9933.74it/s]  9%|▉         | 35830/400000 [00:04<00:35, 10193.65it/s]  9%|▉         | 36912/400000 [00:04<00:35, 10233.77it/s]  9%|▉         | 37979/400000 [00:04<00:35, 10197.82it/s] 10%|▉         | 39030/400000 [00:04<00:35, 10100.25it/s] 10%|█         | 40062/400000 [00:04<00:35, 10101.19it/s] 10%|█         | 41133/400000 [00:04<00:34, 10274.29it/s] 11%|█         | 42190/400000 [00:05<00:34, 10360.32it/s] 11%|█         | 43235/400000 [00:05<00:35, 10039.16it/s] 11%|█         | 44278/400000 [00:05<00:35, 10150.54it/s] 11%|█▏        | 45299/400000 [00:05<00:35, 10081.80it/s] 12%|█▏        | 46410/400000 [00:05<00:34, 10367.95it/s] 12%|█▏        | 47513/400000 [00:05<00:33, 10557.15it/s] 12%|█▏        | 48648/400000 [00:05<00:32, 10780.64it/s] 12%|█▏        | 49745/400000 [00:05<00:32, 10834.55it/s] 13%|█▎        | 50832/400000 [00:05<00:32, 10735.56it/s] 13%|█▎        | 51919/400000 [00:06<00:32, 10774.52it/s] 13%|█▎        | 52999/400000 [00:06<00:32, 10778.27it/s] 14%|█▎        | 54132/400000 [00:06<00:31, 10935.77it/s] 14%|█▍        | 55227/400000 [00:06<00:32, 10739.68it/s] 14%|█▍        | 56303/400000 [00:06<00:32, 10569.75it/s] 14%|█▍        | 57417/400000 [00:06<00:31, 10734.17it/s] 15%|█▍        | 58514/400000 [00:06<00:31, 10801.03it/s] 15%|█▍        | 59596/400000 [00:06<00:31, 10763.14it/s] 15%|█▌        | 60691/400000 [00:06<00:31, 10816.53it/s] 15%|█▌        | 61774/400000 [00:06<00:31, 10658.11it/s] 16%|█▌        | 62885/400000 [00:07<00:31, 10787.56it/s] 16%|█▌        | 63965/400000 [00:07<00:31, 10737.16it/s] 16%|█▋        | 65040/400000 [00:07<00:31, 10718.23it/s] 17%|█▋        | 66113/400000 [00:07<00:32, 10174.43it/s] 17%|█▋        | 67137/400000 [00:07<00:33, 9962.02it/s]  17%|█▋        | 68240/400000 [00:07<00:32, 10258.91it/s] 17%|█▋        | 69377/400000 [00:07<00:31, 10566.85it/s] 18%|█▊        | 70457/400000 [00:07<00:30, 10632.90it/s] 18%|█▊        | 71525/400000 [00:07<00:31, 10502.31it/s] 18%|█▊        | 72579/400000 [00:07<00:31, 10340.76it/s] 18%|█▊        | 73711/400000 [00:08<00:30, 10615.47it/s] 19%|█▊        | 74839/400000 [00:08<00:30, 10804.02it/s] 19%|█▉        | 75961/400000 [00:08<00:29, 10924.42it/s] 19%|█▉        | 77057/400000 [00:08<00:29, 10784.75it/s] 20%|█▉        | 78138/400000 [00:08<00:30, 10474.18it/s] 20%|█▉        | 79190/400000 [00:08<00:31, 10081.41it/s] 20%|██        | 80204/400000 [00:08<00:31, 10069.07it/s] 20%|██        | 81283/400000 [00:08<00:31, 10275.01it/s] 21%|██        | 82384/400000 [00:08<00:30, 10483.62it/s] 21%|██        | 83437/400000 [00:08<00:30, 10285.12it/s] 21%|██        | 84496/400000 [00:09<00:30, 10373.37it/s] 21%|██▏       | 85552/400000 [00:09<00:30, 10426.57it/s] 22%|██▏       | 86597/400000 [00:09<00:30, 10383.06it/s] 22%|██▏       | 87637/400000 [00:09<00:30, 10103.62it/s] 22%|██▏       | 88650/400000 [00:09<00:31, 9809.47it/s]  22%|██▏       | 89635/400000 [00:09<00:31, 9701.14it/s] 23%|██▎       | 90608/400000 [00:09<00:31, 9701.31it/s] 23%|██▎       | 91600/400000 [00:09<00:31, 9764.71it/s] 23%|██▎       | 92667/400000 [00:09<00:30, 10017.02it/s] 23%|██▎       | 93722/400000 [00:10<00:30, 10168.62it/s] 24%|██▎       | 94775/400000 [00:10<00:29, 10273.27it/s] 24%|██▍       | 95896/400000 [00:10<00:28, 10535.30it/s] 24%|██▍       | 97037/400000 [00:10<00:28, 10781.84it/s] 25%|██▍       | 98137/400000 [00:10<00:27, 10845.34it/s] 25%|██▍       | 99225/400000 [00:10<00:28, 10471.48it/s] 25%|██▌       | 100372/400000 [00:10<00:27, 10751.64it/s] 25%|██▌       | 101453/400000 [00:10<00:27, 10704.10it/s] 26%|██▌       | 102528/400000 [00:10<00:28, 10484.83it/s] 26%|██▌       | 103581/400000 [00:10<00:28, 10408.59it/s] 26%|██▌       | 104633/400000 [00:11<00:28, 10440.04it/s] 26%|██▋       | 105679/400000 [00:11<00:28, 10380.82it/s] 27%|██▋       | 106761/400000 [00:11<00:27, 10506.99it/s] 27%|██▋       | 107843/400000 [00:11<00:27, 10597.04it/s] 27%|██▋       | 108904/400000 [00:11<00:28, 10292.54it/s] 27%|██▋       | 109936/400000 [00:11<00:28, 10224.44it/s] 28%|██▊       | 111056/400000 [00:11<00:27, 10496.51it/s] 28%|██▊       | 112109/400000 [00:11<00:27, 10500.94it/s] 28%|██▊       | 113162/400000 [00:11<00:27, 10488.09it/s] 29%|██▊       | 114213/400000 [00:11<00:27, 10252.37it/s] 29%|██▉       | 115241/400000 [00:12<00:27, 10232.61it/s] 29%|██▉       | 116348/400000 [00:12<00:27, 10470.10it/s] 29%|██▉       | 117398/400000 [00:12<00:27, 10360.90it/s] 30%|██▉       | 118437/400000 [00:12<00:27, 10217.09it/s] 30%|██▉       | 119461/400000 [00:12<00:27, 10068.35it/s] 30%|███       | 120470/400000 [00:12<00:28, 9920.83it/s]  30%|███       | 121464/400000 [00:12<00:28, 9794.00it/s] 31%|███       | 122445/400000 [00:12<00:28, 9610.63it/s] 31%|███       | 123496/400000 [00:12<00:28, 9812.82it/s] 31%|███       | 124509/400000 [00:13<00:27, 9904.05it/s] 31%|███▏      | 125573/400000 [00:13<00:27, 10112.77it/s] 32%|███▏      | 126587/400000 [00:13<00:27, 10070.00it/s] 32%|███▏      | 127665/400000 [00:13<00:26, 10272.09it/s] 32%|███▏      | 128756/400000 [00:13<00:25, 10455.07it/s] 32%|███▏      | 129804/400000 [00:13<00:26, 10213.14it/s] 33%|███▎      | 130910/400000 [00:13<00:25, 10452.69it/s] 33%|███▎      | 132030/400000 [00:13<00:25, 10660.61it/s] 33%|███▎      | 133138/400000 [00:13<00:24, 10781.43it/s] 34%|███▎      | 134247/400000 [00:13<00:24, 10869.85it/s] 34%|███▍      | 135337/400000 [00:14<00:24, 10667.14it/s] 34%|███▍      | 136407/400000 [00:14<00:24, 10655.14it/s] 34%|███▍      | 137475/400000 [00:14<00:24, 10559.02it/s] 35%|███▍      | 138533/400000 [00:14<00:25, 10417.01it/s] 35%|███▍      | 139577/400000 [00:14<00:25, 10183.14it/s] 35%|███▌      | 140605/400000 [00:14<00:25, 10208.87it/s] 35%|███▌      | 141628/400000 [00:14<00:25, 10078.85it/s] 36%|███▌      | 142731/400000 [00:14<00:24, 10344.39it/s] 36%|███▌      | 143828/400000 [00:14<00:24, 10522.81it/s] 36%|███▌      | 144959/400000 [00:14<00:23, 10745.02it/s] 37%|███▋      | 146037/400000 [00:15<00:23, 10661.75it/s] 37%|███▋      | 147106/400000 [00:15<00:24, 10534.80it/s] 37%|███▋      | 148162/400000 [00:15<00:23, 10519.01it/s] 37%|███▋      | 149228/400000 [00:15<00:23, 10558.42it/s] 38%|███▊      | 150285/400000 [00:15<00:24, 10387.38it/s] 38%|███▊      | 151326/400000 [00:15<00:24, 10251.53it/s] 38%|███▊      | 152353/400000 [00:15<00:24, 10243.09it/s] 38%|███▊      | 153393/400000 [00:15<00:23, 10287.37it/s] 39%|███▊      | 154475/400000 [00:15<00:23, 10440.27it/s] 39%|███▉      | 155521/400000 [00:15<00:24, 10153.55it/s] 39%|███▉      | 156539/400000 [00:16<00:24, 10027.42it/s] 39%|███▉      | 157544/400000 [00:16<00:25, 9643.85it/s]  40%|███▉      | 158529/400000 [00:16<00:24, 9703.63it/s] 40%|███▉      | 159651/400000 [00:16<00:23, 10113.28it/s] 40%|████      | 160758/400000 [00:16<00:23, 10382.15it/s] 40%|████      | 161803/400000 [00:16<00:22, 10363.14it/s] 41%|████      | 162929/400000 [00:16<00:22, 10614.74it/s] 41%|████      | 164020/400000 [00:16<00:22, 10699.87it/s] 41%|████▏     | 165094/400000 [00:16<00:21, 10693.34it/s] 42%|████▏     | 166166/400000 [00:16<00:22, 10479.99it/s] 42%|████▏     | 167217/400000 [00:17<00:22, 10240.30it/s] 42%|████▏     | 168354/400000 [00:17<00:21, 10554.52it/s] 42%|████▏     | 169415/400000 [00:17<00:22, 10246.07it/s] 43%|████▎     | 170445/400000 [00:17<00:22, 10234.70it/s] 43%|████▎     | 171473/400000 [00:17<00:23, 9929.75it/s]  43%|████▎     | 172471/400000 [00:17<00:22, 9942.57it/s] 43%|████▎     | 173469/400000 [00:17<00:22, 9926.18it/s] 44%|████▎     | 174464/400000 [00:17<00:22, 9831.70it/s] 44%|████▍     | 175581/400000 [00:17<00:22, 10193.06it/s] 44%|████▍     | 176648/400000 [00:18<00:21, 10330.27it/s] 44%|████▍     | 177702/400000 [00:18<00:21, 10390.92it/s] 45%|████▍     | 178842/400000 [00:18<00:20, 10672.99it/s] 45%|████▍     | 179946/400000 [00:18<00:20, 10777.62it/s] 45%|████▌     | 181027/400000 [00:18<00:20, 10589.05it/s] 46%|████▌     | 182089/400000 [00:18<00:21, 10201.18it/s] 46%|████▌     | 183115/400000 [00:18<00:22, 9687.27it/s]  46%|████▌     | 184153/400000 [00:18<00:21, 9882.13it/s] 46%|████▋     | 185220/400000 [00:18<00:21, 10103.61it/s] 47%|████▋     | 186292/400000 [00:18<00:20, 10279.98it/s] 47%|████▋     | 187347/400000 [00:19<00:20, 10358.66it/s] 47%|████▋     | 188387/400000 [00:19<00:20, 10252.08it/s] 47%|████▋     | 189416/400000 [00:19<00:20, 10217.03it/s] 48%|████▊     | 190495/400000 [00:19<00:20, 10380.64it/s] 48%|████▊     | 191536/400000 [00:19<00:20, 10311.92it/s] 48%|████▊     | 192569/400000 [00:19<00:20, 10062.10it/s] 48%|████▊     | 193709/400000 [00:19<00:19, 10427.99it/s] 49%|████▊     | 194824/400000 [00:19<00:19, 10633.08it/s] 49%|████▉     | 195915/400000 [00:19<00:19, 10714.55it/s] 49%|████▉     | 196990/400000 [00:19<00:19, 10662.57it/s] 50%|████▉     | 198059/400000 [00:20<00:19, 10289.14it/s] 50%|████▉     | 199093/400000 [00:20<00:19, 10159.02it/s] 50%|█████     | 200168/400000 [00:20<00:19, 10328.89it/s] 50%|█████     | 201234/400000 [00:20<00:19, 10425.87it/s] 51%|█████     | 202335/400000 [00:20<00:18, 10593.64it/s] 51%|█████     | 203397/400000 [00:20<00:18, 10368.68it/s] 51%|█████     | 204437/400000 [00:20<00:19, 10233.76it/s] 51%|█████▏    | 205501/400000 [00:20<00:18, 10351.76it/s] 52%|█████▏    | 206630/400000 [00:20<00:18, 10616.02it/s] 52%|█████▏    | 207699/400000 [00:21<00:18, 10637.66it/s] 52%|█████▏    | 208765/400000 [00:21<00:18, 10623.07it/s] 52%|█████▏    | 209885/400000 [00:21<00:17, 10788.75it/s] 53%|█████▎    | 210987/400000 [00:21<00:17, 10856.44it/s] 53%|█████▎    | 212092/400000 [00:21<00:17, 10911.70it/s] 53%|█████▎    | 213190/400000 [00:21<00:17, 10930.85it/s] 54%|█████▎    | 214284/400000 [00:21<00:17, 10606.80it/s] 54%|█████▍    | 215348/400000 [00:21<00:17, 10393.61it/s] 54%|█████▍    | 216457/400000 [00:21<00:17, 10591.34it/s] 54%|█████▍    | 217519/400000 [00:21<00:17, 10358.44it/s] 55%|█████▍    | 218558/400000 [00:22<00:17, 10357.83it/s] 55%|█████▍    | 219596/400000 [00:22<00:17, 10357.35it/s] 55%|█████▌    | 220716/400000 [00:22<00:16, 10595.02it/s] 55%|█████▌    | 221778/400000 [00:22<00:18, 9799.84it/s]  56%|█████▌    | 222772/400000 [00:22<00:18, 9691.62it/s] 56%|█████▌    | 223751/400000 [00:22<00:18, 9689.26it/s] 56%|█████▌    | 224789/400000 [00:22<00:17, 9886.41it/s] 56%|█████▋    | 225899/400000 [00:22<00:17, 10220.19it/s] 57%|█████▋    | 226928/400000 [00:22<00:17, 9875.06it/s]  57%|█████▋    | 227923/400000 [00:23<00:17, 9612.75it/s] 57%|█████▋    | 229016/400000 [00:23<00:17, 9972.66it/s] 58%|█████▊    | 230022/400000 [00:23<00:17, 9979.55it/s] 58%|█████▊    | 231134/400000 [00:23<00:16, 10295.61it/s] 58%|█████▊    | 232211/400000 [00:23<00:16, 10430.90it/s] 58%|█████▊    | 233338/400000 [00:23<00:15, 10667.27it/s] 59%|█████▊    | 234410/400000 [00:23<00:16, 10342.10it/s] 59%|█████▉    | 235450/400000 [00:23<00:16, 10263.96it/s] 59%|█████▉    | 236577/400000 [00:23<00:15, 10545.45it/s] 59%|█████▉    | 237637/400000 [00:23<00:15, 10326.92it/s] 60%|█████▉    | 238675/400000 [00:24<00:15, 10192.95it/s] 60%|█████▉    | 239715/400000 [00:24<00:15, 10252.50it/s] 60%|██████    | 240748/400000 [00:24<00:15, 10274.42it/s] 60%|██████    | 241856/400000 [00:24<00:15, 10502.51it/s] 61%|██████    | 242954/400000 [00:24<00:14, 10640.46it/s] 61%|██████    | 244025/400000 [00:24<00:14, 10659.58it/s] 61%|██████▏   | 245093/400000 [00:24<00:14, 10535.75it/s] 62%|██████▏   | 246148/400000 [00:24<00:14, 10463.94it/s] 62%|██████▏   | 247232/400000 [00:24<00:14, 10573.72it/s] 62%|██████▏   | 248308/400000 [00:24<00:14, 10627.19it/s] 62%|██████▏   | 249372/400000 [00:25<00:14, 10596.79it/s] 63%|██████▎   | 250433/400000 [00:25<00:14, 10577.79it/s] 63%|██████▎   | 251492/400000 [00:25<00:14, 10372.63it/s] 63%|██████▎   | 252619/400000 [00:25<00:13, 10624.06it/s] 63%|██████▎   | 253717/400000 [00:25<00:13, 10726.23it/s] 64%|██████▎   | 254792/400000 [00:25<00:13, 10591.63it/s] 64%|██████▍   | 255853/400000 [00:25<00:13, 10571.99it/s] 64%|██████▍   | 256912/400000 [00:25<00:13, 10552.68it/s] 65%|██████▍   | 258009/400000 [00:25<00:13, 10674.06it/s] 65%|██████▍   | 259078/400000 [00:25<00:13, 10510.60it/s] 65%|██████▌   | 260175/400000 [00:26<00:13, 10641.85it/s] 65%|██████▌   | 261269/400000 [00:26<00:12, 10728.13it/s] 66%|██████▌   | 262343/400000 [00:26<00:12, 10676.53it/s] 66%|██████▌   | 263412/400000 [00:26<00:12, 10510.59it/s] 66%|██████▌   | 264517/400000 [00:26<00:12, 10666.65it/s] 66%|██████▋   | 265598/400000 [00:26<00:12, 10707.70it/s] 67%|██████▋   | 266696/400000 [00:26<00:12, 10787.26it/s] 67%|██████▋   | 267776/400000 [00:26<00:12, 10420.07it/s] 67%|██████▋   | 268822/400000 [00:26<00:12, 10229.25it/s] 67%|██████▋   | 269953/400000 [00:26<00:12, 10528.95it/s] 68%|██████▊   | 271060/400000 [00:27<00:12, 10684.08it/s] 68%|██████▊   | 272133/400000 [00:27<00:12, 10359.03it/s] 68%|██████▊   | 273174/400000 [00:27<00:12, 10286.67it/s] 69%|██████▊   | 274207/400000 [00:27<00:12, 10258.37it/s] 69%|██████▉   | 275236/400000 [00:27<00:12, 10225.96it/s] 69%|██████▉   | 276261/400000 [00:27<00:12, 10069.83it/s] 69%|██████▉   | 277279/400000 [00:27<00:12, 10101.25it/s] 70%|██████▉   | 278373/400000 [00:27<00:11, 10336.58it/s] 70%|██████▉   | 279426/400000 [00:27<00:11, 10392.36it/s] 70%|███████   | 280467/400000 [00:28<00:11, 10213.58it/s] 70%|███████   | 281491/400000 [00:28<00:11, 9990.90it/s]  71%|███████   | 282493/400000 [00:28<00:11, 9819.06it/s] 71%|███████   | 283605/400000 [00:28<00:11, 10174.10it/s] 71%|███████   | 284694/400000 [00:28<00:11, 10376.72it/s] 71%|███████▏  | 285737/400000 [00:28<00:11, 10177.31it/s] 72%|███████▏  | 286873/400000 [00:28<00:10, 10504.22it/s] 72%|███████▏  | 287932/400000 [00:28<00:10, 10527.65it/s] 72%|███████▏  | 288989/400000 [00:28<00:10, 10426.39it/s] 73%|███████▎  | 290099/400000 [00:28<00:10, 10617.94it/s] 73%|███████▎  | 291197/400000 [00:29<00:10, 10723.98it/s] 73%|███████▎  | 292272/400000 [00:29<00:10, 10624.46it/s] 73%|███████▎  | 293337/400000 [00:29<00:10, 10486.91it/s] 74%|███████▎  | 294420/400000 [00:29<00:09, 10585.64it/s] 74%|███████▍  | 295535/400000 [00:29<00:09, 10748.85it/s] 74%|███████▍  | 296612/400000 [00:29<00:09, 10568.73it/s] 74%|███████▍  | 297671/400000 [00:29<00:09, 10308.95it/s] 75%|███████▍  | 298735/400000 [00:29<00:09, 10404.78it/s] 75%|███████▍  | 299844/400000 [00:29<00:09, 10599.52it/s] 75%|███████▌  | 300982/400000 [00:29<00:09, 10821.00it/s] 76%|███████▌  | 302084/400000 [00:30<00:09, 10877.10it/s] 76%|███████▌  | 303175/400000 [00:30<00:08, 10884.38it/s] 76%|███████▌  | 304265/400000 [00:30<00:08, 10741.89it/s] 76%|███████▋  | 305341/400000 [00:30<00:08, 10733.54it/s] 77%|███████▋  | 306416/400000 [00:30<00:08, 10533.00it/s] 77%|███████▋  | 307511/400000 [00:30<00:08, 10654.55it/s] 77%|███████▋  | 308634/400000 [00:30<00:08, 10820.54it/s] 77%|███████▋  | 309718/400000 [00:30<00:08, 10632.57it/s] 78%|███████▊  | 310784/400000 [00:30<00:08, 10537.65it/s] 78%|███████▊  | 311850/400000 [00:30<00:08, 10572.23it/s] 78%|███████▊  | 312930/400000 [00:31<00:08, 10637.08it/s] 79%|███████▊  | 314032/400000 [00:31<00:07, 10746.81it/s] 79%|███████▉  | 315108/400000 [00:31<00:08, 10536.71it/s] 79%|███████▉  | 316164/400000 [00:31<00:08, 10089.88it/s] 79%|███████▉  | 317191/400000 [00:31<00:08, 10142.93it/s] 80%|███████▉  | 318282/400000 [00:31<00:07, 10359.36it/s] 80%|███████▉  | 319392/400000 [00:31<00:07, 10570.69it/s] 80%|████████  | 320453/400000 [00:31<00:07, 10502.11it/s] 80%|████████  | 321585/400000 [00:31<00:07, 10733.95it/s] 81%|████████  | 322662/400000 [00:32<00:07, 10589.92it/s] 81%|████████  | 323724/400000 [00:32<00:07, 10333.72it/s] 81%|████████  | 324761/400000 [00:32<00:07, 10219.93it/s] 81%|████████▏ | 325786/400000 [00:32<00:07, 10033.20it/s] 82%|████████▏ | 326904/400000 [00:32<00:07, 10351.69it/s] 82%|████████▏ | 327964/400000 [00:32<00:06, 10423.42it/s] 82%|████████▏ | 329097/400000 [00:32<00:06, 10679.77it/s] 83%|████████▎ | 330169/400000 [00:32<00:06, 10634.54it/s] 83%|████████▎ | 331291/400000 [00:32<00:06, 10802.42it/s] 83%|████████▎ | 332412/400000 [00:32<00:06, 10920.14it/s] 83%|████████▎ | 333526/400000 [00:33<00:06, 10982.69it/s] 84%|████████▎ | 334640/400000 [00:33<00:05, 11028.37it/s] 84%|████████▍ | 335744/400000 [00:33<00:05, 10926.76it/s] 84%|████████▍ | 336838/400000 [00:33<00:05, 10654.82it/s] 84%|████████▍ | 337906/400000 [00:33<00:05, 10537.33it/s] 85%|████████▍ | 338980/400000 [00:33<00:05, 10594.75it/s] 85%|████████▌ | 340056/400000 [00:33<00:05, 10642.16it/s] 85%|████████▌ | 341122/400000 [00:33<00:05, 10627.03it/s] 86%|████████▌ | 342249/400000 [00:33<00:05, 10812.00it/s] 86%|████████▌ | 343332/400000 [00:33<00:05, 10803.80it/s] 86%|████████▌ | 344414/400000 [00:34<00:05, 10785.74it/s] 86%|████████▋ | 345546/400000 [00:34<00:04, 10940.53it/s] 87%|████████▋ | 346648/400000 [00:34<00:04, 10961.85it/s] 87%|████████▋ | 347762/400000 [00:34<00:04, 11012.06it/s] 87%|████████▋ | 348907/400000 [00:34<00:04, 11139.58it/s] 88%|████████▊ | 350022/400000 [00:34<00:04, 11046.34it/s] 88%|████████▊ | 351128/400000 [00:34<00:04, 10892.48it/s] 88%|████████▊ | 352219/400000 [00:34<00:04, 10892.39it/s] 88%|████████▊ | 353309/400000 [00:34<00:04, 10843.70it/s] 89%|████████▊ | 354466/400000 [00:34<00:04, 11051.65it/s] 89%|████████▉ | 355605/400000 [00:35<00:03, 11149.76it/s] 89%|████████▉ | 356722/400000 [00:35<00:03, 11022.54it/s] 89%|████████▉ | 357826/400000 [00:35<00:03, 10682.89it/s] 90%|████████▉ | 358898/400000 [00:35<00:03, 10643.24it/s] 90%|█████████ | 360019/400000 [00:35<00:03, 10805.82it/s] 90%|█████████ | 361144/400000 [00:35<00:03, 10910.48it/s] 91%|█████████ | 362237/400000 [00:35<00:03, 10768.51it/s] 91%|█████████ | 363331/400000 [00:35<00:03, 10818.69it/s] 91%|█████████ | 364424/400000 [00:35<00:03, 10851.47it/s] 91%|█████████▏| 365549/400000 [00:35<00:03, 10967.30it/s] 92%|█████████▏| 366668/400000 [00:36<00:03, 11033.13it/s] 92%|█████████▏| 367773/400000 [00:36<00:02, 10935.40it/s] 92%|█████████▏| 368868/400000 [00:36<00:02, 10931.97it/s] 92%|█████████▏| 369982/400000 [00:36<00:02, 10993.28it/s] 93%|█████████▎| 371082/400000 [00:36<00:02, 10924.44it/s] 93%|█████████▎| 372217/400000 [00:36<00:02, 11048.04it/s] 93%|█████████▎| 373352/400000 [00:36<00:02, 11133.46it/s] 94%|█████████▎| 374466/400000 [00:36<00:02, 10929.24it/s] 94%|█████████▍| 375561/400000 [00:36<00:02, 10751.96it/s] 94%|█████████▍| 376672/400000 [00:36<00:02, 10855.70it/s] 94%|█████████▍| 377810/400000 [00:37<00:02, 11005.15it/s] 95%|█████████▍| 378912/400000 [00:37<00:01, 10836.69it/s] 95%|█████████▍| 379998/400000 [00:37<00:01, 10307.15it/s] 95%|█████████▌| 381036/400000 [00:37<00:01, 10231.58it/s] 96%|█████████▌| 382064/400000 [00:37<00:01, 10202.31it/s] 96%|█████████▌| 383088/400000 [00:37<00:01, 10179.58it/s] 96%|█████████▌| 384207/400000 [00:37<00:01, 10462.37it/s] 96%|█████████▋| 385305/400000 [00:37<00:01, 10610.05it/s] 97%|█████████▋| 386370/400000 [00:37<00:01, 10455.70it/s] 97%|█████████▋| 387498/400000 [00:38<00:01, 10690.02it/s] 97%|█████████▋| 388589/400000 [00:38<00:01, 10753.21it/s] 97%|█████████▋| 389709/400000 [00:38<00:00, 10882.92it/s] 98%|█████████▊| 390805/400000 [00:38<00:00, 10902.00it/s] 98%|█████████▊| 391897/400000 [00:38<00:00, 10518.37it/s] 98%|█████████▊| 392953/400000 [00:38<00:00, 10105.59it/s] 98%|█████████▊| 393970/400000 [00:38<00:00, 9937.15it/s]  99%|█████████▉| 395029/400000 [00:38<00:00, 10123.99it/s] 99%|█████████▉| 396049/400000 [00:38<00:00, 10145.79it/s] 99%|█████████▉| 397075/400000 [00:38<00:00, 10179.41it/s]100%|█████████▉| 398166/400000 [00:39<00:00, 10387.53it/s]100%|█████████▉| 399244/400000 [00:39<00:00, 10500.28it/s]100%|█████████▉| 399999/400000 [00:39<00:00, 10195.14it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1140c0fcc0>, <torchtext.data.dataset.TabularDataset object at 0x7f1140c0fe10>, <torchtext.vocab.Vocab object at 0x7f1140c0fd30>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.37 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.37 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.71 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.71 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.89 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.89 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.41 file/s]2020-07-20 00:16:58.289297: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-20 00:16:58.292884: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-20 00:16:58.293380: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55da7ce4fac0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-20 00:16:58.293393: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 153220.00it/s] 77%|███████▋  | 7659520/9912422 [00:00<00:10, 218688.80it/s]9920512it [00:00, 42716294.20it/s]                           
0it [00:00, ?it/s]32768it [00:00, 588460.01it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 452157.35it/s]1654784it [00:00, 10595645.75it/s]                         
0it [00:00, ?it/s]8192it [00:00, 153586.41it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
