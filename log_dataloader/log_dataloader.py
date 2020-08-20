
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/13f78dac13e826a41e3a7922ab3568a9b02adef6', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '13f78dac13e826a41e3a7922ab3568a9b02adef6', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/13f78dac13e826a41e3a7922ab3568a9b02adef6

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/13f78dac13e826a41e3a7922ab3568a9b02adef6

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/13f78dac13e826a41e3a7922ab3568a9b02adef6

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7494e0f1e0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7494e0f1e0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f74ffb35400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f74ffb35400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f751ed51ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f751ed51ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f74ace710d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f74ace710d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f74ace710d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▋      | 3604480/9912422 [00:00<00:00, 34279104.57it/s]9920512it [00:00, 33628927.08it/s]                             
0it [00:00, ?it/s]32768it [00:00, 534393.08it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 131694.19it/s]1654784it [00:00, 9740699.37it/s]                          
0it [00:00, ?it/s]8192it [00:00, 171040.98it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7494085b70>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7494092e10>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f74ace69d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f74ace69d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f74ace69d08> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:13,  2.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:13,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:13,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:12,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:12,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:51,  3.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:51,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:51,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:51,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:37,  4.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:00<00:27,  5.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:27,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:27,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:26,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:20,  7.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:20,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:20,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:20,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:15,  9.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:15,  9.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:15,  9.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:15,  9.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:12, 11.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:12, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:12, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:12, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 14.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 14.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:09, 14.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:09, 14.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:01<00:08, 16.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:08, 16.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:08, 16.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:08, 16.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:07, 18.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:07, 18.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:06, 18.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:06, 18.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:06, 20.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:06, 20.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:06, 20.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:06, 20.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:05, 22.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:05, 23.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:05, 23.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:05, 23.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 23.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 24.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 24.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:04, 24.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:04, 24.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:02<00:04, 25.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:04, 25.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:04, 25.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:04, 25.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:02<00:04, 25.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:04, 25.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:04, 25.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:04, 25.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 26.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 26.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:04, 26.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:04, 26.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:02<00:04, 26.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:04, 26.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:04, 26.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:04, 26.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:02<00:03, 26.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:03, 26.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:03, 26.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 26.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 26.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 26.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 26.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 26.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 26.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 26.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 27.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 27.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:03, 27.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:03, 27.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:03<00:03, 27.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:03<00:03, 27.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:03<00:03, 27.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:03<00:03, 27.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:03<00:03, 27.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:03<00:03, 27.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:03<00:03, 27.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:03<00:03, 27.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:03<00:03, 27.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:03<00:03, 27.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:03<00:03, 27.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:03<00:03, 27.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:03<00:03, 27.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:03<00:02, 27.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:03<00:02, 30.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:03<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:03<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:03<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:03<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:02, 34.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:01, 37.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:01, 37.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 37.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:01, 37.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:01, 37.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 37.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 38.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 38.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 38.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 38.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 38.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 38.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 38.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 37.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 37.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 37.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 37.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 37.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:04<00:01, 37.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:04<00:01, 37.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:04<00:01, 37.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:04<00:01, 37.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:04<00:01, 37.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:04<00:01, 37.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:04<00:01, 37.29 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:04<00:01, 37.29 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:04<00:01, 37.29 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:04<00:01, 37.29 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:04<00:01, 37.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:04<00:01, 37.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:04<00:01, 37.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:04<00:01, 37.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:04<00:01, 37.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:04<00:01, 35.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:04<00:01, 35.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:04<00:01, 35.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:04<00:01, 35.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:04<00:01, 35.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:04<00:01, 34.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:04<00:01, 34.07 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:04<00:01, 34.07 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:00, 34.07 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:04<00:00, 34.07 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:04<00:00, 32.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:04<00:00, 32.55 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:04<00:00, 32.55 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:04<00:00, 32.55 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:04<00:00, 32.55 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:04<00:00, 32.55 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:04<00:00, 32.55 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:04<00:00, 36.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:04<00:00, 36.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:04<00:00, 36.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 36.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:04<00:00, 36.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:00, 36.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:00, 36.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 39.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 39.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 39.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 39.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 39.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 39.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 39.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 42.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 42.13 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:05<00:00, 42.13 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:05<00:00, 42.13 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:05<00:00, 42.13 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:05<00:00, 42.13 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:05<00:00, 42.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:05<00:00, 42.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:05<00:00, 42.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:05<00:00, 42.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:05<00:00, 42.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:05<00:00, 42.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:05<00:00, 41.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:05<00:00, 41.67 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:05<00:00, 41.67 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:05<00:00, 41.67 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:05<00:00, 41.67 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 41.67 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.33s/ url]Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 41.67 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 41.67 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:05<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.13s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  5.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 41.67 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.13s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.13s/ file]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 22.72 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.13s/ url]
0 examples [00:00, ? examples/s]2020-08-20 18:06:54.511893: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-20 18:06:54.523787: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2593905000 Hz
2020-08-20 18:06:54.523937: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558d116d5950 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-20 18:06:54.523952: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
62 examples [00:00, 618.36 examples/s]193 examples [00:00, 734.70 examples/s]323 examples [00:00, 843.90 examples/s]453 examples [00:00, 942.49 examples/s]584 examples [00:00, 1028.52 examples/s]714 examples [00:00, 1096.07 examples/s]844 examples [00:00, 1148.87 examples/s]974 examples [00:00, 1190.34 examples/s]1104 examples [00:00, 1220.64 examples/s]1234 examples [00:01, 1241.43 examples/s]1365 examples [00:01, 1258.85 examples/s]1496 examples [00:01, 1271.25 examples/s]1627 examples [00:01, 1282.29 examples/s]1758 examples [00:01, 1287.80 examples/s]1888 examples [00:01, 1288.94 examples/s]2019 examples [00:01, 1293.08 examples/s]2150 examples [00:01, 1297.16 examples/s]2280 examples [00:01, 1273.47 examples/s]2411 examples [00:01, 1282.69 examples/s]2541 examples [00:02, 1287.24 examples/s]2670 examples [00:02, 1286.56 examples/s]2800 examples [00:02, 1290.28 examples/s]2930 examples [00:02, 1292.24 examples/s]3060 examples [00:02, 1275.11 examples/s]3189 examples [00:02, 1277.73 examples/s]3320 examples [00:02, 1284.97 examples/s]3451 examples [00:02, 1290.31 examples/s]3582 examples [00:02, 1294.82 examples/s]3713 examples [00:02, 1296.50 examples/s]3843 examples [00:03, 1296.45 examples/s]3973 examples [00:03, 1296.76 examples/s]4103 examples [00:03, 1293.72 examples/s]4233 examples [00:03, 1293.57 examples/s]4363 examples [00:03, 1289.40 examples/s]4492 examples [00:03, 1282.12 examples/s]4621 examples [00:03, 1275.03 examples/s]4752 examples [00:03, 1283.89 examples/s]4883 examples [00:03, 1289.43 examples/s]5014 examples [00:03, 1293.38 examples/s]5144 examples [00:04, 1294.14 examples/s]5274 examples [00:04, 1295.44 examples/s]5405 examples [00:04, 1299.73 examples/s]5535 examples [00:04, 1273.40 examples/s]5666 examples [00:04, 1281.22 examples/s]5796 examples [00:04, 1286.21 examples/s]5927 examples [00:04, 1291.91 examples/s]6057 examples [00:04, 1292.71 examples/s]6188 examples [00:04, 1295.30 examples/s]6319 examples [00:04, 1298.49 examples/s]6449 examples [00:05, 1296.10 examples/s]6580 examples [00:05, 1299.85 examples/s]6711 examples [00:05, 1300.92 examples/s]6842 examples [00:05, 1302.35 examples/s]6974 examples [00:05, 1304.52 examples/s]7105 examples [00:05, 1299.14 examples/s]7237 examples [00:05, 1303.53 examples/s]7368 examples [00:05, 1303.98 examples/s]7499 examples [00:05, 1303.50 examples/s]7630 examples [00:05, 1302.71 examples/s]7761 examples [00:06, 1280.98 examples/s]7891 examples [00:06, 1283.85 examples/s]8022 examples [00:06, 1290.89 examples/s]8153 examples [00:06, 1293.99 examples/s]8284 examples [00:06, 1296.04 examples/s]8414 examples [00:06, 1292.01 examples/s]8544 examples [00:06, 1292.73 examples/s]8675 examples [00:06, 1295.33 examples/s]8805 examples [00:06, 1295.62 examples/s]8935 examples [00:06, 1295.42 examples/s]9065 examples [00:07, 1293.27 examples/s]9195 examples [00:07, 1295.27 examples/s]9325 examples [00:07, 1295.87 examples/s]9455 examples [00:07, 1295.02 examples/s]9585 examples [00:07, 1276.09 examples/s]9715 examples [00:07, 1280.91 examples/s]9845 examples [00:07, 1285.95 examples/s]9974 examples [00:07, 1286.96 examples/s]10103 examples [00:07, 1215.44 examples/s]10233 examples [00:07, 1237.82 examples/s]10362 examples [00:08, 1250.79 examples/s]10492 examples [00:08, 1262.68 examples/s]10622 examples [00:08, 1272.89 examples/s]10752 examples [00:08, 1279.15 examples/s]10881 examples [00:08, 1282.28 examples/s]11010 examples [00:08, 1283.58 examples/s]11140 examples [00:08, 1286.83 examples/s]11269 examples [00:08, 1283.41 examples/s]11399 examples [00:08, 1286.23 examples/s]11530 examples [00:08, 1292.89 examples/s]11660 examples [00:09, 1294.86 examples/s]11791 examples [00:09, 1297.93 examples/s]11922 examples [00:09, 1299.42 examples/s]12053 examples [00:09, 1301.79 examples/s]12184 examples [00:09, 1301.89 examples/s]12315 examples [00:09, 1298.19 examples/s]12446 examples [00:09, 1299.16 examples/s]12577 examples [00:09, 1300.82 examples/s]12709 examples [00:09, 1304.77 examples/s]12840 examples [00:09, 1304.50 examples/s]12971 examples [00:10, 1304.45 examples/s]13102 examples [00:10, 1301.95 examples/s]13233 examples [00:10, 1303.39 examples/s]13364 examples [00:10, 1304.70 examples/s]13495 examples [00:10, 1305.13 examples/s]13626 examples [00:10, 1303.14 examples/s]13757 examples [00:10, 1302.27 examples/s]13888 examples [00:10, 1300.82 examples/s]14019 examples [00:10, 1302.63 examples/s]14150 examples [00:11, 1303.44 examples/s]14281 examples [00:11, 1301.50 examples/s]14412 examples [00:11, 1301.40 examples/s]14543 examples [00:11, 1302.12 examples/s]14674 examples [00:11, 1302.93 examples/s]14805 examples [00:11, 1304.43 examples/s]14936 examples [00:11, 1302.48 examples/s]15068 examples [00:11, 1305.35 examples/s]15199 examples [00:11, 1306.39 examples/s]15330 examples [00:11, 1306.87 examples/s]15461 examples [00:12, 1307.67 examples/s]15592 examples [00:12, 1282.39 examples/s]15722 examples [00:12, 1287.33 examples/s]15853 examples [00:12, 1293.91 examples/s]15983 examples [00:12, 1293.33 examples/s]16114 examples [00:12, 1296.62 examples/s]16244 examples [00:12, 1293.28 examples/s]16375 examples [00:12, 1297.76 examples/s]16506 examples [00:12, 1299.63 examples/s]16637 examples [00:12, 1301.69 examples/s]16768 examples [00:13, 1303.37 examples/s]16899 examples [00:13, 1298.33 examples/s]17029 examples [00:13, 1296.87 examples/s]17160 examples [00:13, 1299.25 examples/s]17290 examples [00:13, 1298.94 examples/s]17421 examples [00:13, 1301.19 examples/s]17552 examples [00:13, 1301.48 examples/s]17683 examples [00:13, 1303.46 examples/s]17814 examples [00:13, 1303.32 examples/s]17945 examples [00:13, 1304.73 examples/s]18077 examples [00:14, 1306.60 examples/s]18208 examples [00:14, 1302.88 examples/s]18339 examples [00:14, 1299.91 examples/s]18470 examples [00:14, 1301.55 examples/s]18602 examples [00:14, 1304.33 examples/s]18733 examples [00:14, 1303.70 examples/s]18864 examples [00:14, 1300.69 examples/s]18995 examples [00:14, 1300.35 examples/s]19126 examples [00:14, 1299.61 examples/s]19257 examples [00:14, 1300.99 examples/s]19388 examples [00:15, 1300.73 examples/s]19519 examples [00:15, 1303.24 examples/s]19650 examples [00:15, 1300.86 examples/s]19781 examples [00:15, 1302.87 examples/s]19912 examples [00:15, 1304.12 examples/s]20043 examples [00:15, 1245.56 examples/s]20175 examples [00:15, 1264.79 examples/s]20305 examples [00:15, 1274.03 examples/s]20436 examples [00:15, 1283.97 examples/s]20567 examples [00:15, 1289.99 examples/s]20698 examples [00:16, 1294.56 examples/s]20828 examples [00:16, 1295.13 examples/s]20959 examples [00:16, 1297.03 examples/s]21090 examples [00:16, 1298.76 examples/s]21221 examples [00:16, 1299.63 examples/s]21351 examples [00:16, 1299.45 examples/s]21482 examples [00:16, 1300.43 examples/s]21613 examples [00:16, 1298.86 examples/s]21744 examples [00:16, 1299.93 examples/s]21875 examples [00:16, 1301.24 examples/s]22006 examples [00:17, 1303.32 examples/s]22137 examples [00:17, 1302.84 examples/s]22268 examples [00:17, 1301.18 examples/s]22399 examples [00:17, 1302.36 examples/s]22530 examples [00:17, 1286.46 examples/s]22660 examples [00:17, 1289.86 examples/s]22790 examples [00:17, 1291.13 examples/s]22920 examples [00:17, 1288.16 examples/s]23050 examples [00:17, 1290.43 examples/s]23180 examples [00:17, 1290.81 examples/s]23311 examples [00:18, 1295.52 examples/s]23441 examples [00:18, 1276.99 examples/s]23569 examples [00:18, 1276.27 examples/s]23700 examples [00:18, 1283.72 examples/s]23830 examples [00:18, 1287.68 examples/s]23959 examples [00:18, 1265.82 examples/s]24090 examples [00:18, 1276.58 examples/s]24219 examples [00:18, 1279.70 examples/s]24349 examples [00:18, 1285.61 examples/s]24479 examples [00:18, 1288.73 examples/s]24609 examples [00:19, 1289.87 examples/s]24740 examples [00:19, 1293.92 examples/s]24870 examples [00:19, 1291.62 examples/s]25000 examples [00:19, 1292.94 examples/s]25130 examples [00:19, 1291.93 examples/s]25260 examples [00:19, 1293.88 examples/s]25391 examples [00:19, 1297.23 examples/s]25521 examples [00:19, 1294.48 examples/s]25651 examples [00:19, 1275.99 examples/s]25781 examples [00:19, 1281.95 examples/s]25912 examples [00:20, 1287.52 examples/s]26042 examples [00:20, 1290.72 examples/s]26172 examples [00:20, 1289.34 examples/s]26301 examples [00:20, 1288.78 examples/s]26431 examples [00:20, 1291.32 examples/s]26562 examples [00:20, 1294.00 examples/s]26692 examples [00:20, 1292.44 examples/s]26822 examples [00:20, 1291.95 examples/s]26953 examples [00:20, 1294.59 examples/s]27083 examples [00:20, 1293.17 examples/s]27214 examples [00:21, 1295.48 examples/s]27344 examples [00:21, 1295.22 examples/s]27474 examples [00:21, 1294.28 examples/s]27605 examples [00:21, 1297.33 examples/s]27735 examples [00:21, 1289.15 examples/s]27864 examples [00:21, 1279.87 examples/s]27993 examples [00:21, 1274.64 examples/s]28122 examples [00:21, 1278.74 examples/s]28252 examples [00:21, 1283.90 examples/s]28382 examples [00:22, 1286.22 examples/s]28511 examples [00:22, 1275.27 examples/s]28641 examples [00:22, 1281.65 examples/s]28770 examples [00:22, 1281.89 examples/s]28900 examples [00:22, 1286.18 examples/s]29030 examples [00:22, 1290.24 examples/s]29160 examples [00:22, 1276.38 examples/s]29290 examples [00:22, 1283.24 examples/s]29420 examples [00:22, 1286.35 examples/s]29550 examples [00:22, 1289.70 examples/s]29681 examples [00:23, 1293.16 examples/s]29811 examples [00:23, 1293.18 examples/s]29941 examples [00:23, 1291.30 examples/s]30071 examples [00:23, 1232.35 examples/s]30200 examples [00:23, 1248.81 examples/s]30330 examples [00:23, 1262.20 examples/s]30460 examples [00:23, 1271.52 examples/s]30591 examples [00:23, 1280.47 examples/s]30720 examples [00:23, 1279.72 examples/s]30849 examples [00:23, 1282.29 examples/s]30979 examples [00:24, 1287.51 examples/s]31108 examples [00:24, 1274.13 examples/s]31236 examples [00:24, 1263.76 examples/s]31363 examples [00:24, 1254.53 examples/s]31489 examples [00:24, 1233.37 examples/s]31613 examples [00:24, 1234.90 examples/s]31737 examples [00:24, 1226.70 examples/s]31866 examples [00:24, 1243.51 examples/s]31996 examples [00:24, 1257.94 examples/s]32122 examples [00:24, 1258.42 examples/s]32249 examples [00:25, 1261.57 examples/s]32379 examples [00:25, 1271.56 examples/s]32509 examples [00:25, 1278.68 examples/s]32639 examples [00:25, 1282.94 examples/s]32768 examples [00:25, 1274.45 examples/s]32897 examples [00:25, 1278.56 examples/s]33028 examples [00:25, 1285.50 examples/s]33157 examples [00:25, 1282.91 examples/s]33287 examples [00:25, 1287.98 examples/s]33416 examples [00:25, 1285.59 examples/s]33547 examples [00:26, 1292.50 examples/s]33678 examples [00:26, 1296.97 examples/s]33808 examples [00:26, 1292.05 examples/s]33938 examples [00:26, 1293.43 examples/s]34069 examples [00:26, 1297.45 examples/s]34199 examples [00:26, 1292.52 examples/s]34331 examples [00:26, 1297.95 examples/s]34461 examples [00:26, 1295.97 examples/s]34591 examples [00:26, 1268.95 examples/s]34720 examples [00:26, 1273.56 examples/s]34851 examples [00:27, 1282.43 examples/s]34983 examples [00:27, 1291.24 examples/s]35113 examples [00:27, 1289.32 examples/s]35243 examples [00:27, 1290.85 examples/s]35373 examples [00:27, 1291.07 examples/s]35504 examples [00:27, 1295.33 examples/s]35634 examples [00:27, 1295.89 examples/s]35764 examples [00:27, 1294.02 examples/s]35895 examples [00:27, 1296.79 examples/s]36025 examples [00:27, 1293.12 examples/s]36157 examples [00:28, 1299.22 examples/s]36287 examples [00:28, 1296.92 examples/s]36417 examples [00:28, 1286.79 examples/s]36548 examples [00:28, 1292.14 examples/s]36679 examples [00:28, 1296.67 examples/s]36809 examples [00:28, 1296.28 examples/s]36940 examples [00:28, 1299.75 examples/s]37070 examples [00:28, 1295.44 examples/s]37201 examples [00:28, 1298.28 examples/s]37332 examples [00:28, 1300.07 examples/s]37463 examples [00:29, 1301.81 examples/s]37594 examples [00:29, 1302.28 examples/s]37725 examples [00:29, 1299.92 examples/s]37856 examples [00:29, 1302.64 examples/s]37987 examples [00:29, 1302.00 examples/s]38118 examples [00:29, 1300.19 examples/s]38249 examples [00:29, 1300.89 examples/s]38380 examples [00:29, 1295.72 examples/s]38510 examples [00:29, 1296.56 examples/s]38640 examples [00:29, 1297.32 examples/s]38770 examples [00:30, 1296.79 examples/s]38901 examples [00:30, 1298.58 examples/s]39031 examples [00:30, 1272.24 examples/s]39161 examples [00:30, 1279.84 examples/s]39291 examples [00:30, 1284.77 examples/s]39420 examples [00:30, 1262.36 examples/s]39550 examples [00:30, 1272.52 examples/s]39679 examples [00:30, 1276.49 examples/s]39809 examples [00:30, 1282.65 examples/s]39939 examples [00:31, 1286.76 examples/s]40068 examples [00:31, 1205.73 examples/s]40198 examples [00:31, 1230.18 examples/s]40327 examples [00:31, 1246.25 examples/s]40457 examples [00:31, 1260.27 examples/s]40587 examples [00:31, 1269.06 examples/s]40717 examples [00:31, 1276.65 examples/s]40847 examples [00:31, 1283.33 examples/s]40976 examples [00:31, 1276.96 examples/s]41107 examples [00:31, 1284.83 examples/s]41238 examples [00:32, 1290.00 examples/s]41368 examples [00:32, 1289.97 examples/s]41498 examples [00:32, 1282.65 examples/s]41627 examples [00:32, 1280.29 examples/s]41758 examples [00:32, 1287.73 examples/s]41887 examples [00:32, 1282.53 examples/s]42016 examples [00:32, 1257.27 examples/s]42146 examples [00:32, 1267.06 examples/s]42273 examples [00:32, 1262.09 examples/s]42403 examples [00:32, 1272.71 examples/s]42531 examples [00:33, 1272.52 examples/s]42661 examples [00:33, 1277.81 examples/s]42790 examples [00:33, 1281.10 examples/s]42919 examples [00:33, 1277.20 examples/s]43047 examples [00:33, 1271.43 examples/s]43176 examples [00:33, 1276.16 examples/s]43307 examples [00:33, 1283.48 examples/s]43436 examples [00:33, 1282.14 examples/s]43565 examples [00:33, 1276.70 examples/s]43693 examples [00:33, 1274.53 examples/s]43821 examples [00:34, 1267.92 examples/s]43949 examples [00:34, 1269.47 examples/s]44079 examples [00:34, 1276.09 examples/s]44207 examples [00:34, 1269.14 examples/s]44337 examples [00:34, 1276.63 examples/s]44467 examples [00:34, 1282.25 examples/s]44596 examples [00:34, 1276.65 examples/s]44725 examples [00:34, 1277.77 examples/s]44853 examples [00:34, 1276.99 examples/s]44983 examples [00:34, 1281.03 examples/s]45113 examples [00:35, 1284.63 examples/s]45243 examples [00:35, 1287.53 examples/s]45372 examples [00:35, 1287.59 examples/s]45501 examples [00:35, 1267.73 examples/s]45632 examples [00:35, 1277.54 examples/s]45763 examples [00:35, 1284.58 examples/s]45893 examples [00:35, 1288.44 examples/s]46022 examples [00:35, 1287.35 examples/s]46151 examples [00:35, 1286.26 examples/s]46282 examples [00:35, 1292.08 examples/s]46413 examples [00:36, 1294.69 examples/s]46543 examples [00:36, 1294.19 examples/s]46673 examples [00:36, 1261.02 examples/s]46800 examples [00:36, 1259.29 examples/s]46929 examples [00:36, 1266.55 examples/s]47057 examples [00:36, 1269.62 examples/s]47185 examples [00:36, 1271.66 examples/s]47314 examples [00:36, 1275.44 examples/s]47442 examples [00:36, 1273.01 examples/s]47571 examples [00:37, 1277.09 examples/s]47700 examples [00:37, 1280.37 examples/s]47830 examples [00:37, 1283.41 examples/s]47960 examples [00:37, 1286.54 examples/s]48089 examples [00:37, 1284.32 examples/s]48218 examples [00:37, 1285.50 examples/s]48347 examples [00:37, 1285.75 examples/s]48476 examples [00:37, 1284.33 examples/s]48605 examples [00:37, 1285.66 examples/s]48734 examples [00:37, 1284.41 examples/s]48864 examples [00:38, 1286.99 examples/s]48994 examples [00:38, 1289.62 examples/s]49123 examples [00:38, 1285.91 examples/s]49252 examples [00:38, 1280.16 examples/s]49381 examples [00:38, 1281.56 examples/s]49510 examples [00:38, 1283.99 examples/s]49640 examples [00:38, 1287.02 examples/s]49770 examples [00:38, 1290.78 examples/s]49900 examples [00:38, 1289.82 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 17%|█▋        | 8496/50000 [00:00<00:00, 84955.02 examples/s] 46%|████▌     | 23076/50000 [00:00<00:00, 97111.58 examples/s] 75%|███████▌  | 37728/50000 [00:00<00:00, 108039.84 examples/s]                                                                0 examples [00:00, ? examples/s]109 examples [00:00, 1084.35 examples/s]239 examples [00:00, 1140.27 examples/s]370 examples [00:00, 1185.42 examples/s]501 examples [00:00, 1219.45 examples/s]632 examples [00:00, 1245.05 examples/s]763 examples [00:00, 1262.94 examples/s]894 examples [00:00, 1275.19 examples/s]1025 examples [00:00, 1283.22 examples/s]1157 examples [00:00, 1291.32 examples/s]1288 examples [00:01, 1294.27 examples/s]1418 examples [00:01, 1295.48 examples/s]1549 examples [00:01, 1299.32 examples/s]1680 examples [00:01, 1300.19 examples/s]1811 examples [00:01, 1302.53 examples/s]1941 examples [00:01, 1300.38 examples/s]2072 examples [00:01, 1302.50 examples/s]2202 examples [00:01, 1297.90 examples/s]2333 examples [00:01, 1299.89 examples/s]2464 examples [00:01, 1301.09 examples/s]2594 examples [00:02, 1296.03 examples/s]2724 examples [00:02, 1296.27 examples/s]2854 examples [00:02, 1294.57 examples/s]2984 examples [00:02, 1295.54 examples/s]3115 examples [00:02, 1297.34 examples/s]3246 examples [00:02, 1299.30 examples/s]3377 examples [00:02, 1301.66 examples/s]3508 examples [00:02, 1302.19 examples/s]3639 examples [00:02, 1303.97 examples/s]3770 examples [00:02, 1305.39 examples/s]3901 examples [00:03, 1301.21 examples/s]4032 examples [00:03, 1286.25 examples/s]4163 examples [00:03, 1290.74 examples/s]4294 examples [00:03, 1293.58 examples/s]4425 examples [00:03, 1295.56 examples/s]4556 examples [00:03, 1299.08 examples/s]4687 examples [00:03, 1302.17 examples/s]4818 examples [00:03, 1053.06 examples/s]4949 examples [00:03, 1117.38 examples/s]5079 examples [00:04, 1165.93 examples/s]5210 examples [00:04, 1203.46 examples/s]5341 examples [00:04, 1231.24 examples/s]5472 examples [00:04, 1253.58 examples/s]5604 examples [00:04, 1270.33 examples/s]5734 examples [00:04, 1278.79 examples/s]5865 examples [00:04, 1287.92 examples/s]5996 examples [00:04, 1293.95 examples/s]6127 examples [00:04, 1296.83 examples/s]6258 examples [00:04, 1299.87 examples/s]6389 examples [00:05, 1296.69 examples/s]6519 examples [00:05, 1297.49 examples/s]6649 examples [00:05, 1298.16 examples/s]6780 examples [00:05, 1299.95 examples/s]6912 examples [00:05, 1303.12 examples/s]7043 examples [00:05, 1295.65 examples/s]7173 examples [00:05, 1295.36 examples/s]7304 examples [00:05, 1297.50 examples/s]7435 examples [00:05, 1298.80 examples/s]7566 examples [00:05, 1300.11 examples/s]7697 examples [00:06, 1298.37 examples/s]7827 examples [00:06, 1297.93 examples/s]7957 examples [00:06, 1298.47 examples/s]8087 examples [00:06, 1297.95 examples/s]8218 examples [00:06, 1300.38 examples/s]8349 examples [00:06, 1300.29 examples/s]8480 examples [00:06, 1299.80 examples/s]8611 examples [00:06, 1302.72 examples/s]8742 examples [00:06, 1303.48 examples/s]8873 examples [00:06, 1301.63 examples/s]9004 examples [00:07, 1302.54 examples/s]9135 examples [00:07, 1301.66 examples/s]9266 examples [00:07, 1301.33 examples/s]9397 examples [00:07, 1302.03 examples/s]9528 examples [00:07, 1302.57 examples/s]9659 examples [00:07, 1301.47 examples/s]9790 examples [00:07, 1297.19 examples/s]9920 examples [00:07, 1295.81 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete84N4T8/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete84N4T8/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f74ace710d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f74ace710d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f74ace710d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f74f884f828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f744c125908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f74ace710d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f74ace710d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f74ace710d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f751edbb4e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f74940b4080>), {}) 

  




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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range

  




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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 248, in compute
    preprocessor_func = load_function(uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.64 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 24.38 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.36 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.36 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.46 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.46 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.43 file/s]2020-08-20 18:07:48.135151: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-20 18:07:48.138936: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2593905000 Hz
2020-08-20 18:07:48.139169: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563df6ec1d80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-20 18:07:48.139190: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist2', 'train', 'test', 'cifar10', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 133835.56it/s] 69%|██████▉   | 6815744/9912422 [00:00<00:16, 191030.43it/s]9920512it [00:00, 38643592.84it/s]                           
0it [00:00, ?it/s]32768it [00:00, 590173.24it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 133213.94it/s]1654784it [00:00, 9505315.28it/s]                          
0it [00:00, ?it/s]8192it [00:00, 133060.21it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
