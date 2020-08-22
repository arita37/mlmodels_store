
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f6d84cf9268> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f6d84cf9268> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f6defa1f400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f6defa1f400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f6e0ec3bea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f6e0ec3bea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6d9cd5b0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6d9cd5b0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6d9cd5b0d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 134176.06it/s] 79%|███████▉  | 7856128/9912422 [00:00<00:10, 191539.59it/s]9920512it [00:00, 41528869.51it/s]                           
0it [00:00, ?it/s]32768it [00:00, 571087.06it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1003875.92it/s]1654784it [00:00, 12404791.92it/s]                           
0it [00:00, ?it/s]8192it [00:00, 229774.16it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6d9adbb320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6d83f7edd8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f6d9cd53d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f6d9cd53d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f6d9cd53d08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:53,  1.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:53,  1.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:53,  1.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:52,  1.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:51,  1.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:50,  1.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:50,  1.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:49,  1.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:48,  1.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:16,  2.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:16,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:15,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:15,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:14,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:14,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:13,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:13,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:12,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:12,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<01:11,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<01:11,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:49,  2.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:49,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:49,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:49,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:48,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:48,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:48,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:47,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:47,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:47,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:46,  2.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:32,  4.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:32,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:32,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:32,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:32,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:31,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:31,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:31,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:31,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:30,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:30,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:30,  4.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:21,  5.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:21,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:21,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:21,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:20,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:20,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:20,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:20,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:20,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:20,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:19,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:19,  5.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:13,  7.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:13,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:13,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:13,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:13,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:13,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:13,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:13,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:13,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:12,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:12,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:12,  7.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:09, 10.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:09, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:08, 10.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 14.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:05, 14.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 19.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:03, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 26.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:02, 26.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 34.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 34.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:01, 43.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:01, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:01, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 43.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 52.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 61.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 61.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.17s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 71.39 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.17s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.18s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.78 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.18s/ url]
0 examples [00:00, ? examples/s]2020-08-22 18:06:55.719143: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-22 18:06:55.734068: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-22 18:06:55.734394: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5622b99df0a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-22 18:06:55.734412: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
47 examples [00:00, 467.94 examples/s]152 examples [00:00, 560.96 examples/s]255 examples [00:00, 649.03 examples/s]348 examples [00:00, 712.52 examples/s]454 examples [00:00, 789.26 examples/s]559 examples [00:00, 851.86 examples/s]667 examples [00:00, 907.45 examples/s]772 examples [00:00, 942.05 examples/s]874 examples [00:00, 961.49 examples/s]972 examples [00:01, 959.86 examples/s]1078 examples [00:01, 986.00 examples/s]1184 examples [00:01, 1004.87 examples/s]1286 examples [00:01, 999.62 examples/s] 1387 examples [00:01, 995.94 examples/s]1487 examples [00:01, 994.01 examples/s]1587 examples [00:01, 985.88 examples/s]1686 examples [00:01, 983.14 examples/s]1785 examples [00:01, 971.99 examples/s]1883 examples [00:01, 967.94 examples/s]1987 examples [00:02, 986.77 examples/s]2096 examples [00:02, 1013.09 examples/s]2205 examples [00:02, 1034.23 examples/s]2312 examples [00:02, 1042.32 examples/s]2417 examples [00:02, 1041.05 examples/s]2522 examples [00:02, 1002.58 examples/s]2623 examples [00:02, 986.74 examples/s] 2723 examples [00:02, 978.42 examples/s]2822 examples [00:02, 968.24 examples/s]2920 examples [00:02, 911.71 examples/s]3012 examples [00:03, 908.14 examples/s]3107 examples [00:03, 918.92 examples/s]3209 examples [00:03, 944.59 examples/s]3304 examples [00:03, 890.03 examples/s]3404 examples [00:03, 919.45 examples/s]3505 examples [00:03, 944.65 examples/s]3608 examples [00:03, 967.60 examples/s]3712 examples [00:03, 987.48 examples/s]3812 examples [00:03, 987.31 examples/s]3920 examples [00:04, 1011.43 examples/s]4022 examples [00:04, 1006.53 examples/s]4123 examples [00:04, 991.01 examples/s] 4228 examples [00:04, 1006.21 examples/s]4329 examples [00:04, 985.02 examples/s] 4428 examples [00:04, 947.44 examples/s]4524 examples [00:04, 921.81 examples/s]4627 examples [00:04, 951.06 examples/s]4731 examples [00:04, 975.87 examples/s]4831 examples [00:04, 982.15 examples/s]4932 examples [00:05, 989.75 examples/s]5037 examples [00:05, 1005.44 examples/s]5147 examples [00:05, 1029.52 examples/s]5252 examples [00:05, 1033.87 examples/s]5356 examples [00:05, 1016.85 examples/s]5466 examples [00:05, 1040.26 examples/s]5571 examples [00:05, 1041.17 examples/s]5676 examples [00:05, 1040.98 examples/s]5787 examples [00:05, 1058.50 examples/s]5894 examples [00:05, 1042.53 examples/s]6000 examples [00:06, 1047.37 examples/s]6105 examples [00:06, 1044.83 examples/s]6216 examples [00:06, 1063.27 examples/s]6323 examples [00:06, 1044.67 examples/s]6428 examples [00:06, 990.21 examples/s] 6530 examples [00:06, 998.95 examples/s]6635 examples [00:06, 1013.48 examples/s]6740 examples [00:06, 1022.79 examples/s]6843 examples [00:06, 1017.35 examples/s]6950 examples [00:06, 1031.76 examples/s]7056 examples [00:07, 1037.70 examples/s]7160 examples [00:07, 1034.47 examples/s]7268 examples [00:07, 1047.62 examples/s]7373 examples [00:07, 1045.01 examples/s]7478 examples [00:07, 1030.34 examples/s]7582 examples [00:07, 1016.99 examples/s]7688 examples [00:07, 1028.21 examples/s]7791 examples [00:07, 1023.30 examples/s]7894 examples [00:07, 997.94 examples/s] 7994 examples [00:08, 992.93 examples/s]8101 examples [00:08, 1013.23 examples/s]8205 examples [00:08, 1019.27 examples/s]8309 examples [00:08, 1023.84 examples/s]8412 examples [00:08, 1013.10 examples/s]8514 examples [00:08, 996.63 examples/s] 8617 examples [00:08, 1002.10 examples/s]8718 examples [00:08, 994.22 examples/s] 8819 examples [00:08, 998.17 examples/s]8919 examples [00:08, 990.62 examples/s]9026 examples [00:09, 1011.44 examples/s]9138 examples [00:09, 1039.57 examples/s]9246 examples [00:09, 1048.78 examples/s]9355 examples [00:09, 1060.23 examples/s]9462 examples [00:09, 1040.10 examples/s]9569 examples [00:09, 1046.48 examples/s]9675 examples [00:09, 1049.78 examples/s]9783 examples [00:09, 1058.29 examples/s]9889 examples [00:09, 1053.54 examples/s]9995 examples [00:09, 1046.70 examples/s]10100 examples [00:10, 981.89 examples/s]10206 examples [00:10, 1001.98 examples/s]10307 examples [00:10, 993.42 examples/s] 10407 examples [00:10, 992.51 examples/s]10507 examples [00:10, 986.27 examples/s]10611 examples [00:10, 1001.10 examples/s]10717 examples [00:10, 1016.56 examples/s]10819 examples [00:10, 995.34 examples/s] 10919 examples [00:10, 995.08 examples/s]11019 examples [00:11, 978.27 examples/s]11118 examples [00:11, 981.41 examples/s]11220 examples [00:11, 990.62 examples/s]11324 examples [00:11, 1004.79 examples/s]11428 examples [00:11, 1014.25 examples/s]11530 examples [00:11, 1010.08 examples/s]11634 examples [00:11, 1018.72 examples/s]11742 examples [00:11, 1034.53 examples/s]11849 examples [00:11, 1043.61 examples/s]11954 examples [00:11, 1019.77 examples/s]12059 examples [00:12, 1026.72 examples/s]12165 examples [00:12, 1034.98 examples/s]12276 examples [00:12, 1054.67 examples/s]12386 examples [00:12, 1066.78 examples/s]12493 examples [00:12, 1060.07 examples/s]12600 examples [00:12, 1052.75 examples/s]12706 examples [00:12, 1046.58 examples/s]12813 examples [00:12, 1052.80 examples/s]12919 examples [00:12, 1051.55 examples/s]13028 examples [00:12, 1060.17 examples/s]13135 examples [00:13, 1049.56 examples/s]13241 examples [00:13, 1043.51 examples/s]13349 examples [00:13, 1051.98 examples/s]13455 examples [00:13, 1051.60 examples/s]13561 examples [00:13, 1052.72 examples/s]13667 examples [00:13, 1025.61 examples/s]13770 examples [00:13, 1026.72 examples/s]13876 examples [00:13, 1035.84 examples/s]13980 examples [00:13, 1034.89 examples/s]14086 examples [00:13, 1041.77 examples/s]14191 examples [00:14, 1040.09 examples/s]14296 examples [00:14, 1036.93 examples/s]14400 examples [00:14, 1037.59 examples/s]14508 examples [00:14, 1045.94 examples/s]14616 examples [00:14, 1054.65 examples/s]14723 examples [00:14, 1055.86 examples/s]14829 examples [00:14, 1045.15 examples/s]14938 examples [00:14, 1057.38 examples/s]15047 examples [00:14, 1065.44 examples/s]15160 examples [00:14, 1083.25 examples/s]15269 examples [00:15, 1073.45 examples/s]15377 examples [00:15, 1050.70 examples/s]15483 examples [00:15, 1053.25 examples/s]15589 examples [00:15, 1047.97 examples/s]15694 examples [00:15, 1045.39 examples/s]15799 examples [00:15, 1026.19 examples/s]15904 examples [00:15, 1030.37 examples/s]16008 examples [00:15, 1024.98 examples/s]16111 examples [00:15, 1024.78 examples/s]16214 examples [00:15, 1017.27 examples/s]16316 examples [00:16, 998.40 examples/s] 16422 examples [00:16, 1013.74 examples/s]16524 examples [00:16, 1001.43 examples/s]16627 examples [00:16, 1007.89 examples/s]16730 examples [00:16, 1012.25 examples/s]16832 examples [00:16, 992.76 examples/s] 16936 examples [00:16, 1005.64 examples/s]17037 examples [00:16, 1002.53 examples/s]17139 examples [00:16, 1006.05 examples/s]17244 examples [00:17, 1017.25 examples/s]17346 examples [00:17, 1013.85 examples/s]17448 examples [00:17, 1014.42 examples/s]17550 examples [00:17, 1014.45 examples/s]17652 examples [00:17, 1015.74 examples/s]17754 examples [00:17, 984.91 examples/s] 17853 examples [00:17, 985.22 examples/s]17952 examples [00:17, 983.36 examples/s]18057 examples [00:17, 1002.40 examples/s]18163 examples [00:17, 1018.30 examples/s]18269 examples [00:18, 1029.95 examples/s]18373 examples [00:18, 1010.25 examples/s]18477 examples [00:18, 1016.68 examples/s]18582 examples [00:18, 1026.16 examples/s]18685 examples [00:18, 1027.02 examples/s]18790 examples [00:18, 1032.07 examples/s]18894 examples [00:18, 1021.95 examples/s]18998 examples [00:18, 1024.90 examples/s]19103 examples [00:18, 1030.49 examples/s]19210 examples [00:18, 1041.43 examples/s]19315 examples [00:19, 1043.17 examples/s]19420 examples [00:19, 1031.97 examples/s]19524 examples [00:19, 1017.60 examples/s]19627 examples [00:19, 1021.09 examples/s]19730 examples [00:19, 1012.06 examples/s]19832 examples [00:19, 1014.42 examples/s]19934 examples [00:19, 982.36 examples/s] 20033 examples [00:19, 933.50 examples/s]20136 examples [00:19, 960.22 examples/s]20239 examples [00:19, 978.35 examples/s]20342 examples [00:20, 992.03 examples/s]20442 examples [00:20, 980.63 examples/s]20543 examples [00:20, 987.78 examples/s]20645 examples [00:20, 995.17 examples/s]20746 examples [00:20, 996.99 examples/s]20849 examples [00:20, 1006.19 examples/s]20950 examples [00:20, 990.50 examples/s] 21050 examples [00:20, 989.68 examples/s]21150 examples [00:20, 967.37 examples/s]21249 examples [00:21, 971.50 examples/s]21347 examples [00:21, 961.38 examples/s]21444 examples [00:21, 943.09 examples/s]21541 examples [00:21, 950.11 examples/s]21640 examples [00:21, 959.75 examples/s]21739 examples [00:21, 967.95 examples/s]21839 examples [00:21, 976.10 examples/s]21937 examples [00:21, 969.01 examples/s]22037 examples [00:21, 976.58 examples/s]22136 examples [00:21, 979.12 examples/s]22238 examples [00:22, 988.87 examples/s]22343 examples [00:22, 1005.87 examples/s]22444 examples [00:22, 979.59 examples/s] 22544 examples [00:22, 983.65 examples/s]22645 examples [00:22, 988.96 examples/s]22745 examples [00:22, 989.69 examples/s]22846 examples [00:22, 993.36 examples/s]22946 examples [00:22, 976.18 examples/s]23044 examples [00:22, 977.25 examples/s]23143 examples [00:22, 980.08 examples/s]23244 examples [00:23, 986.51 examples/s]23343 examples [00:23, 985.18 examples/s]23442 examples [00:23, 974.45 examples/s]23541 examples [00:23, 978.22 examples/s]23640 examples [00:23, 979.36 examples/s]23738 examples [00:23, 965.24 examples/s]23837 examples [00:23, 972.29 examples/s]23935 examples [00:23, 938.82 examples/s]24030 examples [00:23, 921.57 examples/s]24127 examples [00:23, 933.68 examples/s]24227 examples [00:24, 951.90 examples/s]24327 examples [00:24, 965.60 examples/s]24437 examples [00:24, 999.77 examples/s]24538 examples [00:24, 1001.23 examples/s]24639 examples [00:24, 993.86 examples/s] 24739 examples [00:24, 987.50 examples/s]24838 examples [00:24, 965.42 examples/s]24936 examples [00:24, 968.19 examples/s]25040 examples [00:24, 988.36 examples/s]25143 examples [00:25, 998.03 examples/s]25244 examples [00:25, 1000.33 examples/s]25345 examples [00:25, 1001.70 examples/s]25446 examples [00:25, 987.44 examples/s] 25545 examples [00:25, 977.28 examples/s]25644 examples [00:25, 979.49 examples/s]25743 examples [00:25, 976.39 examples/s]25841 examples [00:25, 963.85 examples/s]25946 examples [00:25, 986.92 examples/s]26047 examples [00:25, 993.54 examples/s]26147 examples [00:26, 994.03 examples/s]26247 examples [00:26, 970.27 examples/s]26345 examples [00:26, 963.82 examples/s]26442 examples [00:26, 960.72 examples/s]26540 examples [00:26, 964.23 examples/s]26639 examples [00:26, 971.54 examples/s]26742 examples [00:26, 987.10 examples/s]26841 examples [00:26, 972.58 examples/s]26941 examples [00:26, 979.30 examples/s]27040 examples [00:26, 976.19 examples/s]27139 examples [00:27, 977.60 examples/s]27237 examples [00:27, 977.40 examples/s]27335 examples [00:27, 959.18 examples/s]27433 examples [00:27, 964.53 examples/s]27530 examples [00:27, 963.04 examples/s]27628 examples [00:27, 966.20 examples/s]27728 examples [00:27, 974.96 examples/s]27826 examples [00:27, 965.87 examples/s]27923 examples [00:27, 966.50 examples/s]28022 examples [00:27, 972.90 examples/s]28123 examples [00:28, 983.07 examples/s]28222 examples [00:28, 982.46 examples/s]28321 examples [00:28, 974.11 examples/s]28419 examples [00:28, 975.49 examples/s]28519 examples [00:28, 980.38 examples/s]28620 examples [00:28, 988.27 examples/s]28719 examples [00:28, 987.81 examples/s]28818 examples [00:28, 972.24 examples/s]28918 examples [00:28, 980.18 examples/s]29018 examples [00:28, 985.25 examples/s]29120 examples [00:29, 993.07 examples/s]29221 examples [00:29, 996.87 examples/s]29321 examples [00:29, 976.21 examples/s]29422 examples [00:29, 983.50 examples/s]29523 examples [00:29, 989.87 examples/s]29623 examples [00:29, 984.03 examples/s]29722 examples [00:29, 969.27 examples/s]29822 examples [00:29, 976.78 examples/s]29920 examples [00:29, 946.48 examples/s]30015 examples [00:30, 911.80 examples/s]30122 examples [00:30, 953.70 examples/s]30223 examples [00:30, 969.13 examples/s]30321 examples [00:30, 958.17 examples/s]30422 examples [00:30, 970.42 examples/s]30522 examples [00:30, 977.39 examples/s]30624 examples [00:30, 988.43 examples/s]30729 examples [00:30, 1005.66 examples/s]30830 examples [00:30, 998.77 examples/s] 30931 examples [00:30, 1000.35 examples/s]31032 examples [00:31, 994.48 examples/s] 31132 examples [00:31, 992.50 examples/s]31233 examples [00:31, 995.58 examples/s]31333 examples [00:31, 982.93 examples/s]31434 examples [00:31, 988.28 examples/s]31533 examples [00:31, 977.67 examples/s]31631 examples [00:31, 976.70 examples/s]31730 examples [00:31, 978.40 examples/s]31828 examples [00:31, 962.58 examples/s]31928 examples [00:31, 972.55 examples/s]32032 examples [00:32, 990.25 examples/s]32136 examples [00:32, 1002.20 examples/s]32240 examples [00:32, 1012.06 examples/s]32342 examples [00:32, 1001.15 examples/s]32443 examples [00:32, 986.53 examples/s] 32544 examples [00:32, 992.85 examples/s]32646 examples [00:32, 999.77 examples/s]32747 examples [00:32, 1002.26 examples/s]32848 examples [00:32, 986.74 examples/s] 32955 examples [00:32, 1007.97 examples/s]33060 examples [00:33, 1019.12 examples/s]33163 examples [00:33, 1016.86 examples/s]33265 examples [00:33, 1017.32 examples/s]33367 examples [00:33, 1002.77 examples/s]33468 examples [00:33, 1001.67 examples/s]33570 examples [00:33, 1006.62 examples/s]33676 examples [00:33, 1021.72 examples/s]33779 examples [00:33, 1007.01 examples/s]33880 examples [00:33, 999.19 examples/s] 33981 examples [00:33, 995.37 examples/s]34083 examples [00:34, 1001.87 examples/s]34184 examples [00:34, 1001.97 examples/s]34288 examples [00:34, 1011.54 examples/s]34390 examples [00:34, 994.59 examples/s] 34494 examples [00:34, 1005.67 examples/s]34600 examples [00:34, 1019.56 examples/s]34705 examples [00:34, 1026.30 examples/s]34810 examples [00:34, 1030.86 examples/s]34914 examples [00:34, 1018.54 examples/s]35018 examples [00:35, 1024.05 examples/s]35121 examples [00:35, 1023.13 examples/s]35224 examples [00:35, 1018.83 examples/s]35329 examples [00:35, 1027.23 examples/s]35432 examples [00:35, 1005.38 examples/s]35533 examples [00:35, 993.32 examples/s] 35633 examples [00:35, 984.37 examples/s]35732 examples [00:35, 971.90 examples/s]35832 examples [00:35, 979.10 examples/s]35930 examples [00:35, 960.58 examples/s]36028 examples [00:36, 964.01 examples/s]36129 examples [00:36, 975.10 examples/s]36227 examples [00:36, 967.23 examples/s]36325 examples [00:36, 967.78 examples/s]36422 examples [00:36, 952.42 examples/s]36518 examples [00:36, 951.82 examples/s]36622 examples [00:36, 974.55 examples/s]36721 examples [00:36, 976.68 examples/s]36819 examples [00:36, 946.06 examples/s]36914 examples [00:36, 935.79 examples/s]37014 examples [00:37, 952.37 examples/s]37112 examples [00:37, 959.38 examples/s]37213 examples [00:37, 972.51 examples/s]37314 examples [00:37, 980.76 examples/s]37413 examples [00:37, 965.49 examples/s]37513 examples [00:37, 973.78 examples/s]37613 examples [00:37, 978.69 examples/s]37712 examples [00:37, 980.41 examples/s]37814 examples [00:37, 989.37 examples/s]37914 examples [00:37, 973.46 examples/s]38018 examples [00:38, 990.54 examples/s]38118 examples [00:38, 990.71 examples/s]38221 examples [00:38, 1001.31 examples/s]38327 examples [00:38, 1015.41 examples/s]38429 examples [00:38, 1009.34 examples/s]38534 examples [00:38, 1019.92 examples/s]38640 examples [00:38, 1029.64 examples/s]38747 examples [00:38, 1038.88 examples/s]38851 examples [00:38, 1037.99 examples/s]38955 examples [00:38, 1017.29 examples/s]39061 examples [00:39, 1026.75 examples/s]39164 examples [00:39, 1006.35 examples/s]39265 examples [00:39, 983.24 examples/s] 39367 examples [00:39, 992.19 examples/s]39467 examples [00:39, 982.99 examples/s]39571 examples [00:39, 998.27 examples/s]39671 examples [00:39, 995.56 examples/s]39771 examples [00:39, 996.74 examples/s]39871 examples [00:39, 992.16 examples/s]39971 examples [00:40, 975.76 examples/s]40069 examples [00:40, 926.88 examples/s]40174 examples [00:40, 952.56 examples/s]40276 examples [00:40, 970.32 examples/s]40382 examples [00:40, 994.01 examples/s]40482 examples [00:40, 992.60 examples/s]40585 examples [00:40, 1002.62 examples/s]40686 examples [00:40, 1003.79 examples/s]40787 examples [00:40, 999.11 examples/s] 40889 examples [00:40, 1003.87 examples/s]40990 examples [00:41, 987.19 examples/s] 41096 examples [00:41, 1006.46 examples/s]41201 examples [00:41, 1016.02 examples/s]41303 examples [00:41, 1001.81 examples/s]41405 examples [00:41, 1005.22 examples/s]41506 examples [00:41, 983.30 examples/s] 41613 examples [00:41, 1006.95 examples/s]41722 examples [00:41, 1029.56 examples/s]41826 examples [00:41, 1030.16 examples/s]41930 examples [00:41, 1003.47 examples/s]42031 examples [00:42, 972.61 examples/s] 42134 examples [00:42, 985.96 examples/s]42235 examples [00:42, 992.27 examples/s]42335 examples [00:42, 981.93 examples/s]42434 examples [00:42, 965.79 examples/s]42537 examples [00:42, 984.01 examples/s]42639 examples [00:42, 994.19 examples/s]42743 examples [00:42, 1005.57 examples/s]42847 examples [00:42, 1014.31 examples/s]42949 examples [00:43, 1014.72 examples/s]43051 examples [00:43, 993.06 examples/s] 43151 examples [00:43, 988.74 examples/s]43254 examples [00:43, 999.90 examples/s]43357 examples [00:43, 1007.06 examples/s]43458 examples [00:43, 1000.64 examples/s]43559 examples [00:43, 995.87 examples/s] 43661 examples [00:43, 1001.00 examples/s]43764 examples [00:43, 1009.20 examples/s]43867 examples [00:43, 1014.27 examples/s]43969 examples [00:44, 997.91 examples/s] 44075 examples [00:44, 1013.64 examples/s]44180 examples [00:44, 1021.55 examples/s]44287 examples [00:44, 1034.96 examples/s]44393 examples [00:44, 1039.59 examples/s]44498 examples [00:44, 1039.30 examples/s]44602 examples [00:44, 1019.74 examples/s]44707 examples [00:44, 1026.71 examples/s]44812 examples [00:44, 1032.16 examples/s]44918 examples [00:44, 1038.39 examples/s]45022 examples [00:45, 1027.95 examples/s]45125 examples [00:45, 997.19 examples/s] 45229 examples [00:45, 1008.96 examples/s]45334 examples [00:45, 1018.80 examples/s]45437 examples [00:45, 1019.29 examples/s]45540 examples [00:45, 1014.34 examples/s]45642 examples [00:45, 1014.17 examples/s]45745 examples [00:45, 1018.11 examples/s]45850 examples [00:45, 1027.43 examples/s]45953 examples [00:45, 1012.17 examples/s]46055 examples [00:46, 1006.00 examples/s]46157 examples [00:46, 1008.36 examples/s]46258 examples [00:46, 982.35 examples/s] 46359 examples [00:46, 989.45 examples/s]46460 examples [00:46, 994.89 examples/s]46560 examples [00:46, 995.70 examples/s]46660 examples [00:46, 992.12 examples/s]46762 examples [00:46, 1000.02 examples/s]46863 examples [00:46, 998.35 examples/s] 46963 examples [00:46, 995.88 examples/s]47063 examples [00:47, 993.78 examples/s]47163 examples [00:47, 973.65 examples/s]47264 examples [00:47, 983.22 examples/s]47368 examples [00:47, 997.23 examples/s]47468 examples [00:47, 961.69 examples/s]47565 examples [00:47, 961.83 examples/s]47662 examples [00:47, 958.88 examples/s]47759 examples [00:47, 959.06 examples/s]47864 examples [00:47, 982.60 examples/s]47970 examples [00:48, 1003.60 examples/s]48073 examples [00:48, 1009.26 examples/s]48179 examples [00:48, 1022.09 examples/s]48282 examples [00:48, 1023.35 examples/s]48389 examples [00:48, 1034.99 examples/s]48493 examples [00:48, 1034.80 examples/s]48597 examples [00:48, 1019.81 examples/s]48700 examples [00:48, 1020.68 examples/s]48809 examples [00:48, 1038.46 examples/s]48913 examples [00:48, 1038.68 examples/s]49019 examples [00:49, 1043.92 examples/s]49124 examples [00:49, 1017.14 examples/s]49226 examples [00:49, 1017.29 examples/s]49334 examples [00:49, 1033.54 examples/s]49441 examples [00:49, 1044.13 examples/s]49546 examples [00:49, 1037.02 examples/s]49650 examples [00:49, 1029.84 examples/s]49754 examples [00:49, 1023.03 examples/s]49858 examples [00:49, 1027.90 examples/s]49961 examples [00:49, 1025.74 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 8123/50000 [00:00<00:00, 81228.72 examples/s] 41%|████      | 20302/50000 [00:00<00:00, 90244.46 examples/s] 66%|██████▋   | 33149/50000 [00:00<00:00, 99089.49 examples/s] 92%|█████████▏| 46012/50000 [00:00<00:00, 106418.94 examples/s]                                                                0 examples [00:00, ? examples/s]82 examples [00:00, 818.84 examples/s]186 examples [00:00, 873.61 examples/s]281 examples [00:00, 894.07 examples/s]387 examples [00:00, 937.20 examples/s]496 examples [00:00, 977.94 examples/s]602 examples [00:00, 999.02 examples/s]700 examples [00:00, 992.16 examples/s]795 examples [00:00, 975.70 examples/s]897 examples [00:00, 986.85 examples/s]993 examples [00:01, 963.29 examples/s]1088 examples [00:01, 952.21 examples/s]1189 examples [00:01, 968.53 examples/s]1286 examples [00:01, 962.02 examples/s]1390 examples [00:01, 983.28 examples/s]1492 examples [00:01, 992.95 examples/s]1596 examples [00:01, 1006.46 examples/s]1700 examples [00:01, 1013.75 examples/s]1802 examples [00:01, 1014.55 examples/s]1907 examples [00:01, 1022.62 examples/s]2013 examples [00:02, 1030.55 examples/s]2118 examples [00:02, 1034.29 examples/s]2222 examples [00:02, 1024.49 examples/s]2325 examples [00:02, 995.42 examples/s] 2425 examples [00:02, 992.46 examples/s]2525 examples [00:02, 991.22 examples/s]2625 examples [00:02, 993.63 examples/s]2725 examples [00:02, 992.80 examples/s]2825 examples [00:02, 977.03 examples/s]2926 examples [00:02, 985.33 examples/s]3028 examples [00:03, 994.32 examples/s]3130 examples [00:03, 999.49 examples/s]3231 examples [00:03, 1000.17 examples/s]3332 examples [00:03, 981.69 examples/s] 3431 examples [00:03, 965.15 examples/s]3533 examples [00:03, 980.58 examples/s]3632 examples [00:03, 972.49 examples/s]3734 examples [00:03, 985.41 examples/s]3833 examples [00:03, 977.52 examples/s]3941 examples [00:03, 1003.94 examples/s]4050 examples [00:04, 1027.83 examples/s]4157 examples [00:04, 1037.75 examples/s]4262 examples [00:04, 1031.22 examples/s]4366 examples [00:04, 1005.20 examples/s]4467 examples [00:04, 1005.38 examples/s]4572 examples [00:04, 1015.80 examples/s]4675 examples [00:04, 1018.54 examples/s]4777 examples [00:04, 829.85 examples/s] 4881 examples [00:04, 883.17 examples/s]4982 examples [00:05, 917.54 examples/s]5088 examples [00:05, 955.22 examples/s]5195 examples [00:05, 984.55 examples/s]5296 examples [00:05, 987.56 examples/s]5399 examples [00:05, 998.67 examples/s]5502 examples [00:05, 1007.68 examples/s]5606 examples [00:05, 1016.13 examples/s]5713 examples [00:05, 1030.50 examples/s]5817 examples [00:05, 1009.52 examples/s]5920 examples [00:05, 1013.31 examples/s]6022 examples [00:06, 1006.34 examples/s]6125 examples [00:06, 1012.03 examples/s]6228 examples [00:06, 1015.13 examples/s]6330 examples [00:06, 996.09 examples/s] 6431 examples [00:06, 997.52 examples/s]6531 examples [00:06, 997.22 examples/s]6631 examples [00:06, 997.13 examples/s]6732 examples [00:06, 997.32 examples/s]6832 examples [00:06, 981.66 examples/s]6931 examples [00:06, 975.13 examples/s]7029 examples [00:07, 970.95 examples/s]7130 examples [00:07, 981.12 examples/s]7232 examples [00:07, 990.00 examples/s]7332 examples [00:07, 979.79 examples/s]7434 examples [00:07, 989.88 examples/s]7539 examples [00:07, 1005.26 examples/s]7642 examples [00:07, 1012.38 examples/s]7744 examples [00:07, 1004.73 examples/s]7845 examples [00:07, 989.42 examples/s] 7946 examples [00:08, 994.38 examples/s]8046 examples [00:08, 991.94 examples/s]8146 examples [00:08, 984.91 examples/s]8247 examples [00:08, 990.46 examples/s]8347 examples [00:08, 982.29 examples/s]8448 examples [00:08, 990.13 examples/s]8550 examples [00:08, 996.77 examples/s]8655 examples [00:08, 1010.66 examples/s]8761 examples [00:08, 1024.20 examples/s]8864 examples [00:08, 1009.90 examples/s]8968 examples [00:09, 1018.32 examples/s]9073 examples [00:09, 1025.09 examples/s]9176 examples [00:09, 1019.46 examples/s]9279 examples [00:09, 1007.97 examples/s]9380 examples [00:09, 986.91 examples/s] 9486 examples [00:09, 1006.36 examples/s]9587 examples [00:09, 1005.06 examples/s]9688 examples [00:09, 1004.54 examples/s]9790 examples [00:09, 1007.19 examples/s]9891 examples [00:09, 966.47 examples/s] 9989 examples [00:10, 961.35 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9OBUHH/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9OBUHH/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6d9cd5b0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6d9cd5b0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6d9cd5b0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6de87397b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6e08247f98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6d9cd5b0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6d9cd5b0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6d9cd5b0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6d38501cf8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6d83f9f048>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.74 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 26.56 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.87 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.87 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.32 file/s]2020-08-22 18:08:03.431902: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-22 18:08:03.435757: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-22 18:08:03.435913: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5579aa2c4810 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-22 18:08:03.435948: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 471895.56it/s] 86%|████████▌ | 8527872/9912422 [00:00<00:02, 672532.01it/s]9920512it [00:00, 45919764.06it/s]                           
0it [00:00, ?it/s]32768it [00:00, 668168.66it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 489835.25it/s]1654784it [00:00, 12177527.69it/s]                         
0it [00:00, ?it/s]8192it [00:00, 209611.57it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
