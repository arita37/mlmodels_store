
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f59ec7591e0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f59ec7591e0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5a5747f400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5a5747f400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5a7669bea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5a7669bea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5a047bb0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5a047bb0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5a047bb0d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 19%|█▉        | 1916928/9912422 [00:00<00:00, 19057433.60it/s] 82%|████████▏ | 8175616/9912422 [00:00<00:00, 24080990.10it/s]9920512it [00:00, 28499013.84it/s]                             
0it [00:00, ?it/s]32768it [00:00, 581245.36it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470619.03it/s]1654784it [00:00, 11711300.57it/s]                         
0it [00:00, ?it/s]8192it [00:00, 190435.68it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f59ec66bb70>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f59ec65de10>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5a047b2d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5a047b2d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5a047b2d08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:14,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:14,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:13,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:13,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:51,  3.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:51,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:51,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:50,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:50,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:35,  4.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:35,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:34,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  7.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:12, 10.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:12, 10.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:12, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:12, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:12, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:12, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:12, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:09, 14.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:09, 14.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:08, 14.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 14.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:08, 14.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:08, 14.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:08, 14.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 18.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 18.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 18.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 18.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 18.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 18.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 22.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 22.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 22.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 22.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 22.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 22.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 22.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 26.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 31.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 31.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 31.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 31.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 31.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 33.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 33.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 33.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 33.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 33.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 33.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 35.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 35.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 35.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 34.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 34.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 34.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 34.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 34.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 34.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 33.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 33.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:02, 33.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:02, 33.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:02, 33.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:02, 32.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:02, 32.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:02, 32.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:02, 32.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:02, 32.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:02, 32.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:02, 32.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:02, 32.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:02, 32.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:02, 32.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:02, 32.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:02, 32.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 32.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 32.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 31.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 31.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 31.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 31.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 31.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 32.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 32.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 32.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 32.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 32.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 32.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 32.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 32.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 32.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 32.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 29.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 29.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 29.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 29.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:01, 29.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:01, 27.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:01, 27.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:01, 27.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:01, 27.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:01, 26.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:01, 26.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:01, 26.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:01, 26.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:01, 23.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:01, 23.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:01, 23.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:01, 23.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:01, 21.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:01, 21.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:01, 21.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:01, 21.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:01, 20.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:01, 20.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:04<00:01, 20.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:04<00:01, 20.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:04<00:01, 19.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:04<00:01, 19.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:04<00:01, 19.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:04<00:01, 19.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:04<00:01, 19.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:04<00:01, 19.07 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:04<00:01, 19.07 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:04<00:01, 18.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:04<00:01, 18.71 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:04<00:01, 18.71 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:01, 18.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:01, 18.55 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:04<00:01, 18.55 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:01, 18.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:01, 18.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:01, 18.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:01, 18.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:01, 18.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:01, 18.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 18.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 18.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 18.39 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 18.39 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:05<00:00, 18.39 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:05<00:00, 18.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:05<00:00, 18.81 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:05<00:00, 18.81 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:05<00:00, 18.81 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:05<00:00, 18.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:05<00:00, 18.80 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:05<00:00, 18.80 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:05<00:00, 18.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:05<00:00, 18.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:05<00:00, 18.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:05<00:00, 18.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:05<00:00, 18.63 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:05<00:00, 18.63 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:05<00:00, 18.63 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:05<00:00, 18.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:05<00:00, 18.92 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:05<00:00, 18.92 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:05<00:00, 18.92 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:05<00:00, 19.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:05<00:00, 19.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 19.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.80s/ url]Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.80s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 19.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.80s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 19.31 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:05<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.29s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:09<00:00,  5.80s/ url]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 19.31 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.29s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.29s/ file]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 17.43 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:09<00:00,  9.29s/ url]
0 examples [00:00, ? examples/s]2020-08-22 06:09:46.427231: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-22 06:09:46.441889: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-22 06:09:46.442119: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c3504534c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-22 06:09:46.442144: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
10 examples [00:00, 98.84 examples/s]75 examples [00:00, 132.56 examples/s]140 examples [00:00, 173.99 examples/s]205 examples [00:00, 222.81 examples/s]267 examples [00:00, 275.29 examples/s]327 examples [00:00, 328.36 examples/s]388 examples [00:00, 380.03 examples/s]446 examples [00:00, 423.52 examples/s]510 examples [00:00, 470.81 examples/s]571 examples [00:01, 504.07 examples/s]630 examples [00:01, 516.81 examples/s]697 examples [00:01, 553.22 examples/s]761 examples [00:01, 575.39 examples/s]825 examples [00:01, 592.54 examples/s]888 examples [00:01, 600.67 examples/s]950 examples [00:01, 604.58 examples/s]1015 examples [00:01, 616.12 examples/s]1078 examples [00:01, 608.32 examples/s]1141 examples [00:01, 613.72 examples/s]1203 examples [00:02, 609.19 examples/s]1265 examples [00:02, 611.56 examples/s]1331 examples [00:02, 624.44 examples/s]1398 examples [00:02, 637.35 examples/s]1462 examples [00:02, 635.45 examples/s]1526 examples [00:02, 632.21 examples/s]1590 examples [00:02, 631.50 examples/s]1654 examples [00:02, 626.91 examples/s]1719 examples [00:02, 631.58 examples/s]1787 examples [00:02, 643.68 examples/s]1855 examples [00:03, 652.17 examples/s]1921 examples [00:03, 651.32 examples/s]1987 examples [00:03, 644.78 examples/s]2053 examples [00:03, 646.73 examples/s]2124 examples [00:03, 662.94 examples/s]2191 examples [00:03, 656.89 examples/s]2257 examples [00:03, 637.16 examples/s]2324 examples [00:03, 646.09 examples/s]2390 examples [00:03, 649.72 examples/s]2460 examples [00:03, 662.59 examples/s]2527 examples [00:04, 663.52 examples/s]2594 examples [00:04, 652.41 examples/s]2660 examples [00:04, 637.84 examples/s]2724 examples [00:04, 628.31 examples/s]2787 examples [00:04, 618.13 examples/s]2850 examples [00:04, 619.27 examples/s]2914 examples [00:04, 624.26 examples/s]2980 examples [00:04, 632.62 examples/s]3046 examples [00:04, 637.86 examples/s]3113 examples [00:05, 646.61 examples/s]3178 examples [00:05, 638.65 examples/s]3242 examples [00:05, 613.69 examples/s]3305 examples [00:05, 618.40 examples/s]3368 examples [00:05, 613.26 examples/s]3430 examples [00:05, 605.34 examples/s]3491 examples [00:05, 582.49 examples/s]3554 examples [00:05, 594.89 examples/s]3621 examples [00:05, 613.13 examples/s]3685 examples [00:05, 618.55 examples/s]3748 examples [00:06, 616.21 examples/s]3810 examples [00:06, 616.69 examples/s]3880 examples [00:06, 637.27 examples/s]3945 examples [00:06, 640.79 examples/s]4010 examples [00:06, 629.47 examples/s]4075 examples [00:06, 632.90 examples/s]4139 examples [00:06, 629.69 examples/s]4203 examples [00:06, 631.88 examples/s]4270 examples [00:06, 641.60 examples/s]4336 examples [00:06, 645.75 examples/s]4401 examples [00:07, 645.18 examples/s]4468 examples [00:07, 650.06 examples/s]4534 examples [00:07, 637.67 examples/s]4601 examples [00:07, 646.89 examples/s]4669 examples [00:07, 654.86 examples/s]4735 examples [00:07, 649.30 examples/s]4801 examples [00:07, 632.19 examples/s]4865 examples [00:07, 619.37 examples/s]4928 examples [00:07, 619.13 examples/s]4991 examples [00:08, 617.63 examples/s]5054 examples [00:08, 620.31 examples/s]5117 examples [00:08, 607.76 examples/s]5182 examples [00:08, 617.61 examples/s]5247 examples [00:08, 626.92 examples/s]5310 examples [00:08, 601.88 examples/s]5375 examples [00:08, 612.59 examples/s]5439 examples [00:08, 619.55 examples/s]5502 examples [00:08, 591.76 examples/s]5562 examples [00:08, 585.86 examples/s]5623 examples [00:09, 590.49 examples/s]5683 examples [00:09, 586.33 examples/s]5742 examples [00:09, 578.88 examples/s]5805 examples [00:09, 590.62 examples/s]5867 examples [00:09, 597.39 examples/s]5928 examples [00:09, 600.53 examples/s]5993 examples [00:09, 614.28 examples/s]6055 examples [00:09, 608.40 examples/s]6122 examples [00:09, 625.61 examples/s]6188 examples [00:09, 633.99 examples/s]6256 examples [00:10, 645.17 examples/s]6321 examples [00:10, 637.35 examples/s]6385 examples [00:10, 629.32 examples/s]6449 examples [00:10, 621.21 examples/s]6512 examples [00:10, 614.63 examples/s]6574 examples [00:10, 613.21 examples/s]6636 examples [00:10, 599.39 examples/s]6700 examples [00:10, 609.29 examples/s]6766 examples [00:10, 621.24 examples/s]6831 examples [00:11, 627.86 examples/s]6894 examples [00:11, 607.53 examples/s]6958 examples [00:11, 615.90 examples/s]7024 examples [00:11, 626.36 examples/s]7087 examples [00:11, 621.41 examples/s]7150 examples [00:11, 621.83 examples/s]7215 examples [00:11, 627.46 examples/s]7279 examples [00:11, 629.77 examples/s]7345 examples [00:11, 636.92 examples/s]7413 examples [00:11, 646.84 examples/s]7482 examples [00:12, 657.23 examples/s]7548 examples [00:12, 656.36 examples/s]7614 examples [00:12, 657.03 examples/s]7683 examples [00:12, 664.67 examples/s]7751 examples [00:12, 668.41 examples/s]7818 examples [00:12, 661.45 examples/s]7885 examples [00:12, 660.70 examples/s]7952 examples [00:12, 655.23 examples/s]8018 examples [00:12, 652.60 examples/s]8084 examples [00:12, 640.66 examples/s]8149 examples [00:13, 637.47 examples/s]8213 examples [00:13, 630.27 examples/s]8277 examples [00:13, 603.63 examples/s]8339 examples [00:13, 607.42 examples/s]8407 examples [00:13, 626.81 examples/s]8477 examples [00:13, 645.65 examples/s]8542 examples [00:13, 643.68 examples/s]8607 examples [00:13, 599.92 examples/s]8670 examples [00:13, 606.24 examples/s]8732 examples [00:14, 609.61 examples/s]8794 examples [00:14, 593.99 examples/s]8854 examples [00:14, 593.28 examples/s]8918 examples [00:14, 604.29 examples/s]8979 examples [00:14, 602.05 examples/s]9040 examples [00:14, 600.56 examples/s]9101 examples [00:14, 581.93 examples/s]9160 examples [00:14, 582.91 examples/s]9220 examples [00:14, 586.25 examples/s]9279 examples [00:14, 582.19 examples/s]9338 examples [00:15, 580.68 examples/s]9400 examples [00:15, 590.63 examples/s]9460 examples [00:15, 585.52 examples/s]9525 examples [00:15, 600.99 examples/s]9591 examples [00:15, 615.49 examples/s]9656 examples [00:15, 624.13 examples/s]9719 examples [00:15, 622.14 examples/s]9783 examples [00:15, 626.15 examples/s]9849 examples [00:15, 635.93 examples/s]9913 examples [00:15, 630.98 examples/s]9979 examples [00:16, 637.69 examples/s]10043 examples [00:16, 579.11 examples/s]10104 examples [00:16, 585.46 examples/s]10168 examples [00:16, 599.98 examples/s]10229 examples [00:16, 594.66 examples/s]10297 examples [00:16, 616.38 examples/s]10361 examples [00:16, 623.24 examples/s]10424 examples [00:16, 621.03 examples/s]10491 examples [00:16, 634.45 examples/s]10557 examples [00:17, 639.28 examples/s]10625 examples [00:17, 649.46 examples/s]10691 examples [00:17, 648.86 examples/s]10758 examples [00:17, 652.97 examples/s]10828 examples [00:17, 664.22 examples/s]10895 examples [00:17, 664.86 examples/s]10962 examples [00:17, 656.00 examples/s]11028 examples [00:17, 648.87 examples/s]11093 examples [00:17, 639.40 examples/s]11159 examples [00:17, 645.01 examples/s]11227 examples [00:18, 653.71 examples/s]11293 examples [00:18, 646.59 examples/s]11358 examples [00:18, 638.31 examples/s]11426 examples [00:18, 650.02 examples/s]11492 examples [00:18, 633.07 examples/s]11556 examples [00:18, 625.12 examples/s]11619 examples [00:18, 622.10 examples/s]11682 examples [00:18, 611.91 examples/s]11744 examples [00:18, 609.55 examples/s]11806 examples [00:18, 605.03 examples/s]11867 examples [00:19, 595.55 examples/s]11930 examples [00:19, 603.39 examples/s]11991 examples [00:19, 605.03 examples/s]12053 examples [00:19, 608.25 examples/s]12114 examples [00:19, 602.67 examples/s]12178 examples [00:19, 612.89 examples/s]12242 examples [00:19, 620.28 examples/s]12305 examples [00:19, 614.01 examples/s]12367 examples [00:19, 559.11 examples/s]12424 examples [00:20, 463.52 examples/s]12474 examples [00:20, 473.87 examples/s]12529 examples [00:20, 494.09 examples/s]12592 examples [00:20, 527.54 examples/s]12657 examples [00:20, 558.37 examples/s]12722 examples [00:20, 581.17 examples/s]12782 examples [00:20, 578.27 examples/s]12844 examples [00:20, 588.97 examples/s]12908 examples [00:20, 602.62 examples/s]12971 examples [00:21, 609.34 examples/s]13034 examples [00:21, 613.73 examples/s]13096 examples [00:21, 604.81 examples/s]13161 examples [00:21, 616.76 examples/s]13231 examples [00:21, 637.72 examples/s]13300 examples [00:21, 650.30 examples/s]13366 examples [00:21, 630.71 examples/s]13430 examples [00:21, 627.47 examples/s]13498 examples [00:21, 641.94 examples/s]13567 examples [00:21, 653.29 examples/s]13633 examples [00:22, 650.47 examples/s]13699 examples [00:22, 650.21 examples/s]13765 examples [00:22, 647.64 examples/s]13830 examples [00:22, 641.85 examples/s]13896 examples [00:22, 646.65 examples/s]13961 examples [00:22, 636.64 examples/s]14025 examples [00:22, 634.15 examples/s]14089 examples [00:22, 626.94 examples/s]14152 examples [00:22, 623.58 examples/s]14216 examples [00:22, 627.21 examples/s]14282 examples [00:23, 634.48 examples/s]14346 examples [00:23, 596.69 examples/s]14407 examples [00:23, 585.64 examples/s]14467 examples [00:23, 589.31 examples/s]14529 examples [00:23, 597.54 examples/s]14590 examples [00:23, 599.07 examples/s]14651 examples [00:23, 576.15 examples/s]14709 examples [00:23, 561.49 examples/s]14768 examples [00:23, 568.46 examples/s]14831 examples [00:24, 584.97 examples/s]14890 examples [00:24, 579.72 examples/s]14952 examples [00:24, 590.32 examples/s]15012 examples [00:24, 586.72 examples/s]15071 examples [00:24, 586.67 examples/s]15130 examples [00:24, 584.64 examples/s]15190 examples [00:24, 587.36 examples/s]15249 examples [00:24, 582.73 examples/s]15308 examples [00:24, 573.48 examples/s]15369 examples [00:24, 582.00 examples/s]15432 examples [00:25, 595.39 examples/s]15499 examples [00:25, 613.64 examples/s]15562 examples [00:25, 618.08 examples/s]15624 examples [00:25, 612.22 examples/s]15693 examples [00:25, 633.28 examples/s]15757 examples [00:25, 633.87 examples/s]15826 examples [00:25, 649.09 examples/s]15892 examples [00:25, 647.11 examples/s]15957 examples [00:25, 620.94 examples/s]16021 examples [00:25, 626.01 examples/s]16084 examples [00:26, 623.00 examples/s]16150 examples [00:26, 631.47 examples/s]16217 examples [00:26, 641.03 examples/s]16288 examples [00:26, 658.14 examples/s]16359 examples [00:26, 672.14 examples/s]16427 examples [00:26, 673.69 examples/s]16495 examples [00:26, 640.94 examples/s]16560 examples [00:26, 625.11 examples/s]16623 examples [00:26, 619.60 examples/s]16689 examples [00:27, 630.93 examples/s]16753 examples [00:27, 628.91 examples/s]16824 examples [00:27, 650.60 examples/s]16897 examples [00:27, 670.94 examples/s]16966 examples [00:27, 673.85 examples/s]17034 examples [00:27, 645.42 examples/s]17103 examples [00:27, 656.91 examples/s]17173 examples [00:27, 668.22 examples/s]17241 examples [00:27, 658.45 examples/s]17308 examples [00:27, 634.81 examples/s]17372 examples [00:28, 614.18 examples/s]17434 examples [00:28, 608.34 examples/s]17496 examples [00:28, 602.65 examples/s]17557 examples [00:28, 583.14 examples/s]17616 examples [00:28, 571.66 examples/s]17677 examples [00:28, 580.48 examples/s]17739 examples [00:28, 589.97 examples/s]17803 examples [00:28, 603.84 examples/s]17864 examples [00:28, 600.05 examples/s]17925 examples [00:28, 595.11 examples/s]17985 examples [00:29, 583.35 examples/s]18050 examples [00:29, 599.97 examples/s]18116 examples [00:29, 615.01 examples/s]18186 examples [00:29, 636.94 examples/s]18252 examples [00:29, 642.00 examples/s]18321 examples [00:29, 653.35 examples/s]18387 examples [00:29, 639.24 examples/s]18452 examples [00:29, 627.57 examples/s]18516 examples [00:29, 626.43 examples/s]18579 examples [00:30, 626.85 examples/s]18653 examples [00:30, 656.98 examples/s]18724 examples [00:30, 671.49 examples/s]18794 examples [00:30, 678.74 examples/s]18863 examples [00:30, 681.06 examples/s]18932 examples [00:30, 672.68 examples/s]19000 examples [00:30, 667.75 examples/s]19067 examples [00:30, 658.08 examples/s]19133 examples [00:30, 636.56 examples/s]19197 examples [00:30, 618.53 examples/s]19260 examples [00:31, 603.37 examples/s]19321 examples [00:31, 582.42 examples/s]19380 examples [00:31, 576.68 examples/s]19439 examples [00:31, 578.95 examples/s]19505 examples [00:31, 599.06 examples/s]19573 examples [00:31, 620.94 examples/s]19640 examples [00:31, 634.12 examples/s]19712 examples [00:31, 655.16 examples/s]19785 examples [00:31, 674.73 examples/s]19856 examples [00:31, 683.45 examples/s]19927 examples [00:32, 689.46 examples/s]19997 examples [00:32, 688.33 examples/s]20067 examples [00:32, 626.04 examples/s]20134 examples [00:32, 636.74 examples/s]20199 examples [00:32, 635.10 examples/s]20264 examples [00:32, 635.94 examples/s]20329 examples [00:32, 608.28 examples/s]20392 examples [00:32, 613.73 examples/s]20454 examples [00:32, 614.79 examples/s]20517 examples [00:33, 618.19 examples/s]20584 examples [00:33, 631.51 examples/s]20648 examples [00:33, 625.11 examples/s]20716 examples [00:33, 639.33 examples/s]20786 examples [00:33, 655.62 examples/s]20858 examples [00:33, 671.89 examples/s]20926 examples [00:33, 648.26 examples/s]20992 examples [00:33, 626.43 examples/s]21056 examples [00:33, 619.35 examples/s]21119 examples [00:34, 619.73 examples/s]21182 examples [00:34, 620.05 examples/s]21245 examples [00:34, 616.59 examples/s]21311 examples [00:34, 626.84 examples/s]21376 examples [00:34, 633.22 examples/s]21440 examples [00:34, 610.11 examples/s]21502 examples [00:34, 612.37 examples/s]21564 examples [00:34, 608.79 examples/s]21626 examples [00:34, 584.29 examples/s]21687 examples [00:34, 589.52 examples/s]21747 examples [00:35, 589.55 examples/s]21810 examples [00:35, 599.22 examples/s]21875 examples [00:35, 613.06 examples/s]21939 examples [00:35, 620.32 examples/s]22004 examples [00:35, 628.41 examples/s]22070 examples [00:35, 635.36 examples/s]22140 examples [00:35, 653.46 examples/s]22206 examples [00:35, 625.80 examples/s]22269 examples [00:35, 615.99 examples/s]22339 examples [00:35, 636.84 examples/s]22406 examples [00:36, 644.00 examples/s]22473 examples [00:36, 651.52 examples/s]22540 examples [00:36, 654.69 examples/s]22606 examples [00:36, 654.54 examples/s]22674 examples [00:36, 659.67 examples/s]22741 examples [00:36, 659.76 examples/s]22808 examples [00:36, 628.24 examples/s]22872 examples [00:36, 607.70 examples/s]22934 examples [00:36, 606.95 examples/s]22995 examples [00:37, 603.29 examples/s]23060 examples [00:37, 614.14 examples/s]23127 examples [00:37, 627.79 examples/s]23194 examples [00:37, 637.42 examples/s]23263 examples [00:37, 650.06 examples/s]23331 examples [00:37, 657.48 examples/s]23397 examples [00:37, 657.97 examples/s]23463 examples [00:37, 647.16 examples/s]23528 examples [00:37, 647.59 examples/s]23596 examples [00:37, 656.21 examples/s]23662 examples [00:38, 656.50 examples/s]23728 examples [00:38, 656.71 examples/s]23795 examples [00:38, 658.65 examples/s]23861 examples [00:38, 646.78 examples/s]23926 examples [00:38, 638.55 examples/s]23990 examples [00:38, 635.69 examples/s]24054 examples [00:38, 633.74 examples/s]24118 examples [00:38, 607.84 examples/s]24180 examples [00:38, 600.11 examples/s]24244 examples [00:38, 608.79 examples/s]24306 examples [00:39, 610.52 examples/s]24373 examples [00:39, 625.33 examples/s]24437 examples [00:39, 627.47 examples/s]24500 examples [00:39, 625.61 examples/s]24566 examples [00:39, 633.13 examples/s]24630 examples [00:39, 626.74 examples/s]24693 examples [00:39, 604.17 examples/s]24754 examples [00:39, 584.56 examples/s]24815 examples [00:39, 590.94 examples/s]24875 examples [00:40, 562.93 examples/s]24944 examples [00:40, 593.65 examples/s]25014 examples [00:40, 620.63 examples/s]25079 examples [00:40, 626.96 examples/s]25143 examples [00:40, 627.30 examples/s]25210 examples [00:40, 637.27 examples/s]25279 examples [00:40, 651.15 examples/s]25345 examples [00:40, 638.43 examples/s]25410 examples [00:40, 609.31 examples/s]25472 examples [00:40, 587.57 examples/s]25532 examples [00:41, 582.49 examples/s]25594 examples [00:41, 592.35 examples/s]25654 examples [00:41, 593.93 examples/s]25720 examples [00:41, 612.26 examples/s]25784 examples [00:41, 619.45 examples/s]25855 examples [00:41, 642.91 examples/s]25920 examples [00:41, 635.65 examples/s]25984 examples [00:41, 632.34 examples/s]26048 examples [00:41, 613.74 examples/s]26114 examples [00:41, 624.96 examples/s]26180 examples [00:42, 634.49 examples/s]26244 examples [00:42, 635.90 examples/s]26312 examples [00:42, 648.16 examples/s]26377 examples [00:42, 644.67 examples/s]26449 examples [00:42, 664.10 examples/s]26517 examples [00:42, 667.75 examples/s]26584 examples [00:42, 657.42 examples/s]26658 examples [00:42, 678.05 examples/s]26728 examples [00:42, 681.98 examples/s]26800 examples [00:43, 690.81 examples/s]26872 examples [00:43, 696.89 examples/s]26942 examples [00:43, 694.48 examples/s]27014 examples [00:43, 700.12 examples/s]27085 examples [00:43, 689.56 examples/s]27157 examples [00:43, 696.87 examples/s]27227 examples [00:43, 659.65 examples/s]27294 examples [00:43, 643.66 examples/s]27364 examples [00:43, 658.27 examples/s]27436 examples [00:43, 674.49 examples/s]27506 examples [00:44, 680.82 examples/s]27575 examples [00:44, 645.10 examples/s]27641 examples [00:44, 622.19 examples/s]27704 examples [00:44, 623.11 examples/s]27770 examples [00:44, 626.34 examples/s]27838 examples [00:44, 640.31 examples/s]27903 examples [00:44, 638.13 examples/s]27968 examples [00:44, 635.64 examples/s]28033 examples [00:44, 639.83 examples/s]28104 examples [00:45, 657.40 examples/s]28170 examples [00:45, 647.08 examples/s]28237 examples [00:45, 652.06 examples/s]28305 examples [00:45, 659.99 examples/s]28372 examples [00:45, 652.62 examples/s]28439 examples [00:45, 655.74 examples/s]28505 examples [00:45, 646.27 examples/s]28570 examples [00:45, 621.47 examples/s]28633 examples [00:45, 604.65 examples/s]28699 examples [00:45, 619.13 examples/s]28763 examples [00:46, 625.14 examples/s]28826 examples [00:46, 624.76 examples/s]28890 examples [00:46, 629.18 examples/s]28957 examples [00:46, 639.36 examples/s]29026 examples [00:46, 653.46 examples/s]29096 examples [00:46, 666.18 examples/s]29166 examples [00:46, 675.28 examples/s]29234 examples [00:46, 658.16 examples/s]29301 examples [00:46, 650.49 examples/s]29367 examples [00:46, 645.02 examples/s]29432 examples [00:47, 641.88 examples/s]29497 examples [00:47, 611.94 examples/s]29560 examples [00:47, 615.91 examples/s]29623 examples [00:47, 618.24 examples/s]29691 examples [00:47, 633.43 examples/s]29758 examples [00:47, 641.52 examples/s]29827 examples [00:47, 653.49 examples/s]29894 examples [00:47, 657.55 examples/s]29960 examples [00:47, 650.69 examples/s]30026 examples [00:48, 593.04 examples/s]30094 examples [00:48, 614.81 examples/s]30158 examples [00:48, 620.52 examples/s]30221 examples [00:48, 621.50 examples/s]30285 examples [00:48, 625.41 examples/s]30354 examples [00:48, 643.47 examples/s]30421 examples [00:48, 650.85 examples/s]30487 examples [00:48, 650.46 examples/s]30553 examples [00:48, 638.11 examples/s]30621 examples [00:48, 649.63 examples/s]30687 examples [00:49, 651.85 examples/s]30753 examples [00:49, 651.58 examples/s]30819 examples [00:49, 648.92 examples/s]30884 examples [00:49, 640.54 examples/s]30949 examples [00:49, 639.55 examples/s]31018 examples [00:49, 652.50 examples/s]31084 examples [00:49, 650.20 examples/s]31150 examples [00:49, 650.66 examples/s]31216 examples [00:49, 650.43 examples/s]31285 examples [00:49, 659.96 examples/s]31352 examples [00:50, 658.09 examples/s]31418 examples [00:50, 636.83 examples/s]31485 examples [00:50, 645.70 examples/s]31552 examples [00:50, 650.88 examples/s]31621 examples [00:50, 660.44 examples/s]31688 examples [00:50, 658.70 examples/s]31754 examples [00:50, 654.69 examples/s]31820 examples [00:50, 642.47 examples/s]31885 examples [00:50, 634.54 examples/s]31949 examples [00:50, 635.07 examples/s]32013 examples [00:51, 633.51 examples/s]32084 examples [00:51, 653.63 examples/s]32152 examples [00:51, 660.02 examples/s]32221 examples [00:51, 668.15 examples/s]32288 examples [00:51, 659.03 examples/s]32355 examples [00:51, 658.97 examples/s]32421 examples [00:51, 642.57 examples/s]32486 examples [00:51, 639.36 examples/s]32551 examples [00:51, 639.39 examples/s]32616 examples [00:52, 641.62 examples/s]32681 examples [00:52, 642.77 examples/s]32749 examples [00:52, 652.89 examples/s]32815 examples [00:52, 648.21 examples/s]32883 examples [00:52, 655.43 examples/s]32954 examples [00:52, 670.24 examples/s]33024 examples [00:52, 678.61 examples/s]33092 examples [00:52, 669.23 examples/s]33160 examples [00:52, 640.33 examples/s]33225 examples [00:52, 636.52 examples/s]33292 examples [00:53, 643.33 examples/s]33357 examples [00:53, 637.49 examples/s]33421 examples [00:53, 634.48 examples/s]33490 examples [00:53, 650.14 examples/s]33556 examples [00:53, 647.33 examples/s]33622 examples [00:53, 650.60 examples/s]33688 examples [00:53, 651.87 examples/s]33758 examples [00:53, 663.67 examples/s]33828 examples [00:53, 673.82 examples/s]33896 examples [00:53, 658.05 examples/s]33962 examples [00:54, 651.11 examples/s]34028 examples [00:54, 644.07 examples/s]34093 examples [00:54, 643.59 examples/s]34159 examples [00:54, 648.11 examples/s]34224 examples [00:54, 643.72 examples/s]34289 examples [00:54, 644.36 examples/s]34354 examples [00:54, 614.39 examples/s]34418 examples [00:54, 618.54 examples/s]34481 examples [00:54, 600.95 examples/s]34542 examples [00:55, 583.17 examples/s]34601 examples [00:55, 578.98 examples/s]34663 examples [00:55, 590.12 examples/s]34731 examples [00:55, 613.24 examples/s]34795 examples [00:55, 618.97 examples/s]34859 examples [00:55, 623.74 examples/s]34925 examples [00:55, 632.72 examples/s]34990 examples [00:55, 636.31 examples/s]35058 examples [00:55, 645.66 examples/s]35123 examples [00:55, 643.92 examples/s]35188 examples [00:56, 632.47 examples/s]35252 examples [00:56, 632.00 examples/s]35316 examples [00:56, 610.79 examples/s]35382 examples [00:56, 622.57 examples/s]35445 examples [00:56, 619.54 examples/s]35508 examples [00:56, 619.68 examples/s]35577 examples [00:56, 638.06 examples/s]35642 examples [00:56, 639.91 examples/s]35708 examples [00:56, 643.44 examples/s]35775 examples [00:56, 648.42 examples/s]35843 examples [00:57, 657.58 examples/s]35909 examples [00:57, 641.20 examples/s]35974 examples [00:57, 607.36 examples/s]36036 examples [00:57, 598.34 examples/s]36097 examples [00:57, 599.51 examples/s]36161 examples [00:57, 609.04 examples/s]36225 examples [00:57, 617.00 examples/s]36287 examples [00:57, 612.16 examples/s]36349 examples [00:57, 610.54 examples/s]36411 examples [00:58, 606.36 examples/s]36473 examples [00:58, 607.93 examples/s]36541 examples [00:58, 627.34 examples/s]36605 examples [00:58, 630.47 examples/s]36677 examples [00:58, 653.73 examples/s]36743 examples [00:58, 655.01 examples/s]36810 examples [00:58, 657.27 examples/s]36879 examples [00:58, 664.24 examples/s]36946 examples [00:58, 658.50 examples/s]37014 examples [00:58, 662.20 examples/s]37081 examples [00:59, 646.88 examples/s]37146 examples [00:59, 640.48 examples/s]37211 examples [00:59, 638.87 examples/s]37275 examples [00:59, 630.64 examples/s]37341 examples [00:59, 637.46 examples/s]37405 examples [00:59, 633.19 examples/s]37471 examples [00:59, 639.49 examples/s]37540 examples [00:59, 651.42 examples/s]37606 examples [00:59, 646.38 examples/s]37671 examples [00:59, 629.59 examples/s]37735 examples [01:00, 598.73 examples/s]37801 examples [01:00, 613.68 examples/s]37869 examples [01:00, 631.54 examples/s]37940 examples [01:00, 652.76 examples/s]38008 examples [01:00, 659.88 examples/s]38075 examples [01:00, 645.42 examples/s]38140 examples [01:00, 645.16 examples/s]38205 examples [01:00, 645.60 examples/s]38275 examples [01:00, 660.22 examples/s]38342 examples [01:01, 651.89 examples/s]38408 examples [01:01, 622.34 examples/s]38478 examples [01:01, 640.86 examples/s]38547 examples [01:01, 654.57 examples/s]38614 examples [01:01, 656.52 examples/s]38682 examples [01:01, 661.63 examples/s]38751 examples [01:01, 669.31 examples/s]38819 examples [01:01, 659.17 examples/s]38886 examples [01:01, 643.22 examples/s]38952 examples [01:01, 646.25 examples/s]39017 examples [01:02, 642.60 examples/s]39082 examples [01:02, 635.58 examples/s]39148 examples [01:02, 642.13 examples/s]39213 examples [01:02, 636.54 examples/s]39277 examples [01:02, 630.15 examples/s]39341 examples [01:02, 628.22 examples/s]39404 examples [01:02, 622.45 examples/s]39467 examples [01:02, 616.38 examples/s]39529 examples [01:02, 599.73 examples/s]39594 examples [01:02, 612.15 examples/s]39656 examples [01:03, 613.41 examples/s]39718 examples [01:03, 608.48 examples/s]39784 examples [01:03, 623.01 examples/s]39851 examples [01:03, 634.32 examples/s]39918 examples [01:03, 642.23 examples/s]39983 examples [01:03, 639.72 examples/s]40048 examples [01:03, 580.59 examples/s]40116 examples [01:03, 606.56 examples/s]40179 examples [01:03, 611.88 examples/s]40241 examples [01:04, 612.11 examples/s]40303 examples [01:04, 608.97 examples/s]40365 examples [01:04, 587.98 examples/s]40425 examples [01:04, 590.39 examples/s]40489 examples [01:04, 603.27 examples/s]40550 examples [01:04, 595.22 examples/s]40614 examples [01:04, 605.45 examples/s]40681 examples [01:04, 621.99 examples/s]40748 examples [01:04, 634.44 examples/s]40812 examples [01:04, 618.53 examples/s]40875 examples [01:05, 613.12 examples/s]40937 examples [01:05, 614.73 examples/s]41002 examples [01:05, 622.54 examples/s]41065 examples [01:05, 611.59 examples/s]41127 examples [01:05, 599.20 examples/s]41189 examples [01:05, 603.52 examples/s]41250 examples [01:05, 587.48 examples/s]41312 examples [01:05, 595.20 examples/s]41377 examples [01:05, 609.89 examples/s]41440 examples [01:06, 614.45 examples/s]41502 examples [01:06, 594.22 examples/s]41569 examples [01:06, 613.25 examples/s]41633 examples [01:06, 620.73 examples/s]41700 examples [01:06, 632.90 examples/s]41768 examples [01:06, 644.11 examples/s]41834 examples [01:06, 646.92 examples/s]41903 examples [01:06, 657.68 examples/s]41972 examples [01:06, 667.03 examples/s]42041 examples [01:06, 671.25 examples/s]42109 examples [01:07, 669.77 examples/s]42177 examples [01:07, 662.17 examples/s]42244 examples [01:07, 659.12 examples/s]42312 examples [01:07, 663.59 examples/s]42379 examples [01:07, 633.95 examples/s]42448 examples [01:07, 649.48 examples/s]42514 examples [01:07, 647.76 examples/s]42579 examples [01:07, 647.85 examples/s]42650 examples [01:07, 663.13 examples/s]42719 examples [01:07, 669.51 examples/s]42788 examples [01:08, 675.24 examples/s]42856 examples [01:08, 669.70 examples/s]42924 examples [01:08, 670.47 examples/s]42992 examples [01:08, 670.52 examples/s]43060 examples [01:08, 656.61 examples/s]43126 examples [01:08, 656.62 examples/s]43192 examples [01:08, 648.13 examples/s]43257 examples [01:08, 636.02 examples/s]43321 examples [01:08, 627.91 examples/s]43385 examples [01:08, 630.52 examples/s]43451 examples [01:09, 637.43 examples/s]43515 examples [01:09, 637.89 examples/s]43583 examples [01:09, 647.20 examples/s]43654 examples [01:09, 662.60 examples/s]43724 examples [01:09, 670.95 examples/s]43793 examples [01:09, 676.26 examples/s]43861 examples [01:09, 661.76 examples/s]43928 examples [01:09, 656.30 examples/s]43996 examples [01:09, 662.90 examples/s]44065 examples [01:10, 668.60 examples/s]44133 examples [01:10, 670.29 examples/s]44201 examples [01:10, 667.23 examples/s]44269 examples [01:10, 670.16 examples/s]44337 examples [01:10, 660.08 examples/s]44404 examples [01:10, 651.67 examples/s]44470 examples [01:10, 646.01 examples/s]44535 examples [01:10, 636.37 examples/s]44602 examples [01:10, 645.44 examples/s]44667 examples [01:10, 615.20 examples/s]44729 examples [01:11, 606.04 examples/s]44790 examples [01:11, 602.69 examples/s]44853 examples [01:11, 608.35 examples/s]44918 examples [01:11, 618.99 examples/s]44981 examples [01:11, 621.58 examples/s]45049 examples [01:11, 637.35 examples/s]45116 examples [01:11, 645.23 examples/s]45181 examples [01:11, 629.50 examples/s]45248 examples [01:11, 638.69 examples/s]45313 examples [01:11, 637.43 examples/s]45380 examples [01:12, 645.10 examples/s]45445 examples [01:12, 638.32 examples/s]45515 examples [01:12, 653.89 examples/s]45581 examples [01:12, 654.95 examples/s]45650 examples [01:12, 664.96 examples/s]45720 examples [01:12, 673.05 examples/s]45788 examples [01:12, 675.08 examples/s]45857 examples [01:12, 679.19 examples/s]45927 examples [01:12, 683.67 examples/s]45997 examples [01:12, 686.50 examples/s]46068 examples [01:13, 692.09 examples/s]46138 examples [01:13, 678.41 examples/s]46206 examples [01:13, 676.27 examples/s]46274 examples [01:13, 668.88 examples/s]46341 examples [01:13, 654.54 examples/s]46407 examples [01:13, 645.04 examples/s]46472 examples [01:13, 637.33 examples/s]46536 examples [01:13, 615.16 examples/s]46598 examples [01:13, 595.63 examples/s]46658 examples [01:14, 579.89 examples/s]46718 examples [01:14, 584.67 examples/s]46777 examples [01:14, 576.36 examples/s]46838 examples [01:14, 583.34 examples/s]46899 examples [01:14, 582.79 examples/s]46960 examples [01:14, 587.96 examples/s]47020 examples [01:14, 588.93 examples/s]47079 examples [01:14, 586.96 examples/s]47144 examples [01:14, 602.20 examples/s]47208 examples [01:14, 608.41 examples/s]47273 examples [01:15, 619.56 examples/s]47336 examples [01:15, 616.05 examples/s]47399 examples [01:15, 617.77 examples/s]47463 examples [01:15, 621.43 examples/s]47526 examples [01:15, 617.00 examples/s]47588 examples [01:15, 608.99 examples/s]47649 examples [01:15, 581.24 examples/s]47710 examples [01:15, 587.69 examples/s]47772 examples [01:15, 596.27 examples/s]47832 examples [01:16, 597.22 examples/s]47892 examples [01:16, 592.27 examples/s]47954 examples [01:16, 597.71 examples/s]48014 examples [01:16, 585.71 examples/s]48074 examples [01:16, 589.62 examples/s]48134 examples [01:16, 589.45 examples/s]48203 examples [01:16, 614.94 examples/s]48265 examples [01:16, 606.52 examples/s]48332 examples [01:16, 624.03 examples/s]48402 examples [01:16, 643.48 examples/s]48469 examples [01:17, 648.90 examples/s]48535 examples [01:17, 649.44 examples/s]48601 examples [01:17, 645.69 examples/s]48666 examples [01:17, 632.24 examples/s]48730 examples [01:17, 631.37 examples/s]48794 examples [01:17, 619.99 examples/s]48857 examples [01:17, 612.17 examples/s]48919 examples [01:17, 612.10 examples/s]48981 examples [01:17, 606.90 examples/s]49046 examples [01:17, 616.57 examples/s]49109 examples [01:18, 619.06 examples/s]49171 examples [01:18, 617.35 examples/s]49233 examples [01:18, 598.72 examples/s]49294 examples [01:18, 600.59 examples/s]49358 examples [01:18, 610.77 examples/s]49426 examples [01:18, 629.79 examples/s]49492 examples [01:18, 638.38 examples/s]49557 examples [01:18, 601.79 examples/s]49621 examples [01:18, 612.72 examples/s]49690 examples [01:19, 631.81 examples/s]49756 examples [01:19, 638.50 examples/s]49825 examples [01:19, 650.93 examples/s]49891 examples [01:19, 645.92 examples/s]49961 examples [01:19, 658.87 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s]  5%|▌         | 2689/50000 [00:00<00:01, 26887.14 examples/s] 14%|█▎        | 6859/50000 [00:00<00:01, 30094.05 examples/s] 25%|██▌       | 12579/50000 [00:00<00:01, 35081.09 examples/s] 41%|████      | 20464/50000 [00:00<00:00, 42089.88 examples/s] 55%|█████▍    | 27410/50000 [00:00<00:00, 47732.18 examples/s] 71%|███████   | 35389/50000 [00:00<00:00, 54273.77 examples/s] 87%|████████▋ | 43569/50000 [00:00<00:00, 60367.93 examples/s]                                                               0 examples [00:00, ? examples/s]39 examples [00:00, 385.50 examples/s]104 examples [00:00, 438.42 examples/s]171 examples [00:00, 489.01 examples/s]233 examples [00:00, 521.26 examples/s]299 examples [00:00, 555.74 examples/s]362 examples [00:00, 575.12 examples/s]432 examples [00:00, 605.33 examples/s]497 examples [00:00, 616.22 examples/s]563 examples [00:00, 626.88 examples/s]629 examples [00:01, 634.88 examples/s]695 examples [00:01, 641.76 examples/s]760 examples [00:01, 643.43 examples/s]825 examples [00:01, 641.29 examples/s]893 examples [00:01, 651.08 examples/s]959 examples [00:01, 621.04 examples/s]1022 examples [00:01, 568.13 examples/s]1085 examples [00:01, 583.85 examples/s]1152 examples [00:01, 606.35 examples/s]1217 examples [00:01, 618.35 examples/s]1280 examples [00:02, 620.03 examples/s]1347 examples [00:02, 633.08 examples/s]1411 examples [00:02, 629.15 examples/s]1477 examples [00:02, 637.09 examples/s]1541 examples [00:02, 637.70 examples/s]1605 examples [00:02, 638.10 examples/s]1673 examples [00:02, 650.10 examples/s]1742 examples [00:02, 659.91 examples/s]1809 examples [00:02, 645.70 examples/s]1874 examples [00:02, 644.54 examples/s]1939 examples [00:03, 642.43 examples/s]2007 examples [00:03, 651.61 examples/s]2078 examples [00:03, 666.79 examples/s]2145 examples [00:03, 660.49 examples/s]2215 examples [00:03, 669.36 examples/s]2283 examples [00:03, 658.14 examples/s]2353 examples [00:03, 668.00 examples/s]2424 examples [00:03, 678.08 examples/s]2493 examples [00:03, 679.77 examples/s]2562 examples [00:04, 656.02 examples/s]2628 examples [00:04, 647.39 examples/s]2701 examples [00:04, 669.98 examples/s]2773 examples [00:04, 682.86 examples/s]2842 examples [00:04, 676.17 examples/s]2912 examples [00:04, 680.33 examples/s]2981 examples [00:04, 678.85 examples/s]3049 examples [00:04, 661.64 examples/s]3116 examples [00:04, 647.32 examples/s]3181 examples [00:04, 631.06 examples/s]3245 examples [00:05, 622.21 examples/s]3311 examples [00:05, 632.55 examples/s]3377 examples [00:05, 638.27 examples/s]3443 examples [00:05, 641.75 examples/s]3509 examples [00:05, 645.97 examples/s]3576 examples [00:05, 649.60 examples/s]3648 examples [00:05, 668.55 examples/s]3718 examples [00:05, 677.03 examples/s]3786 examples [00:05, 670.89 examples/s]3854 examples [00:05, 672.02 examples/s]3922 examples [00:06, 666.68 examples/s]3992 examples [00:06, 675.96 examples/s]4063 examples [00:06, 684.56 examples/s]4132 examples [00:06, 679.37 examples/s]4201 examples [00:06, 664.41 examples/s]4268 examples [00:06, 652.23 examples/s]4335 examples [00:06, 656.20 examples/s]4406 examples [00:06, 669.62 examples/s]4474 examples [00:06, 671.93 examples/s]4546 examples [00:06, 684.50 examples/s]4615 examples [00:07, 680.53 examples/s]4686 examples [00:07, 686.12 examples/s]4755 examples [00:07, 681.02 examples/s]4824 examples [00:07, 664.77 examples/s]4892 examples [00:07, 668.67 examples/s]4959 examples [00:07, 642.07 examples/s]5024 examples [00:07, 634.74 examples/s]5090 examples [00:07, 641.91 examples/s]5158 examples [00:07, 652.80 examples/s]5224 examples [00:08, 654.43 examples/s]5290 examples [00:08, 638.81 examples/s]5355 examples [00:08, 578.41 examples/s]5417 examples [00:08, 588.23 examples/s]5477 examples [00:08, 469.80 examples/s]5547 examples [00:08, 519.50 examples/s]5609 examples [00:08, 544.45 examples/s]5676 examples [00:08, 575.90 examples/s]5741 examples [00:08, 594.68 examples/s]5807 examples [00:09, 610.25 examples/s]5876 examples [00:09, 630.75 examples/s]5941 examples [00:09, 635.29 examples/s]6012 examples [00:09, 654.91 examples/s]6079 examples [00:09, 646.93 examples/s]6149 examples [00:09, 661.63 examples/s]6216 examples [00:09, 651.95 examples/s]6282 examples [00:09, 644.73 examples/s]6349 examples [00:09, 649.25 examples/s]6416 examples [00:09, 654.68 examples/s]6482 examples [00:10, 648.23 examples/s]6549 examples [00:10, 654.29 examples/s]6616 examples [00:10, 658.23 examples/s]6682 examples [00:10, 653.73 examples/s]6750 examples [00:10, 658.87 examples/s]6816 examples [00:10, 653.55 examples/s]6885 examples [00:10, 662.24 examples/s]6952 examples [00:10, 643.26 examples/s]7017 examples [00:10, 634.96 examples/s]7081 examples [00:11, 618.82 examples/s]7144 examples [00:11, 602.23 examples/s]7205 examples [00:11, 591.68 examples/s]7265 examples [00:11, 589.11 examples/s]7330 examples [00:11, 605.01 examples/s]7392 examples [00:11, 607.72 examples/s]7454 examples [00:11, 609.65 examples/s]7516 examples [00:11, 589.10 examples/s]7578 examples [00:11, 596.12 examples/s]7645 examples [00:11, 615.31 examples/s]7707 examples [00:12, 600.73 examples/s]7770 examples [00:12, 607.24 examples/s]7835 examples [00:12, 618.60 examples/s]7898 examples [00:12, 601.12 examples/s]7959 examples [00:12, 585.60 examples/s]8018 examples [00:12, 572.73 examples/s]8076 examples [00:12, 573.47 examples/s]8135 examples [00:12, 575.81 examples/s]8193 examples [00:12, 576.11 examples/s]8255 examples [00:13, 586.71 examples/s]8315 examples [00:13, 588.51 examples/s]8378 examples [00:13, 599.77 examples/s]8444 examples [00:13, 615.76 examples/s]8506 examples [00:13, 609.28 examples/s]8569 examples [00:13, 613.18 examples/s]8634 examples [00:13, 622.26 examples/s]8700 examples [00:13, 633.10 examples/s]8764 examples [00:13, 631.08 examples/s]8828 examples [00:13, 628.64 examples/s]8891 examples [00:14, 617.57 examples/s]8953 examples [00:14, 581.70 examples/s]9015 examples [00:14, 590.14 examples/s]9075 examples [00:14, 590.72 examples/s]9135 examples [00:14, 573.86 examples/s]9199 examples [00:14, 590.19 examples/s]9259 examples [00:14, 573.91 examples/s]9320 examples [00:14, 583.82 examples/s]9386 examples [00:14, 602.02 examples/s]9447 examples [00:14, 600.80 examples/s]9508 examples [00:15, 588.98 examples/s]9568 examples [00:15, 575.00 examples/s]9626 examples [00:15, 567.99 examples/s]9683 examples [00:15, 549.56 examples/s]9739 examples [00:15, 532.89 examples/s]9796 examples [00:15, 543.24 examples/s]9855 examples [00:15, 556.23 examples/s]9914 examples [00:15, 565.32 examples/s]9976 examples [00:15, 580.02 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 65%|██████▌   | 6523/10000 [00:00<00:00, 65227.26 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSTUMKL/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSTUMKL/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5a047bb0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5a047bb0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5a047bb0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a50199828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f59a00edb70>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5a047bb0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5a047bb0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5a047bb0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f598e942be0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f59ec67f080>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.73 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.73 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.73 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.25 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.25 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.59 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.59 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.71 file/s]2020-08-22 06:11:34.120589: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-22 06:11:34.127178: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-22 06:11:34.127421: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cc098e4540 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-22 06:11:34.127447: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:00, 163061.06it/s] 36%|███▌      | 3530752/9912422 [00:00<00:27, 232477.15it/s]9920512it [00:00, 34033427.77it/s]                           
0it [00:00, ?it/s]32768it [00:00, 510019.20it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 153299.64it/s]1654784it [00:00, 9919504.17it/s]                          
0it [00:00, ?it/s]8192it [00:00, 170218.22it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
