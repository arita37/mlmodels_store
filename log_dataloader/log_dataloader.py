
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f83cd2a9268> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f83cd2a9268> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f8437fcf400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f8437fcf400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f84571ebea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f84571ebea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f83e530b0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f83e530b0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f83e530b0d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 132752.27it/s] 40%|████      | 3973120/9912422 [00:00<00:31, 189364.87it/s] 92%|█████████▏| 9125888/9912422 [00:00<00:02, 270085.82it/s]9920512it [00:00, 21042805.19it/s]                           
0it [00:00, ?it/s]32768it [00:00, 610736.65it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 473019.45it/s]1654784it [00:00, 10873215.69it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191386.10it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83cc520b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83cc52edd8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f83e5303d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f83e5303d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f83e5303d08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:26,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:26,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:25,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:25,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:24,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:24,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:59,  2.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:59,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:58,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:58,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:58,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:57,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:57,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:56,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:56,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:39,  3.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:39,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:39,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:39,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:39,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:38,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:38,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:38,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:37,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:26,  5.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:26,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:26,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:26,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:26,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:25,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:25,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:18,  7.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:18,  7.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:18,  7.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:18,  7.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:17,  7.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:17,  7.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:17,  7.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:17,  7.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:17,  7.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:12,  9.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:12,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:12,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:12,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:12,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:12,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 13.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:08, 13.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 13.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 13.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 13.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 17.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 17.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 17.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 17.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 17.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 17.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 17.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 17.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 22.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 28.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 35.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 35.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 35.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 35.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 35.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 35.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 35.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 35.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 41.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 41.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 41.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 53.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 53.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 53.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 53.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 53.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 53.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 53.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 58.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 61.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 61.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 61.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 61.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 61.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 61.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 61.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 61.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 61.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 70.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 70.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.71s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.71s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.71s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.11 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.85s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.71s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.11 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.85s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.85s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.40 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.85s/ url]
0 examples [00:00, ? examples/s]2020-08-21 18:07:19.026970: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-21 18:07:19.135707: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-08-21 18:07:19.135941: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5627a541ff50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-21 18:07:19.135962: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.34 examples/s]100 examples [00:00,  3.34 examples/s]196 examples [00:00,  4.77 examples/s]287 examples [00:00,  6.79 examples/s]380 examples [00:00,  9.67 examples/s]474 examples [00:00, 13.76 examples/s]573 examples [00:01, 19.54 examples/s]668 examples [00:01, 27.67 examples/s]756 examples [00:01, 38.98 examples/s]847 examples [00:01, 54.69 examples/s]940 examples [00:01, 76.19 examples/s]1033 examples [00:01, 105.13 examples/s]1129 examples [00:01, 143.42 examples/s]1226 examples [00:01, 192.60 examples/s]1323 examples [00:01, 253.50 examples/s]1425 examples [00:01, 326.92 examples/s]1522 examples [00:02, 406.66 examples/s]1618 examples [00:02, 483.02 examples/s]1711 examples [00:02, 562.84 examples/s]1811 examples [00:02, 646.60 examples/s]1908 examples [00:02, 718.44 examples/s]2004 examples [00:02, 777.00 examples/s]2103 examples [00:02, 830.31 examples/s]2200 examples [00:02, 863.40 examples/s]2300 examples [00:02, 897.70 examples/s]2403 examples [00:02, 932.12 examples/s]2506 examples [00:03, 956.82 examples/s]2608 examples [00:03, 972.29 examples/s]2709 examples [00:03, 943.91 examples/s]2806 examples [00:03, 940.49 examples/s]2907 examples [00:03, 958.52 examples/s]3005 examples [00:03, 964.47 examples/s]3103 examples [00:03, 965.75 examples/s]3201 examples [00:03, 939.40 examples/s]3299 examples [00:03, 949.81 examples/s]3399 examples [00:03, 964.20 examples/s]3496 examples [00:04, 953.65 examples/s]3592 examples [00:04, 940.89 examples/s]3687 examples [00:04, 933.85 examples/s]3784 examples [00:04, 943.82 examples/s]3881 examples [00:04, 949.36 examples/s]3979 examples [00:04, 957.61 examples/s]4077 examples [00:04, 962.77 examples/s]4174 examples [00:04, 963.28 examples/s]4271 examples [00:04, 952.17 examples/s]4367 examples [00:05, 942.81 examples/s]4462 examples [00:05, 937.76 examples/s]4556 examples [00:05, 930.49 examples/s]4650 examples [00:05, 914.32 examples/s]4742 examples [00:05, 878.22 examples/s]4831 examples [00:05, 866.72 examples/s]4925 examples [00:05, 885.51 examples/s]5014 examples [00:05, 884.48 examples/s]5108 examples [00:05, 898.54 examples/s]5199 examples [00:05, 883.95 examples/s]5288 examples [00:06, 875.33 examples/s]5380 examples [00:06, 886.02 examples/s]5469 examples [00:06, 881.65 examples/s]5558 examples [00:06, 879.65 examples/s]5647 examples [00:06, 865.93 examples/s]5743 examples [00:06, 891.74 examples/s]5838 examples [00:06, 908.33 examples/s]5933 examples [00:06, 917.25 examples/s]6030 examples [00:06, 929.90 examples/s]6124 examples [00:06, 923.66 examples/s]6219 examples [00:07, 928.35 examples/s]6313 examples [00:07, 931.78 examples/s]6407 examples [00:07, 923.15 examples/s]6500 examples [00:07, 908.59 examples/s]6591 examples [00:07, 891.88 examples/s]6689 examples [00:07, 914.37 examples/s]6782 examples [00:07, 918.60 examples/s]6878 examples [00:07, 930.17 examples/s]6973 examples [00:07, 934.34 examples/s]7068 examples [00:07, 938.91 examples/s]7168 examples [00:08, 954.35 examples/s]7264 examples [00:08, 952.59 examples/s]7360 examples [00:08, 931.55 examples/s]7454 examples [00:08, 924.31 examples/s]7553 examples [00:08, 940.20 examples/s]7650 examples [00:08, 947.54 examples/s]7749 examples [00:08, 959.16 examples/s]7846 examples [00:08, 928.83 examples/s]7940 examples [00:08, 926.80 examples/s]8033 examples [00:09, 915.58 examples/s]8135 examples [00:09, 941.49 examples/s]8230 examples [00:09, 935.56 examples/s]8326 examples [00:09, 942.73 examples/s]8421 examples [00:09, 927.25 examples/s]8518 examples [00:09, 937.71 examples/s]8614 examples [00:09, 943.73 examples/s]8709 examples [00:09, 936.13 examples/s]8803 examples [00:09, 908.93 examples/s]8895 examples [00:09, 908.51 examples/s]8992 examples [00:10, 925.44 examples/s]9096 examples [00:10, 955.89 examples/s]9192 examples [00:10, 949.26 examples/s]9288 examples [00:10, 939.94 examples/s]9383 examples [00:10, 926.14 examples/s]9476 examples [00:10, 925.57 examples/s]9569 examples [00:10, 924.96 examples/s]9666 examples [00:10, 936.81 examples/s]9761 examples [00:10, 938.81 examples/s]9857 examples [00:10, 945.04 examples/s]9952 examples [00:11, 929.27 examples/s]10046 examples [00:11, 822.98 examples/s]10131 examples [00:11, 824.84 examples/s]10216 examples [00:11, 821.28 examples/s]10307 examples [00:11, 845.36 examples/s]10393 examples [00:11, 841.53 examples/s]10484 examples [00:11, 859.14 examples/s]10575 examples [00:11, 871.35 examples/s]10670 examples [00:11, 892.07 examples/s]10768 examples [00:12, 914.66 examples/s]10862 examples [00:12, 920.08 examples/s]10955 examples [00:12, 885.40 examples/s]11051 examples [00:12, 904.89 examples/s]11142 examples [00:12, 900.42 examples/s]11233 examples [00:12, 901.47 examples/s]11324 examples [00:12, 901.37 examples/s]11422 examples [00:12, 922.67 examples/s]11519 examples [00:12, 934.02 examples/s]11613 examples [00:12, 928.39 examples/s]11709 examples [00:13, 936.53 examples/s]11804 examples [00:13, 938.58 examples/s]11898 examples [00:13, 928.66 examples/s]11991 examples [00:13, 921.17 examples/s]12084 examples [00:13, 894.31 examples/s]12174 examples [00:13, 891.54 examples/s]12264 examples [00:13, 877.98 examples/s]12354 examples [00:13, 884.39 examples/s]12447 examples [00:13, 897.05 examples/s]12537 examples [00:13, 897.58 examples/s]12637 examples [00:14, 924.09 examples/s]12734 examples [00:14, 934.06 examples/s]12828 examples [00:14, 923.25 examples/s]12922 examples [00:14, 926.83 examples/s]13015 examples [00:14, 921.78 examples/s]13108 examples [00:14, 907.80 examples/s]13204 examples [00:14, 921.09 examples/s]13299 examples [00:14, 927.49 examples/s]13400 examples [00:14, 949.06 examples/s]13496 examples [00:15, 950.15 examples/s]13592 examples [00:15, 947.87 examples/s]13689 examples [00:15, 952.35 examples/s]13785 examples [00:15, 933.58 examples/s]13879 examples [00:15, 919.12 examples/s]13977 examples [00:15, 934.84 examples/s]14078 examples [00:15, 954.53 examples/s]14177 examples [00:15, 962.72 examples/s]14274 examples [00:15, 958.98 examples/s]14371 examples [00:15, 951.11 examples/s]14467 examples [00:16, 944.88 examples/s]14562 examples [00:16, 944.15 examples/s]14657 examples [00:16, 937.25 examples/s]14751 examples [00:16, 937.63 examples/s]14850 examples [00:16, 952.43 examples/s]14946 examples [00:16, 938.17 examples/s]15045 examples [00:16, 950.59 examples/s]15144 examples [00:16, 959.90 examples/s]15248 examples [00:16, 982.28 examples/s]15352 examples [00:16, 997.11 examples/s]15452 examples [00:17, 979.04 examples/s]15551 examples [00:17, 933.53 examples/s]15651 examples [00:17, 950.84 examples/s]15747 examples [00:17, 909.89 examples/s]15841 examples [00:17, 913.79 examples/s]15935 examples [00:17, 920.16 examples/s]16030 examples [00:17, 925.53 examples/s]16127 examples [00:17, 935.88 examples/s]16223 examples [00:17, 939.94 examples/s]16318 examples [00:17, 941.46 examples/s]16413 examples [00:18, 934.47 examples/s]16507 examples [00:18, 928.98 examples/s]16600 examples [00:18, 915.69 examples/s]16693 examples [00:18, 919.91 examples/s]16786 examples [00:18, 897.11 examples/s]16876 examples [00:18, 873.10 examples/s]16965 examples [00:18, 875.80 examples/s]17056 examples [00:18, 884.73 examples/s]17148 examples [00:18, 893.80 examples/s]17238 examples [00:19, 872.99 examples/s]17326 examples [00:19, 874.64 examples/s]17415 examples [00:19, 878.30 examples/s]17507 examples [00:19, 889.34 examples/s]17601 examples [00:19, 903.56 examples/s]17693 examples [00:19, 905.44 examples/s]17786 examples [00:19, 909.24 examples/s]17879 examples [00:19, 913.78 examples/s]17972 examples [00:19, 918.38 examples/s]18065 examples [00:19, 920.33 examples/s]18158 examples [00:20, 912.58 examples/s]18250 examples [00:20, 905.04 examples/s]18343 examples [00:20, 910.57 examples/s]18438 examples [00:20, 921.26 examples/s]18531 examples [00:20, 915.70 examples/s]18623 examples [00:20, 913.07 examples/s]18719 examples [00:20, 923.81 examples/s]18820 examples [00:20, 945.97 examples/s]18927 examples [00:20, 978.69 examples/s]19029 examples [00:20, 988.20 examples/s]19129 examples [00:21, 958.53 examples/s]19226 examples [00:21, 951.02 examples/s]19323 examples [00:21, 955.18 examples/s]19425 examples [00:21, 970.98 examples/s]19523 examples [00:21, 949.88 examples/s]19619 examples [00:21, 948.76 examples/s]19718 examples [00:21, 960.39 examples/s]19819 examples [00:21, 972.56 examples/s]19922 examples [00:21, 987.70 examples/s]20021 examples [00:22, 921.90 examples/s]20116 examples [00:22, 927.49 examples/s]20210 examples [00:22, 926.37 examples/s]20304 examples [00:22, 922.85 examples/s]20403 examples [00:22, 940.04 examples/s]20506 examples [00:22, 963.72 examples/s]20603 examples [00:22, 955.43 examples/s]20699 examples [00:22, 951.79 examples/s]20795 examples [00:22, 947.74 examples/s]20893 examples [00:22, 955.19 examples/s]20992 examples [00:23, 964.11 examples/s]21089 examples [00:23, 942.38 examples/s]21184 examples [00:23, 932.06 examples/s]21283 examples [00:23, 945.88 examples/s]21380 examples [00:23, 951.71 examples/s]21476 examples [00:23, 945.40 examples/s]21571 examples [00:23, 901.38 examples/s]21662 examples [00:23, 886.70 examples/s]21756 examples [00:23, 900.21 examples/s]21851 examples [00:23, 912.50 examples/s]21943 examples [00:24, 904.98 examples/s]22034 examples [00:24, 891.25 examples/s]22124 examples [00:24, 884.89 examples/s]22223 examples [00:24, 912.80 examples/s]22323 examples [00:24, 935.98 examples/s]22420 examples [00:24, 944.31 examples/s]22515 examples [00:24, 943.87 examples/s]22620 examples [00:24, 972.40 examples/s]22722 examples [00:24, 985.84 examples/s]22823 examples [00:24, 992.47 examples/s]22925 examples [00:25, 997.71 examples/s]23025 examples [00:25, 952.60 examples/s]23121 examples [00:25, 949.10 examples/s]23220 examples [00:25, 960.49 examples/s]23320 examples [00:25, 969.29 examples/s]23418 examples [00:25, 962.16 examples/s]23515 examples [00:25, 957.57 examples/s]23611 examples [00:25, 942.62 examples/s]23706 examples [00:25, 929.97 examples/s]23801 examples [00:26, 933.24 examples/s]23895 examples [00:26, 909.45 examples/s]23987 examples [00:26, 900.18 examples/s]24080 examples [00:26, 906.61 examples/s]24171 examples [00:26, 907.26 examples/s]24263 examples [00:26, 909.00 examples/s]24354 examples [00:26, 892.78 examples/s]24444 examples [00:26, 893.73 examples/s]24536 examples [00:26, 898.43 examples/s]24627 examples [00:26, 901.87 examples/s]24718 examples [00:27, 882.20 examples/s]24808 examples [00:27, 885.19 examples/s]24897 examples [00:27, 872.66 examples/s]24987 examples [00:27, 878.33 examples/s]25079 examples [00:27, 887.11 examples/s]25173 examples [00:27, 900.41 examples/s]25264 examples [00:27, 900.88 examples/s]25355 examples [00:27, 899.21 examples/s]25445 examples [00:27, 899.26 examples/s]25537 examples [00:27, 905.06 examples/s]25631 examples [00:28, 913.45 examples/s]25724 examples [00:28, 914.67 examples/s]25816 examples [00:28, 913.40 examples/s]25908 examples [00:28, 909.48 examples/s]26004 examples [00:28, 923.19 examples/s]26100 examples [00:28, 933.93 examples/s]26194 examples [00:28, 927.50 examples/s]26287 examples [00:28, 927.89 examples/s]26382 examples [00:28, 933.91 examples/s]26480 examples [00:28, 945.20 examples/s]26575 examples [00:29, 927.31 examples/s]26670 examples [00:29, 933.14 examples/s]26764 examples [00:29, 921.21 examples/s]26859 examples [00:29, 927.73 examples/s]26953 examples [00:29, 929.52 examples/s]27047 examples [00:29, 929.96 examples/s]27141 examples [00:29, 930.88 examples/s]27235 examples [00:29, 929.09 examples/s]27328 examples [00:29, 926.96 examples/s]27430 examples [00:29, 952.33 examples/s]27532 examples [00:30, 969.21 examples/s]27630 examples [00:30, 970.18 examples/s]27728 examples [00:30, 946.57 examples/s]27826 examples [00:30, 953.49 examples/s]27924 examples [00:30, 960.85 examples/s]28025 examples [00:30, 973.38 examples/s]28123 examples [00:30, 957.74 examples/s]28219 examples [00:30, 933.15 examples/s]28317 examples [00:30, 945.03 examples/s]28418 examples [00:31, 961.12 examples/s]28516 examples [00:31, 966.56 examples/s]28613 examples [00:31, 950.26 examples/s]28709 examples [00:31, 911.31 examples/s]28805 examples [00:31, 925.35 examples/s]28898 examples [00:31, 923.31 examples/s]28999 examples [00:31, 947.51 examples/s]29095 examples [00:31, 943.96 examples/s]29193 examples [00:31, 951.96 examples/s]29296 examples [00:31, 971.97 examples/s]29399 examples [00:32, 988.06 examples/s]29499 examples [00:32, 986.66 examples/s]29598 examples [00:32, 944.57 examples/s]29693 examples [00:32, 944.27 examples/s]29797 examples [00:32, 968.81 examples/s]29905 examples [00:32, 998.84 examples/s]30006 examples [00:32, 957.00 examples/s]30103 examples [00:32, 952.04 examples/s]30199 examples [00:32, 949.19 examples/s]30300 examples [00:32, 966.03 examples/s]30399 examples [00:33, 970.22 examples/s]30498 examples [00:33, 975.15 examples/s]30596 examples [00:33, 968.81 examples/s]30694 examples [00:33, 964.90 examples/s]30793 examples [00:33, 971.00 examples/s]30891 examples [00:33, 966.20 examples/s]30989 examples [00:33, 968.57 examples/s]31086 examples [00:33, 947.69 examples/s]31181 examples [00:33, 948.21 examples/s]31279 examples [00:34, 956.16 examples/s]31384 examples [00:34, 981.67 examples/s]31484 examples [00:34, 985.75 examples/s]31583 examples [00:34, 967.92 examples/s]31680 examples [00:34, 952.15 examples/s]31779 examples [00:34, 960.87 examples/s]31877 examples [00:34, 964.70 examples/s]31976 examples [00:34, 972.15 examples/s]32074 examples [00:34, 965.46 examples/s]32171 examples [00:34, 939.97 examples/s]32273 examples [00:35, 960.46 examples/s]32371 examples [00:35, 965.50 examples/s]32471 examples [00:35, 973.38 examples/s]32569 examples [00:35, 964.06 examples/s]32670 examples [00:35, 974.92 examples/s]32773 examples [00:35, 990.21 examples/s]32873 examples [00:35, 973.61 examples/s]32975 examples [00:35, 985.53 examples/s]33078 examples [00:35, 997.60 examples/s]33178 examples [00:35, 997.61 examples/s]33278 examples [00:36, 983.26 examples/s]33380 examples [00:36, 993.43 examples/s]33480 examples [00:36, 991.94 examples/s]33582 examples [00:36, 998.36 examples/s]33682 examples [00:36, 997.45 examples/s]33782 examples [00:36, 996.10 examples/s]33882 examples [00:36, 995.64 examples/s]33986 examples [00:36, 1007.25 examples/s]34088 examples [00:36, 1009.94 examples/s]34195 examples [00:36, 1025.07 examples/s]34298 examples [00:37, 993.78 examples/s] 34398 examples [00:37, 975.72 examples/s]34501 examples [00:37, 990.55 examples/s]34601 examples [00:37, 970.25 examples/s]34699 examples [00:37, 970.23 examples/s]34806 examples [00:37, 995.86 examples/s]34910 examples [00:37, 1008.39 examples/s]35012 examples [00:37, 1009.99 examples/s]35114 examples [00:37, 988.69 examples/s] 35214 examples [00:38, 973.14 examples/s]35312 examples [00:38, 960.03 examples/s]35409 examples [00:38, 953.70 examples/s]35513 examples [00:38, 977.00 examples/s]35611 examples [00:38, 977.49 examples/s]35709 examples [00:38, 974.11 examples/s]35813 examples [00:38, 990.52 examples/s]35913 examples [00:38, 953.50 examples/s]36009 examples [00:38, 925.14 examples/s]36107 examples [00:38, 938.87 examples/s]36202 examples [00:39, 929.42 examples/s]36302 examples [00:39, 946.81 examples/s]36397 examples [00:39, 946.91 examples/s]36495 examples [00:39, 955.20 examples/s]36591 examples [00:39, 949.47 examples/s]36693 examples [00:39, 967.39 examples/s]36794 examples [00:39, 978.28 examples/s]36893 examples [00:39, 979.75 examples/s]36992 examples [00:39, 976.99 examples/s]37092 examples [00:39, 982.35 examples/s]37194 examples [00:40, 990.59 examples/s]37298 examples [00:40, 1003.52 examples/s]37399 examples [00:40, 989.20 examples/s] 37499 examples [00:40, 991.42 examples/s]37599 examples [00:40, 976.18 examples/s]37697 examples [00:40, 969.85 examples/s]37796 examples [00:40, 974.63 examples/s]37899 examples [00:40, 988.09 examples/s]38004 examples [00:40, 1005.74 examples/s]38105 examples [00:40, 992.68 examples/s] 38207 examples [00:41, 1000.63 examples/s]38308 examples [00:41, 990.07 examples/s] 38408 examples [00:41, 987.15 examples/s]38507 examples [00:41, 935.31 examples/s]38602 examples [00:41, 932.59 examples/s]38702 examples [00:41, 950.91 examples/s]38812 examples [00:41, 989.01 examples/s]38912 examples [00:41, 954.95 examples/s]39009 examples [00:41, 945.25 examples/s]39110 examples [00:42, 963.66 examples/s]39209 examples [00:42, 969.17 examples/s]39313 examples [00:42, 987.07 examples/s]39413 examples [00:42, 971.83 examples/s]39511 examples [00:42, 967.56 examples/s]39613 examples [00:42, 981.97 examples/s]39714 examples [00:42, 989.02 examples/s]39814 examples [00:42, 987.94 examples/s]39914 examples [00:42, 991.26 examples/s]40014 examples [00:42, 905.23 examples/s]40118 examples [00:43, 939.86 examples/s]40218 examples [00:43, 956.71 examples/s]40317 examples [00:43, 965.07 examples/s]40416 examples [00:43, 971.77 examples/s]40514 examples [00:43, 966.68 examples/s]40619 examples [00:43, 988.66 examples/s]40724 examples [00:43, 1003.81 examples/s]40825 examples [00:43, 986.24 examples/s] 40924 examples [00:43, 971.34 examples/s]41022 examples [00:43, 959.92 examples/s]41121 examples [00:44, 967.70 examples/s]41221 examples [00:44, 976.69 examples/s]41326 examples [00:44, 997.21 examples/s]41428 examples [00:44, 1001.63 examples/s]41529 examples [00:44, 998.45 examples/s] 41629 examples [00:44, 983.20 examples/s]41730 examples [00:44, 989.87 examples/s]41834 examples [00:44, 1004.12 examples/s]41935 examples [00:44, 981.58 examples/s] 42034 examples [00:45, 942.58 examples/s]42137 examples [00:45, 965.26 examples/s]42239 examples [00:45, 979.44 examples/s]42338 examples [00:45, 959.47 examples/s]42439 examples [00:45, 973.84 examples/s]42537 examples [00:45, 963.27 examples/s]42638 examples [00:45, 975.17 examples/s]42736 examples [00:45, 951.67 examples/s]42832 examples [00:45, 941.10 examples/s]42930 examples [00:45, 951.90 examples/s]43026 examples [00:46, 950.70 examples/s]43123 examples [00:46, 955.34 examples/s]43224 examples [00:46, 969.97 examples/s]43322 examples [00:46, 971.42 examples/s]43420 examples [00:46, 973.34 examples/s]43518 examples [00:46, 934.34 examples/s]43613 examples [00:46, 938.92 examples/s]43709 examples [00:46, 945.05 examples/s]43804 examples [00:46, 942.17 examples/s]43906 examples [00:46, 963.54 examples/s]44003 examples [00:47, 954.03 examples/s]44099 examples [00:47, 951.76 examples/s]44195 examples [00:47, 931.69 examples/s]44289 examples [00:47, 931.45 examples/s]44384 examples [00:47, 936.70 examples/s]44483 examples [00:47, 950.47 examples/s]44592 examples [00:47, 986.54 examples/s]44692 examples [00:47, 978.19 examples/s]44791 examples [00:47, 956.26 examples/s]44891 examples [00:48, 968.84 examples/s]44989 examples [00:48, 958.66 examples/s]45090 examples [00:48, 972.90 examples/s]45190 examples [00:48, 980.43 examples/s]45290 examples [00:48, 984.58 examples/s]45389 examples [00:48, 979.70 examples/s]45489 examples [00:48, 985.03 examples/s]45594 examples [00:48, 1001.47 examples/s]45695 examples [00:48, 979.33 examples/s] 45797 examples [00:48, 990.66 examples/s]45898 examples [00:49, 994.73 examples/s]45998 examples [00:49, 973.83 examples/s]46099 examples [00:49, 982.17 examples/s]46198 examples [00:49, 984.35 examples/s]46302 examples [00:49, 998.86 examples/s]46410 examples [00:49, 1020.31 examples/s]46513 examples [00:49, 1011.96 examples/s]46618 examples [00:49, 1022.84 examples/s]46721 examples [00:49, 995.52 examples/s] 46822 examples [00:49, 997.63 examples/s]46925 examples [00:50, 1006.99 examples/s]47026 examples [00:50, 1006.94 examples/s]47131 examples [00:50, 1018.15 examples/s]47238 examples [00:50, 1033.10 examples/s]47344 examples [00:50, 1040.35 examples/s]47449 examples [00:50, 1019.47 examples/s]47555 examples [00:50, 1030.36 examples/s]47659 examples [00:50, 1027.88 examples/s]47763 examples [00:50, 1030.81 examples/s]47867 examples [00:50, 1006.49 examples/s]47968 examples [00:51, 1005.56 examples/s]48072 examples [00:51, 1012.55 examples/s]48174 examples [00:51, 1006.02 examples/s]48277 examples [00:51, 1010.97 examples/s]48379 examples [00:51, 1005.11 examples/s]48480 examples [00:51, 979.70 examples/s] 48584 examples [00:51, 996.57 examples/s]48684 examples [00:51, 993.04 examples/s]48788 examples [00:51, 1004.69 examples/s]48889 examples [00:51, 1005.57 examples/s]48990 examples [00:52, 941.92 examples/s] 49095 examples [00:52, 971.07 examples/s]49200 examples [00:52, 993.37 examples/s]49301 examples [00:52, 983.44 examples/s]49400 examples [00:52, 975.64 examples/s]49498 examples [00:52, 964.56 examples/s]49601 examples [00:52, 981.43 examples/s]49704 examples [00:52, 994.18 examples/s]49804 examples [00:52, 975.37 examples/s]49909 examples [00:53, 994.89 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 17%|█▋        | 8266/50000 [00:00<00:00, 82656.14 examples/s] 42%|████▎     | 21250/50000 [00:00<00:00, 92769.41 examples/s] 67%|██████▋   | 33546/50000 [00:00<00:00, 100145.34 examples/s] 93%|█████████▎| 46644/50000 [00:00<00:00, 107754.27 examples/s]                                                                0 examples [00:00, ? examples/s]83 examples [00:00, 828.71 examples/s]191 examples [00:00, 890.19 examples/s]290 examples [00:00, 916.58 examples/s]389 examples [00:00, 935.40 examples/s]489 examples [00:00, 953.83 examples/s]581 examples [00:00, 941.87 examples/s]677 examples [00:00, 944.50 examples/s]780 examples [00:00, 967.20 examples/s]882 examples [00:00, 979.96 examples/s]977 examples [00:01, 970.72 examples/s]1074 examples [00:01, 968.48 examples/s]1176 examples [00:01, 981.83 examples/s]1279 examples [00:01, 994.82 examples/s]1384 examples [00:01, 1010.47 examples/s]1485 examples [00:01, 1005.39 examples/s]1586 examples [00:01, 983.37 examples/s] 1687 examples [00:01, 989.45 examples/s]1790 examples [00:01, 1000.85 examples/s]1897 examples [00:01, 1019.15 examples/s]1999 examples [00:02, 995.79 examples/s] 2099 examples [00:02, 980.89 examples/s]2199 examples [00:02, 985.09 examples/s]2310 examples [00:02, 1017.37 examples/s]2415 examples [00:02, 1026.64 examples/s]2518 examples [00:02, 1006.51 examples/s]2619 examples [00:02, 992.37 examples/s] 2719 examples [00:02, 991.88 examples/s]2822 examples [00:02, 1000.59 examples/s]2923 examples [00:02, 999.88 examples/s] 3024 examples [00:03, 992.32 examples/s]3124 examples [00:03, 973.38 examples/s]3222 examples [00:03, 972.14 examples/s]3320 examples [00:03, 972.78 examples/s]3420 examples [00:03, 979.11 examples/s]3520 examples [00:03, 985.06 examples/s]3619 examples [00:03, 979.98 examples/s]3718 examples [00:03, 976.33 examples/s]3817 examples [00:03, 979.73 examples/s]3917 examples [00:03, 983.42 examples/s]4019 examples [00:04, 991.85 examples/s]4119 examples [00:04, 963.35 examples/s]4219 examples [00:04, 971.52 examples/s]4318 examples [00:04, 976.87 examples/s]4418 examples [00:04, 981.50 examples/s]4525 examples [00:04, 1004.57 examples/s]4626 examples [00:04, 978.37 examples/s] 4727 examples [00:04, 986.84 examples/s]4826 examples [00:04, 818.20 examples/s]4923 examples [00:05, 857.17 examples/s]5021 examples [00:05, 889.63 examples/s]5122 examples [00:05, 920.94 examples/s]5223 examples [00:05, 944.39 examples/s]5320 examples [00:05, 951.62 examples/s]5417 examples [00:05, 946.39 examples/s]5513 examples [00:05, 923.33 examples/s]5607 examples [00:05, 917.18 examples/s]5700 examples [00:05, 917.59 examples/s]5793 examples [00:05, 909.66 examples/s]5891 examples [00:06, 929.28 examples/s]5985 examples [00:06, 927.28 examples/s]6078 examples [00:06, 925.34 examples/s]6171 examples [00:06, 924.04 examples/s]6266 examples [00:06, 931.66 examples/s]6362 examples [00:06, 939.68 examples/s]6457 examples [00:06, 909.92 examples/s]6553 examples [00:06, 922.59 examples/s]6652 examples [00:06, 940.67 examples/s]6748 examples [00:07, 945.67 examples/s]6845 examples [00:07, 952.45 examples/s]6941 examples [00:07, 939.47 examples/s]7036 examples [00:07, 926.32 examples/s]7130 examples [00:07, 929.43 examples/s]7224 examples [00:07, 924.90 examples/s]7317 examples [00:07, 912.47 examples/s]7409 examples [00:07, 898.70 examples/s]7504 examples [00:07, 911.42 examples/s]7600 examples [00:07, 924.98 examples/s]7700 examples [00:08, 943.91 examples/s]7800 examples [00:08, 958.94 examples/s]7898 examples [00:08, 962.44 examples/s]8001 examples [00:08, 979.38 examples/s]8104 examples [00:08, 993.15 examples/s]8209 examples [00:08, 1006.41 examples/s]8310 examples [00:08, 990.16 examples/s] 8410 examples [00:08, 965.84 examples/s]8517 examples [00:08, 994.35 examples/s]8625 examples [00:08, 1015.98 examples/s]8728 examples [00:09, 1017.88 examples/s]8831 examples [00:09, 986.15 examples/s] 8931 examples [00:09, 971.99 examples/s]9033 examples [00:09, 984.67 examples/s]9140 examples [00:09, 1006.35 examples/s]9241 examples [00:09, 967.21 examples/s] 9339 examples [00:09, 956.29 examples/s]9436 examples [00:09, 957.14 examples/s]9537 examples [00:09, 971.19 examples/s]9635 examples [00:09, 967.28 examples/s]9742 examples [00:10, 995.05 examples/s]9842 examples [00:10, 979.48 examples/s]9941 examples [00:10, 943.72 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYPUXDM/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYPUXDM/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f83e530b0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f83e530b0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f83e530b0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8430ce9eb8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f838034b240>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f83e530b0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f83e530b0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f83e530b0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83cc54f0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83cc54f128>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.13 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 20.13 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.57 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.57 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.09 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.09 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.64 file/s]2020-08-21 18:08:30.177299: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-21 18:08:30.181798: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-08-21 18:08:30.182058: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56135b7983d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-21 18:08:30.182079: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 137577.36it/s] 67%|██████▋   | 6602752/9912422 [00:00<00:16, 196361.96it/s]9920512it [00:00, 39178278.39it/s]                           
0it [00:00, ?it/s]32768it [00:00, 524316.00it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 450028.33it/s]1654784it [00:00, 11355320.54it/s]                         
0it [00:00, ?it/s]8192it [00:00, 219875.59it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
