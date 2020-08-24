
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f4ced3411e0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f4ced3411e0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f4d58067400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f4d58067400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f4d77283ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f4d77283ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4d053a40d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4d053a40d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4d053a40d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3686400/9912422 [00:00<00:00, 36860959.24it/s]9920512it [00:00, 32319098.23it/s]                             
0it [00:00, ?it/s]32768it [00:00, 529752.36it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 154372.37it/s]1654784it [00:00, 10492379.62it/s]                         
0it [00:00, ?it/s]8192it [00:00, 165160.88it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4ced253b70>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4ced245e10>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f4d0539bd08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f4d0539bd08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f4d0539bd08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:47,  1.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:46,  1.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:45,  1.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:45,  1.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:44,  1.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:43,  1.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:12,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:11,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:11,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:10,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:10,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:10,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:09,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:09,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:08,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:48,  2.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:48,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:47,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:47,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:47,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:46,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:46,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:46,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:45,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:45,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:45,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:31,  4.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:30,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:30,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:30,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:30,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:29,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:29,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  5.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:20,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:20,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:20,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:20,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:20,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:19,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:19,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:19,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:19,  5.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:13,  8.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:12,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:12,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 11.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:09, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:09, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:08, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:08, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:08, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:08, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:08, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:08, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 15.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 15.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 15.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 15.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 15.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 15.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 15.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 15.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 15.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:05, 15.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 20.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 20.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 26.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 26.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 34.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 34.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 42.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 42.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 50.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 50.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 58.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 58.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 66.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.41s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.41s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.41s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.84 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.52s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.41s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 78.84 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.52s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.52s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.82 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.52s/ url]
0 examples [00:00, ? examples/s]2020-08-24 00:07:59.374770: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-24 00:07:59.388224: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095199999 Hz
2020-08-24 00:07:59.388480: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fe52889c70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-24 00:07:59.388500: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.89 examples/s]104 examples [00:00,  4.12 examples/s]215 examples [00:00,  5.88 examples/s]326 examples [00:00,  8.38 examples/s]435 examples [00:00, 11.93 examples/s]545 examples [00:00, 16.97 examples/s]656 examples [00:00, 24.08 examples/s]765 examples [00:01, 34.08 examples/s]867 examples [00:01, 48.00 examples/s]972 examples [00:01, 67.25 examples/s]1081 examples [00:01, 93.59 examples/s]1185 examples [00:01, 128.45 examples/s]1292 examples [00:01, 174.49 examples/s]1403 examples [00:01, 233.49 examples/s]1514 examples [00:01, 305.94 examples/s]1622 examples [00:01, 389.72 examples/s]1733 examples [00:01, 483.45 examples/s]1843 examples [00:02, 581.17 examples/s]1955 examples [00:02, 678.96 examples/s]2066 examples [00:02, 768.01 examples/s]2176 examples [00:02, 832.16 examples/s]2287 examples [00:02, 897.80 examples/s]2397 examples [00:02, 948.34 examples/s]2506 examples [00:02, 965.61 examples/s]2617 examples [00:02, 1004.25 examples/s]2727 examples [00:02, 1030.97 examples/s]2836 examples [00:02, 1017.29 examples/s]2947 examples [00:03, 1041.18 examples/s]3059 examples [00:03, 1062.48 examples/s]3168 examples [00:03, 1068.20 examples/s]3277 examples [00:03, 1051.47 examples/s]3384 examples [00:03, 1050.45 examples/s]3493 examples [00:03, 1059.38 examples/s]3602 examples [00:03, 1067.08 examples/s]3713 examples [00:03, 1079.59 examples/s]3825 examples [00:03, 1088.89 examples/s]3935 examples [00:03, 1088.10 examples/s]4045 examples [00:04, 1088.88 examples/s]4156 examples [00:04, 1093.54 examples/s]4267 examples [00:04, 1097.42 examples/s]4377 examples [00:04, 1094.06 examples/s]4487 examples [00:04, 1095.19 examples/s]4597 examples [00:04, 1089.97 examples/s]4707 examples [00:04, 1065.18 examples/s]4816 examples [00:04, 1072.24 examples/s]4928 examples [00:04, 1083.89 examples/s]5038 examples [00:05, 1088.28 examples/s]5147 examples [00:05, 1078.29 examples/s]5255 examples [00:05, 1073.56 examples/s]5363 examples [00:05, 1064.15 examples/s]5470 examples [00:05, 1055.42 examples/s]5577 examples [00:05, 1059.08 examples/s]5686 examples [00:05, 1065.91 examples/s]5797 examples [00:05, 1078.55 examples/s]5910 examples [00:05, 1091.63 examples/s]6020 examples [00:05, 1083.78 examples/s]6131 examples [00:06, 1089.80 examples/s]6243 examples [00:06, 1097.92 examples/s]6353 examples [00:06, 1098.43 examples/s]6463 examples [00:06, 1097.53 examples/s]6575 examples [00:06, 1102.97 examples/s]6686 examples [00:06, 1100.19 examples/s]6799 examples [00:06, 1106.18 examples/s]6910 examples [00:06, 1101.83 examples/s]7021 examples [00:06, 1082.05 examples/s]7130 examples [00:06, 1076.75 examples/s]7238 examples [00:07, 1070.10 examples/s]7347 examples [00:07, 1075.88 examples/s]7456 examples [00:07, 1079.80 examples/s]7565 examples [00:07, 1068.20 examples/s]7672 examples [00:07, 1056.76 examples/s]7778 examples [00:07, 1050.90 examples/s]7884 examples [00:07, 1050.48 examples/s]7990 examples [00:07, 1050.80 examples/s]8098 examples [00:07, 1058.17 examples/s]8205 examples [00:07, 1060.04 examples/s]8313 examples [00:08, 1063.41 examples/s]8422 examples [00:08, 1069.73 examples/s]8532 examples [00:08, 1077.29 examples/s]8640 examples [00:08, 1065.05 examples/s]8747 examples [00:08, 1052.24 examples/s]8854 examples [00:08, 1056.73 examples/s]8960 examples [00:08, 1051.75 examples/s]9070 examples [00:08, 1064.10 examples/s]9177 examples [00:08, 1062.17 examples/s]9284 examples [00:08, 1061.31 examples/s]9392 examples [00:09, 1066.50 examples/s]9499 examples [00:09, 1067.20 examples/s]9606 examples [00:09, 1064.48 examples/s]9713 examples [00:09, 1044.04 examples/s]9818 examples [00:09, 1042.45 examples/s]9923 examples [00:09, 1038.75 examples/s]10027 examples [00:09, 994.63 examples/s]10131 examples [00:09, 1005.23 examples/s]10240 examples [00:09, 1027.13 examples/s]10344 examples [00:10, 1012.64 examples/s]10448 examples [00:10, 1018.78 examples/s]10555 examples [00:10, 1032.09 examples/s]10667 examples [00:10, 1055.05 examples/s]10776 examples [00:10, 1063.58 examples/s]10885 examples [00:10, 1069.48 examples/s]10993 examples [00:10, 1065.29 examples/s]11102 examples [00:10, 1069.90 examples/s]11213 examples [00:10, 1079.38 examples/s]11323 examples [00:10, 1083.39 examples/s]11433 examples [00:11, 1087.78 examples/s]11542 examples [00:11, 1081.84 examples/s]11651 examples [00:11, 1080.74 examples/s]11761 examples [00:11, 1085.32 examples/s]11871 examples [00:11, 1087.84 examples/s]11980 examples [00:11, 1079.16 examples/s]12090 examples [00:11, 1083.29 examples/s]12199 examples [00:11, 1083.22 examples/s]12308 examples [00:11, 1080.53 examples/s]12417 examples [00:11, 1069.75 examples/s]12525 examples [00:12, 1066.87 examples/s]12635 examples [00:12, 1075.94 examples/s]12746 examples [00:12, 1085.35 examples/s]12857 examples [00:12, 1090.96 examples/s]12968 examples [00:12, 1095.62 examples/s]13078 examples [00:12, 1089.88 examples/s]13188 examples [00:12, 1087.97 examples/s]13298 examples [00:12, 1089.38 examples/s]13409 examples [00:12, 1093.59 examples/s]13520 examples [00:12, 1098.36 examples/s]13632 examples [00:13, 1102.01 examples/s]13743 examples [00:13, 1096.71 examples/s]13853 examples [00:13, 1092.65 examples/s]13964 examples [00:13, 1097.48 examples/s]14076 examples [00:13, 1101.86 examples/s]14187 examples [00:13, 1087.76 examples/s]14296 examples [00:13, 1087.50 examples/s]14406 examples [00:13, 1088.89 examples/s]14517 examples [00:13, 1094.63 examples/s]14627 examples [00:13, 1091.96 examples/s]14737 examples [00:14, 1071.28 examples/s]14847 examples [00:14, 1078.77 examples/s]14957 examples [00:14, 1084.90 examples/s]15069 examples [00:14, 1092.67 examples/s]15179 examples [00:14, 1089.94 examples/s]15291 examples [00:14, 1096.00 examples/s]15402 examples [00:14, 1099.67 examples/s]15512 examples [00:14, 1093.42 examples/s]15623 examples [00:14, 1095.42 examples/s]15734 examples [00:14, 1098.90 examples/s]15845 examples [00:15, 1101.95 examples/s]15956 examples [00:15, 1098.86 examples/s]16066 examples [00:15, 1095.05 examples/s]16176 examples [00:15, 1095.64 examples/s]16287 examples [00:15, 1097.17 examples/s]16399 examples [00:15, 1101.31 examples/s]16510 examples [00:15, 1098.24 examples/s]16620 examples [00:15, 1094.71 examples/s]16731 examples [00:15, 1098.69 examples/s]16842 examples [00:15, 1099.54 examples/s]16952 examples [00:16, 1095.78 examples/s]17064 examples [00:16, 1100.45 examples/s]17175 examples [00:16, 1098.20 examples/s]17286 examples [00:16, 1098.71 examples/s]17397 examples [00:16, 1099.65 examples/s]17507 examples [00:16, 1096.92 examples/s]17617 examples [00:16, 1094.28 examples/s]17727 examples [00:16, 1093.63 examples/s]17838 examples [00:16, 1096.57 examples/s]17948 examples [00:16, 1095.00 examples/s]18059 examples [00:17, 1096.58 examples/s]18169 examples [00:17, 1097.55 examples/s]18279 examples [00:17, 1096.32 examples/s]18389 examples [00:17, 1096.93 examples/s]18500 examples [00:17, 1098.38 examples/s]18610 examples [00:17, 1095.47 examples/s]18720 examples [00:17, 1094.93 examples/s]18830 examples [00:17, 1092.83 examples/s]18941 examples [00:17, 1096.14 examples/s]19051 examples [00:17, 1083.47 examples/s]19162 examples [00:18, 1089.96 examples/s]19272 examples [00:18, 1074.09 examples/s]19380 examples [00:18, 1070.44 examples/s]19491 examples [00:18, 1079.75 examples/s]19600 examples [00:18, 1078.96 examples/s]19711 examples [00:18, 1087.19 examples/s]19821 examples [00:18, 1090.94 examples/s]19931 examples [00:18, 1089.09 examples/s]20040 examples [00:18, 1040.14 examples/s]20152 examples [00:19, 1061.61 examples/s]20264 examples [00:19, 1076.78 examples/s]20376 examples [00:19, 1086.89 examples/s]20486 examples [00:19, 1088.25 examples/s]20597 examples [00:19, 1094.62 examples/s]20707 examples [00:19, 1095.57 examples/s]20817 examples [00:19, 1080.43 examples/s]20927 examples [00:19, 1086.16 examples/s]21036 examples [00:19, 1081.48 examples/s]21147 examples [00:19, 1088.87 examples/s]21258 examples [00:20, 1093.47 examples/s]21369 examples [00:20, 1097.82 examples/s]21479 examples [00:20, 1082.24 examples/s]21588 examples [00:20, 1075.73 examples/s]21699 examples [00:20, 1085.43 examples/s]21810 examples [00:20, 1092.29 examples/s]21921 examples [00:20, 1096.12 examples/s]22031 examples [00:20, 1063.30 examples/s]22141 examples [00:20, 1071.46 examples/s]22249 examples [00:20, 1063.77 examples/s]22362 examples [00:21, 1080.85 examples/s]22475 examples [00:21, 1092.81 examples/s]22588 examples [00:21, 1101.07 examples/s]22700 examples [00:21, 1104.07 examples/s]22813 examples [00:21, 1109.90 examples/s]22925 examples [00:21, 1112.40 examples/s]23037 examples [00:21, 1111.64 examples/s]23149 examples [00:21, 1111.82 examples/s]23261 examples [00:21, 1099.95 examples/s]23373 examples [00:21, 1105.34 examples/s]23486 examples [00:22, 1109.83 examples/s]23598 examples [00:22, 1112.26 examples/s]23711 examples [00:22, 1115.69 examples/s]23823 examples [00:22, 1109.76 examples/s]23934 examples [00:22, 1088.82 examples/s]24046 examples [00:22, 1097.78 examples/s]24158 examples [00:22, 1102.19 examples/s]24269 examples [00:22, 1097.74 examples/s]24382 examples [00:22, 1104.73 examples/s]24493 examples [00:22, 1106.02 examples/s]24606 examples [00:23, 1110.45 examples/s]24719 examples [00:23, 1113.71 examples/s]24831 examples [00:23, 1114.33 examples/s]24943 examples [00:23, 1088.75 examples/s]25054 examples [00:23, 1093.09 examples/s]25164 examples [00:23, 1092.93 examples/s]25274 examples [00:23, 1089.70 examples/s]25386 examples [00:23, 1096.25 examples/s]25496 examples [00:23, 1078.50 examples/s]25604 examples [00:24, 1045.94 examples/s]25711 examples [00:24, 1050.91 examples/s]25819 examples [00:24, 1057.54 examples/s]25927 examples [00:24, 1062.40 examples/s]26037 examples [00:24, 1071.39 examples/s]26147 examples [00:24, 1079.78 examples/s]26256 examples [00:24, 1057.84 examples/s]26366 examples [00:24, 1067.55 examples/s]26474 examples [00:24, 1070.05 examples/s]26582 examples [00:24, 1067.60 examples/s]26689 examples [00:25, 1068.09 examples/s]26797 examples [00:25, 1069.62 examples/s]26906 examples [00:25, 1074.08 examples/s]27016 examples [00:25, 1079.91 examples/s]27125 examples [00:25, 1079.70 examples/s]27235 examples [00:25, 1083.13 examples/s]27344 examples [00:25, 1067.98 examples/s]27451 examples [00:25, 1067.99 examples/s]27562 examples [00:25, 1077.84 examples/s]27672 examples [00:25, 1083.06 examples/s]27781 examples [00:26, 1065.97 examples/s]27892 examples [00:26, 1076.32 examples/s]28000 examples [00:26, 1073.61 examples/s]28108 examples [00:26, 1061.48 examples/s]28216 examples [00:26, 1064.70 examples/s]28323 examples [00:26, 1059.97 examples/s]28432 examples [00:26, 1066.87 examples/s]28539 examples [00:26, 1052.74 examples/s]28648 examples [00:26, 1063.44 examples/s]28755 examples [00:26, 1063.94 examples/s]28862 examples [00:27, 1064.58 examples/s]28970 examples [00:27, 1068.09 examples/s]29078 examples [00:27, 1070.41 examples/s]29189 examples [00:27, 1079.31 examples/s]29297 examples [00:27, 1066.73 examples/s]29405 examples [00:27, 1068.38 examples/s]29516 examples [00:27, 1079.13 examples/s]29627 examples [00:27, 1087.18 examples/s]29736 examples [00:27, 1082.16 examples/s]29845 examples [00:27, 1060.56 examples/s]29955 examples [00:28, 1070.10 examples/s]30063 examples [00:28, 1024.98 examples/s]30166 examples [00:28, 1003.59 examples/s]30273 examples [00:28, 1020.43 examples/s]30380 examples [00:28, 1031.43 examples/s]30489 examples [00:28, 1046.35 examples/s]30594 examples [00:28, 1040.98 examples/s]30703 examples [00:28, 1053.97 examples/s]30809 examples [00:28, 1039.60 examples/s]30917 examples [00:29, 1051.25 examples/s]31025 examples [00:29, 1059.66 examples/s]31133 examples [00:29, 1064.08 examples/s]31243 examples [00:29, 1074.15 examples/s]31353 examples [00:29, 1079.48 examples/s]31462 examples [00:29, 1074.64 examples/s]31570 examples [00:29, 1071.22 examples/s]31678 examples [00:29, 1070.41 examples/s]31787 examples [00:29, 1075.21 examples/s]31897 examples [00:29, 1078.81 examples/s]32005 examples [00:30, 1057.28 examples/s]32115 examples [00:30, 1069.73 examples/s]32228 examples [00:30, 1084.91 examples/s]32338 examples [00:30, 1087.91 examples/s]32447 examples [00:30, 1083.97 examples/s]32556 examples [00:30, 1079.10 examples/s]32667 examples [00:30, 1086.17 examples/s]32778 examples [00:30, 1092.35 examples/s]32889 examples [00:30, 1095.11 examples/s]32999 examples [00:30, 1093.56 examples/s]33109 examples [00:31, 1076.27 examples/s]33217 examples [00:31, 1074.25 examples/s]33325 examples [00:31, 1058.57 examples/s]33435 examples [00:31, 1070.00 examples/s]33544 examples [00:31, 1075.25 examples/s]33652 examples [00:31, 1063.31 examples/s]33759 examples [00:31, 1043.22 examples/s]33865 examples [00:31, 1047.09 examples/s]33973 examples [00:31, 1054.62 examples/s]34081 examples [00:31, 1060.76 examples/s]34190 examples [00:32, 1067.16 examples/s]34301 examples [00:32, 1078.16 examples/s]34412 examples [00:32, 1084.78 examples/s]34523 examples [00:32, 1089.20 examples/s]34632 examples [00:32, 1073.43 examples/s]34740 examples [00:32, 1075.05 examples/s]34849 examples [00:32, 1078.82 examples/s]34957 examples [00:32, 1077.06 examples/s]35065 examples [00:32, 1067.31 examples/s]35175 examples [00:32, 1075.54 examples/s]35284 examples [00:33, 1076.83 examples/s]35392 examples [00:33, 1051.69 examples/s]35500 examples [00:33, 1058.90 examples/s]35607 examples [00:33, 1046.02 examples/s]35716 examples [00:33, 1056.41 examples/s]35825 examples [00:33, 1066.24 examples/s]35934 examples [00:33, 1072.82 examples/s]36044 examples [00:33, 1080.58 examples/s]36153 examples [00:33, 1076.81 examples/s]36261 examples [00:33, 1075.87 examples/s]36369 examples [00:34, 1073.14 examples/s]36478 examples [00:34, 1075.15 examples/s]36587 examples [00:34, 1078.42 examples/s]36695 examples [00:34, 1073.81 examples/s]36803 examples [00:34, 1052.86 examples/s]36909 examples [00:34, 1051.40 examples/s]37020 examples [00:34, 1068.29 examples/s]37131 examples [00:34, 1078.21 examples/s]37242 examples [00:34, 1087.53 examples/s]37353 examples [00:34, 1093.02 examples/s]37463 examples [00:35, 1087.45 examples/s]37573 examples [00:35, 1088.49 examples/s]37683 examples [00:35, 1091.88 examples/s]37793 examples [00:35, 1086.93 examples/s]37905 examples [00:35, 1094.53 examples/s]38015 examples [00:35, 1091.50 examples/s]38125 examples [00:35, 1071.68 examples/s]38233 examples [00:35, 1042.62 examples/s]38338 examples [00:35, 1041.82 examples/s]38446 examples [00:36, 1052.84 examples/s]38553 examples [00:36, 1057.04 examples/s]38662 examples [00:36, 1060.72 examples/s]38771 examples [00:36, 1069.32 examples/s]38880 examples [00:36, 1074.67 examples/s]38989 examples [00:36, 1077.14 examples/s]39097 examples [00:36, 1071.78 examples/s]39205 examples [00:36, 1067.38 examples/s]39314 examples [00:36, 1073.35 examples/s]39423 examples [00:36, 1076.15 examples/s]39531 examples [00:37, 1066.49 examples/s]39638 examples [00:37, 1061.56 examples/s]39748 examples [00:37, 1070.22 examples/s]39856 examples [00:37, 1066.19 examples/s]39963 examples [00:37, 1063.96 examples/s]40070 examples [00:37, 994.14 examples/s] 40171 examples [00:37, 982.22 examples/s]40275 examples [00:37, 998.75 examples/s]40385 examples [00:37, 1024.96 examples/s]40492 examples [00:37, 1036.07 examples/s]40600 examples [00:38, 1048.76 examples/s]40708 examples [00:38, 1056.93 examples/s]40814 examples [00:38, 1056.38 examples/s]40924 examples [00:38, 1068.00 examples/s]41035 examples [00:38, 1077.27 examples/s]41144 examples [00:38, 1079.02 examples/s]41253 examples [00:38, 1081.93 examples/s]41362 examples [00:38, 1072.28 examples/s]41472 examples [00:38, 1078.23 examples/s]41582 examples [00:38, 1083.26 examples/s]41692 examples [00:39, 1087.20 examples/s]41801 examples [00:39, 1084.40 examples/s]41910 examples [00:39, 1084.07 examples/s]42019 examples [00:39, 1073.97 examples/s]42130 examples [00:39, 1082.34 examples/s]42239 examples [00:39, 1081.22 examples/s]42348 examples [00:39, 1064.64 examples/s]42457 examples [00:39, 1072.09 examples/s]42565 examples [00:39, 1073.31 examples/s]42673 examples [00:39, 1068.06 examples/s]42783 examples [00:40, 1074.55 examples/s]42891 examples [00:40, 1070.18 examples/s]43002 examples [00:40, 1079.20 examples/s]43110 examples [00:40, 1078.32 examples/s]43221 examples [00:40, 1086.02 examples/s]43331 examples [00:40, 1089.51 examples/s]43440 examples [00:40, 1084.41 examples/s]43550 examples [00:40, 1088.98 examples/s]43659 examples [00:40, 1059.95 examples/s]43770 examples [00:41, 1072.33 examples/s]43880 examples [00:41, 1079.32 examples/s]43989 examples [00:41, 1058.11 examples/s]44100 examples [00:41, 1072.14 examples/s]44210 examples [00:41, 1079.64 examples/s]44319 examples [00:41, 1079.91 examples/s]44431 examples [00:41, 1088.78 examples/s]44540 examples [00:41, 1072.83 examples/s]44648 examples [00:41, 1052.89 examples/s]44757 examples [00:41, 1063.02 examples/s]44866 examples [00:42, 1069.01 examples/s]44974 examples [00:42, 1069.67 examples/s]45082 examples [00:42, 1072.07 examples/s]45191 examples [00:42, 1076.17 examples/s]45303 examples [00:42, 1086.19 examples/s]45412 examples [00:42, 1079.99 examples/s]45521 examples [00:42, 1082.70 examples/s]45632 examples [00:42, 1088.58 examples/s]45744 examples [00:42, 1095.00 examples/s]45854 examples [00:42, 1093.91 examples/s]45964 examples [00:43, 1085.11 examples/s]46075 examples [00:43, 1091.50 examples/s]46185 examples [00:43, 1082.96 examples/s]46294 examples [00:43, 1081.65 examples/s]46405 examples [00:43, 1089.52 examples/s]46515 examples [00:43, 1092.46 examples/s]46625 examples [00:43, 1081.96 examples/s]46734 examples [00:43, 1060.04 examples/s]46844 examples [00:43, 1070.60 examples/s]46952 examples [00:43, 1047.00 examples/s]47060 examples [00:44, 1055.74 examples/s]47170 examples [00:44, 1066.79 examples/s]47277 examples [00:44, 1061.82 examples/s]47385 examples [00:44, 1064.42 examples/s]47494 examples [00:44, 1070.21 examples/s]47605 examples [00:44, 1080.45 examples/s]47714 examples [00:44, 1080.43 examples/s]47823 examples [00:44, 1077.86 examples/s]47933 examples [00:44, 1082.10 examples/s]48042 examples [00:44, 1080.47 examples/s]48151 examples [00:45, 1081.97 examples/s]48260 examples [00:45, 1080.63 examples/s]48369 examples [00:45, 1082.12 examples/s]48478 examples [00:45, 1081.34 examples/s]48588 examples [00:45, 1084.82 examples/s]48698 examples [00:45, 1088.23 examples/s]48808 examples [00:45, 1090.75 examples/s]48918 examples [00:45, 1087.82 examples/s]49029 examples [00:45, 1093.44 examples/s]49141 examples [00:45, 1099.20 examples/s]49251 examples [00:46, 1097.02 examples/s]49363 examples [00:46, 1102.27 examples/s]49474 examples [00:46, 1100.73 examples/s]49585 examples [00:46, 1033.04 examples/s]49690 examples [00:46, 1036.34 examples/s]49797 examples [00:46, 1045.25 examples/s]49902 examples [00:46, 1041.74 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6768/50000 [00:00<00:00, 67676.19 examples/s] 38%|███▊      | 19161/50000 [00:00<00:00, 78343.51 examples/s] 65%|██████▍   | 32422/50000 [00:00<00:00, 89306.93 examples/s] 92%|█████████▏| 45801/50000 [00:00<00:00, 99199.61 examples/s]                                                               0 examples [00:00, ? examples/s]88 examples [00:00, 877.81 examples/s]198 examples [00:00, 934.13 examples/s]310 examples [00:00, 981.39 examples/s]421 examples [00:00, 1014.85 examples/s]530 examples [00:00, 1036.24 examples/s]640 examples [00:00, 1053.15 examples/s]750 examples [00:00, 1064.50 examples/s]858 examples [00:00, 1067.78 examples/s]961 examples [00:00, 1021.19 examples/s]1067 examples [00:01, 1030.63 examples/s]1177 examples [00:01, 1048.79 examples/s]1288 examples [00:01, 1065.47 examples/s]1394 examples [00:01, 1034.69 examples/s]1502 examples [00:01, 1045.55 examples/s]1610 examples [00:01, 1055.51 examples/s]1716 examples [00:01, 1048.60 examples/s]1823 examples [00:01, 1052.18 examples/s]1933 examples [00:01, 1064.62 examples/s]2045 examples [00:01, 1080.08 examples/s]2155 examples [00:02, 1083.79 examples/s]2264 examples [00:02, 1083.16 examples/s]2373 examples [00:02, 1085.09 examples/s]2484 examples [00:02, 1090.55 examples/s]2594 examples [00:02, 1077.58 examples/s]2702 examples [00:02, 1062.85 examples/s]2811 examples [00:02, 1068.85 examples/s]2920 examples [00:02, 1074.69 examples/s]3029 examples [00:02, 1078.69 examples/s]3140 examples [00:02, 1086.61 examples/s]3249 examples [00:03, 1086.06 examples/s]3358 examples [00:03, 1085.22 examples/s]3467 examples [00:03, 1077.77 examples/s]3577 examples [00:03, 1083.98 examples/s]3688 examples [00:03, 1090.31 examples/s]3799 examples [00:03, 1094.13 examples/s]3909 examples [00:03, 1074.76 examples/s]4020 examples [00:03, 1083.05 examples/s]4131 examples [00:03, 1088.51 examples/s]4242 examples [00:03, 1092.73 examples/s]4352 examples [00:04, 1091.12 examples/s]4462 examples [00:04, 1074.41 examples/s]4574 examples [00:04, 1086.19 examples/s]4685 examples [00:04, 1091.21 examples/s]4796 examples [00:04, 1095.27 examples/s]4907 examples [00:04, 1097.34 examples/s]5017 examples [00:04, 1091.01 examples/s]5127 examples [00:04, 1088.70 examples/s]5236 examples [00:04, 1080.28 examples/s]5345 examples [00:04, 1051.93 examples/s]5451 examples [00:05, 824.45 examples/s] 5560 examples [00:05, 889.03 examples/s]5671 examples [00:05, 944.35 examples/s]5780 examples [00:05, 981.75 examples/s]5885 examples [00:05, 999.27 examples/s]5989 examples [00:05, 1002.80 examples/s]6095 examples [00:05, 1018.81 examples/s]6205 examples [00:05, 1040.12 examples/s]6315 examples [00:05, 1057.35 examples/s]6422 examples [00:06, 1058.35 examples/s]6529 examples [00:06, 1056.87 examples/s]6640 examples [00:06, 1070.83 examples/s]6750 examples [00:06, 1076.96 examples/s]6858 examples [00:06, 1071.70 examples/s]6966 examples [00:06, 1066.11 examples/s]7074 examples [00:06, 1068.46 examples/s]7184 examples [00:06, 1076.70 examples/s]7294 examples [00:06, 1082.51 examples/s]7405 examples [00:06, 1089.26 examples/s]7515 examples [00:07, 1092.22 examples/s]7625 examples [00:07, 1090.22 examples/s]7735 examples [00:07, 1081.39 examples/s]7844 examples [00:07, 1082.34 examples/s]7953 examples [00:07, 1054.29 examples/s]8062 examples [00:07, 1063.71 examples/s]8170 examples [00:07, 1068.37 examples/s]8280 examples [00:07, 1076.56 examples/s]8389 examples [00:07, 1080.37 examples/s]8499 examples [00:08, 1085.62 examples/s]8608 examples [00:08, 1075.12 examples/s]8716 examples [00:08, 1075.90 examples/s]8824 examples [00:08, 1076.57 examples/s]8932 examples [00:08, 1075.11 examples/s]9043 examples [00:08, 1083.25 examples/s]9152 examples [00:08, 1067.83 examples/s]9260 examples [00:08, 1070.77 examples/s]9371 examples [00:08, 1079.64 examples/s]9481 examples [00:08, 1084.95 examples/s]9592 examples [00:09, 1088.72 examples/s]9701 examples [00:09, 1066.21 examples/s]9809 examples [00:09, 1068.75 examples/s]9920 examples [00:09, 1077.98 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS0O2D5/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS0O2D5/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4d053a40d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4d053a40d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4d053a40d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4d50d81828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4c983ca208>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4d053a40d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4d053a40d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4d053a40d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4ced2660f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4ced266048>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.25 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 20.56 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.49 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.49 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.31 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.31 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.91 file/s]2020-08-24 00:09:03.089708: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-24 00:09:03.093847: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095199999 Hz
2020-08-24 00:09:03.094003: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558eebb06580 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-24 00:09:03.094018: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157508.71it/s] 19%|█▉        | 1884160/9912422 [00:00<00:35, 224131.28it/s] 44%|████▍     | 4407296/9912422 [00:00<00:17, 318947.23it/s] 67%|██████▋   | 6660096/9912422 [00:00<00:07, 452859.29it/s] 94%|█████████▍| 9347072/9912422 [00:00<00:00, 642144.04it/s]9920512it [00:00, 17081786.44it/s]                           
0it [00:00, ?it/s]32768it [00:00, 522198.36it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 155293.69it/s]1654784it [00:00, 10703010.97it/s]                         
0it [00:00, ?it/s]8192it [00:00, 166463.54it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
