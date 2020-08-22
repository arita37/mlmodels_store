
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f972c2f31e0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f972c2f31e0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f9797019400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f9797019400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f97b6235ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f97b6235ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f97443550d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f97443550d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f97443550d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 20%|██        | 2023424/9912422 [00:00<00:00, 20124479.81it/s]9920512it [00:00, 32684874.70it/s]                             
0it [00:00, ?it/s]32768it [00:00, 598101.56it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 147308.95it/s]1654784it [00:00, 10629527.87it/s]                         
0it [00:00, ?it/s]8192it [00:00, 188803.26it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9742518b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f97424fa2b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f974434dd08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f974434dd08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f974434dd08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:35,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:35,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:34,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:33,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:33,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:32,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:04,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:04,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:03,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:03,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:03,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:02,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:02,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:43,  3.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:42,  3.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:42,  3.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:42,  3.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:29,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:29,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:29,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:28,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:28,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:28,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:28,  4.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:19,  6.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  8.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  8.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:13,  8.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:13,  8.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:13,  8.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:13,  8.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:12,  8.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  8.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  8.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 12.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:08, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 27.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 34.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 34.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 49.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 49.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 49.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 49.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 49.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 49.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 55.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 60.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 60.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.63s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.63s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.63s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.63s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.30 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.90 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ url]
0 examples [00:00, ? examples/s]2020-08-22 00:07:16.993640: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-22 00:07:17.008043: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-22 00:07:17.008222: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562eb4eaffe0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-22 00:07:17.008239: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.80 examples/s]100 examples [00:00,  3.99 examples/s]205 examples [00:00,  5.69 examples/s]309 examples [00:00,  8.11 examples/s]416 examples [00:00, 11.55 examples/s]528 examples [00:00, 16.43 examples/s]627 examples [00:00, 23.30 examples/s]729 examples [00:01, 32.97 examples/s]828 examples [00:01, 46.43 examples/s]934 examples [00:01, 65.11 examples/s]1035 examples [00:01, 90.50 examples/s]1135 examples [00:01, 124.29 examples/s]1235 examples [00:01, 168.51 examples/s]1339 examples [00:01, 225.07 examples/s]1445 examples [00:01, 294.57 examples/s]1553 examples [00:01, 376.73 examples/s]1662 examples [00:01, 468.56 examples/s]1768 examples [00:02, 554.10 examples/s]1871 examples [00:02, 633.96 examples/s]1976 examples [00:02, 719.30 examples/s]2078 examples [00:02, 783.91 examples/s]2182 examples [00:02, 845.97 examples/s]2284 examples [00:02, 872.68 examples/s]2384 examples [00:02, 896.67 examples/s]2484 examples [00:02, 924.92 examples/s]2585 examples [00:02, 948.49 examples/s]2689 examples [00:03, 972.02 examples/s]2790 examples [00:03, 978.56 examples/s]2891 examples [00:03, 966.71 examples/s]2990 examples [00:03, 971.76 examples/s]3091 examples [00:03, 980.78 examples/s]3191 examples [00:03, 984.91 examples/s]3297 examples [00:03, 1006.17 examples/s]3401 examples [00:03, 1013.13 examples/s]3513 examples [00:03, 1042.97 examples/s]3618 examples [00:03, 1037.74 examples/s]3724 examples [00:04, 1042.29 examples/s]3829 examples [00:04, 1031.64 examples/s]3933 examples [00:04, 1013.27 examples/s]4039 examples [00:04, 1024.65 examples/s]4142 examples [00:04, 1018.89 examples/s]4248 examples [00:04, 1030.00 examples/s]4354 examples [00:04, 1036.41 examples/s]4458 examples [00:04, 1012.39 examples/s]4563 examples [00:04, 1023.31 examples/s]4670 examples [00:04, 1036.39 examples/s]4778 examples [00:05, 1049.06 examples/s]4884 examples [00:05, 1050.31 examples/s]4990 examples [00:05, 1022.15 examples/s]5094 examples [00:05, 1025.45 examples/s]5197 examples [00:05, 988.35 examples/s] 5297 examples [00:05, 974.22 examples/s]5395 examples [00:05, 972.58 examples/s]5493 examples [00:05, 916.34 examples/s]5596 examples [00:05, 945.57 examples/s]5694 examples [00:05, 954.17 examples/s]5801 examples [00:06, 979.35 examples/s]5900 examples [00:06, 975.65 examples/s]5998 examples [00:06, 934.85 examples/s]6093 examples [00:06, 914.95 examples/s]6192 examples [00:06, 934.48 examples/s]6297 examples [00:06, 966.02 examples/s]6395 examples [00:06, 960.61 examples/s]6492 examples [00:06, 944.38 examples/s]6589 examples [00:06, 949.32 examples/s]6690 examples [00:07, 965.57 examples/s]6793 examples [00:07, 981.98 examples/s]6892 examples [00:07, 983.31 examples/s]6991 examples [00:07, 955.69 examples/s]7091 examples [00:07, 966.61 examples/s]7201 examples [00:07, 1000.82 examples/s]7303 examples [00:07, 1004.39 examples/s]7411 examples [00:07, 1024.26 examples/s]7514 examples [00:07, 1008.34 examples/s]7619 examples [00:07, 1016.84 examples/s]7721 examples [00:08, 1011.52 examples/s]7823 examples [00:08, 962.13 examples/s] 7920 examples [00:08, 954.23 examples/s]8016 examples [00:08, 945.00 examples/s]8115 examples [00:08, 956.42 examples/s]8211 examples [00:08, 947.44 examples/s]8306 examples [00:08, 942.85 examples/s]8403 examples [00:08, 949.25 examples/s]8500 examples [00:08, 954.43 examples/s]8599 examples [00:08, 964.27 examples/s]8705 examples [00:09, 989.11 examples/s]8806 examples [00:09, 993.91 examples/s]8908 examples [00:09, 999.75 examples/s]9009 examples [00:09, 962.45 examples/s]9106 examples [00:09, 947.43 examples/s]9209 examples [00:09, 970.07 examples/s]9307 examples [00:09, 969.62 examples/s]9405 examples [00:09, 954.02 examples/s]9501 examples [00:09, 879.70 examples/s]9591 examples [00:10, 858.36 examples/s]9689 examples [00:10, 889.55 examples/s]9791 examples [00:10, 923.37 examples/s]9895 examples [00:10, 954.83 examples/s]9997 examples [00:10, 971.06 examples/s]10095 examples [00:10, 873.51 examples/s]10199 examples [00:10, 915.82 examples/s]10304 examples [00:10, 949.32 examples/s]10403 examples [00:10, 961.01 examples/s]10506 examples [00:11, 977.53 examples/s]10605 examples [00:11, 966.61 examples/s]10703 examples [00:11, 944.45 examples/s]10799 examples [00:11, 939.61 examples/s]10898 examples [00:11, 952.95 examples/s]10994 examples [00:11, 921.54 examples/s]11094 examples [00:11, 942.83 examples/s]11194 examples [00:11, 958.45 examples/s]11291 examples [00:11, 939.22 examples/s]11388 examples [00:11, 947.95 examples/s]11492 examples [00:12, 973.05 examples/s]11590 examples [00:12, 957.00 examples/s]11692 examples [00:12, 974.45 examples/s]11790 examples [00:12, 973.92 examples/s]11888 examples [00:12, 956.90 examples/s]11992 examples [00:12, 977.68 examples/s]12099 examples [00:12, 1001.56 examples/s]12205 examples [00:12, 1015.94 examples/s]12307 examples [00:12, 956.78 examples/s] 12413 examples [00:12, 985.47 examples/s]12513 examples [00:13, 981.42 examples/s]12618 examples [00:13, 999.42 examples/s]12722 examples [00:13, 1010.04 examples/s]12824 examples [00:13, 986.10 examples/s] 12926 examples [00:13, 994.72 examples/s]13032 examples [00:13, 1011.39 examples/s]13134 examples [00:13, 1000.96 examples/s]13235 examples [00:13, 983.40 examples/s] 13334 examples [00:13, 979.02 examples/s]13434 examples [00:14, 985.01 examples/s]13535 examples [00:14, 989.67 examples/s]13637 examples [00:14, 998.37 examples/s]13740 examples [00:14, 1007.05 examples/s]13841 examples [00:14, 982.65 examples/s] 13940 examples [00:14, 975.11 examples/s]14038 examples [00:14, 971.64 examples/s]14141 examples [00:14, 986.95 examples/s]14250 examples [00:14, 1013.10 examples/s]14352 examples [00:14, 1001.37 examples/s]14457 examples [00:15, 1013.99 examples/s]14559 examples [00:15, 986.90 examples/s] 14658 examples [00:15, 986.26 examples/s]14757 examples [00:15, 964.83 examples/s]14854 examples [00:15, 958.34 examples/s]14955 examples [00:15, 973.27 examples/s]15057 examples [00:15, 986.50 examples/s]15156 examples [00:15, 986.60 examples/s]15260 examples [00:15, 1000.67 examples/s]15362 examples [00:15, 1004.20 examples/s]15467 examples [00:16, 1015.56 examples/s]15569 examples [00:16, 944.61 examples/s] 15671 examples [00:16, 964.48 examples/s]15773 examples [00:16, 976.57 examples/s]15872 examples [00:16, 973.47 examples/s]15970 examples [00:16, 964.73 examples/s]16075 examples [00:16, 987.87 examples/s]16181 examples [00:16, 1006.54 examples/s]16285 examples [00:16, 1014.68 examples/s]16387 examples [00:16, 991.46 examples/s] 16487 examples [00:17, 983.66 examples/s]16596 examples [00:17, 1012.16 examples/s]16705 examples [00:17, 1031.97 examples/s]16809 examples [00:17, 1022.42 examples/s]16912 examples [00:17, 1001.65 examples/s]17014 examples [00:17, 1007.05 examples/s]17116 examples [00:17, 1009.62 examples/s]17218 examples [00:17, 995.59 examples/s] 17320 examples [00:17, 1001.86 examples/s]17421 examples [00:18, 987.30 examples/s] 17523 examples [00:18, 996.66 examples/s]17627 examples [00:18, 1007.77 examples/s]17730 examples [00:18, 1011.96 examples/s]17832 examples [00:18, 1001.57 examples/s]17933 examples [00:18, 983.46 examples/s] 18033 examples [00:18, 985.89 examples/s]18137 examples [00:18, 999.97 examples/s]18241 examples [00:18, 1009.68 examples/s]18343 examples [00:18, 972.26 examples/s] 18441 examples [00:19, 960.38 examples/s]18542 examples [00:19, 973.66 examples/s]18646 examples [00:19, 991.68 examples/s]18750 examples [00:19, 1002.02 examples/s]18851 examples [00:19, 1003.72 examples/s]18952 examples [00:19, 1001.29 examples/s]19056 examples [00:19, 1010.54 examples/s]19160 examples [00:19, 1017.93 examples/s]19264 examples [00:19, 1023.48 examples/s]19367 examples [00:19, 1007.62 examples/s]19471 examples [00:20, 1014.67 examples/s]19574 examples [00:20, 1018.25 examples/s]19680 examples [00:20, 1027.78 examples/s]19787 examples [00:20, 1038.80 examples/s]19891 examples [00:20, 1036.56 examples/s]19995 examples [00:20, 1027.07 examples/s]20098 examples [00:20, 971.16 examples/s] 20205 examples [00:20, 997.82 examples/s]20310 examples [00:20, 1010.23 examples/s]20412 examples [00:20, 1011.96 examples/s]20515 examples [00:21, 1016.18 examples/s]20620 examples [00:21, 1025.47 examples/s]20725 examples [00:21, 1031.33 examples/s]20829 examples [00:21, 1009.31 examples/s]20931 examples [00:21, 989.86 examples/s] 21034 examples [00:21, 999.27 examples/s]21141 examples [00:21, 1019.45 examples/s]21244 examples [00:21, 1019.03 examples/s]21347 examples [00:21, 1019.13 examples/s]21450 examples [00:22, 1017.29 examples/s]21553 examples [00:22, 1020.57 examples/s]21656 examples [00:22, 1007.35 examples/s]21769 examples [00:22, 1039.05 examples/s]21880 examples [00:22, 1057.85 examples/s]21987 examples [00:22, 1052.43 examples/s]22093 examples [00:22, 1041.39 examples/s]22203 examples [00:22, 1055.74 examples/s]22315 examples [00:22, 1072.82 examples/s]22429 examples [00:22, 1089.55 examples/s]22539 examples [00:23, 1049.26 examples/s]22645 examples [00:23, 1046.65 examples/s]22757 examples [00:23, 1065.81 examples/s]22864 examples [00:23, 1059.12 examples/s]22972 examples [00:23, 1063.22 examples/s]23079 examples [00:23, 1055.31 examples/s]23185 examples [00:23, 1053.93 examples/s]23300 examples [00:23, 1080.84 examples/s]23409 examples [00:23, 1080.89 examples/s]23521 examples [00:23, 1091.94 examples/s]23631 examples [00:24, 1050.35 examples/s]23737 examples [00:24, 1031.37 examples/s]23850 examples [00:24, 1056.99 examples/s]23957 examples [00:24, 1058.06 examples/s]24064 examples [00:24, 1040.01 examples/s]24169 examples [00:24, 1023.15 examples/s]24276 examples [00:24, 1035.22 examples/s]24387 examples [00:24, 1056.24 examples/s]24493 examples [00:24, 1055.92 examples/s]24599 examples [00:25, 1044.65 examples/s]24704 examples [00:25, 1028.08 examples/s]24814 examples [00:25, 1046.93 examples/s]24924 examples [00:25, 1061.63 examples/s]25032 examples [00:25, 1065.94 examples/s]25139 examples [00:25, 1056.60 examples/s]25245 examples [00:25, 1031.63 examples/s]25351 examples [00:25, 1038.00 examples/s]25461 examples [00:25, 1055.00 examples/s]25571 examples [00:25, 1066.09 examples/s]25681 examples [00:26, 1073.58 examples/s]25789 examples [00:26, 1048.02 examples/s]25895 examples [00:26, 1039.80 examples/s]26005 examples [00:26, 1057.00 examples/s]26111 examples [00:26, 1044.56 examples/s]26216 examples [00:26, 992.58 examples/s] 26317 examples [00:26, 996.75 examples/s]26423 examples [00:26, 1013.87 examples/s]26530 examples [00:26, 1029.91 examples/s]26639 examples [00:26, 1044.68 examples/s]26744 examples [00:27, 1041.51 examples/s]26849 examples [00:27, 1011.83 examples/s]26957 examples [00:27, 1027.86 examples/s]27063 examples [00:27, 1034.66 examples/s]27167 examples [00:27, 1031.76 examples/s]27277 examples [00:27, 1050.12 examples/s]27383 examples [00:27, 1044.36 examples/s]27491 examples [00:27, 1053.16 examples/s]27601 examples [00:27, 1065.20 examples/s]27709 examples [00:27, 1068.94 examples/s]27816 examples [00:28, 1035.01 examples/s]27920 examples [00:28, 1000.38 examples/s]28026 examples [00:28, 1017.53 examples/s]28129 examples [00:28, 959.51 examples/s] 28236 examples [00:28, 988.81 examples/s]28340 examples [00:28, 1000.95 examples/s]28441 examples [00:28, 953.56 examples/s] 28540 examples [00:28, 962.22 examples/s]28642 examples [00:28, 976.76 examples/s]28748 examples [00:29, 999.28 examples/s]28849 examples [00:29, 995.31 examples/s]28949 examples [00:29, 967.84 examples/s]29047 examples [00:29, 961.61 examples/s]29155 examples [00:29, 992.48 examples/s]29261 examples [00:29, 1011.80 examples/s]29363 examples [00:29, 1009.43 examples/s]29465 examples [00:29, 1002.29 examples/s]29567 examples [00:29, 1007.02 examples/s]29671 examples [00:29, 1016.04 examples/s]29779 examples [00:30, 1032.37 examples/s]29891 examples [00:30, 1055.61 examples/s]29997 examples [00:30, 1048.34 examples/s]30103 examples [00:30, 1008.48 examples/s]30213 examples [00:30, 1031.49 examples/s]30322 examples [00:30, 1045.99 examples/s]30427 examples [00:30, 1038.09 examples/s]30537 examples [00:30, 1055.66 examples/s]30648 examples [00:30, 1070.22 examples/s]30756 examples [00:30, 1065.96 examples/s]30868 examples [00:31, 1080.10 examples/s]30978 examples [00:31, 1082.19 examples/s]31087 examples [00:31, 1050.15 examples/s]31193 examples [00:31, 1014.91 examples/s]31295 examples [00:31, 990.94 examples/s] 31395 examples [00:31, 973.33 examples/s]31493 examples [00:31, 961.54 examples/s]31599 examples [00:31, 987.89 examples/s]31703 examples [00:31, 1002.56 examples/s]31805 examples [00:32, 1005.56 examples/s]31906 examples [00:32, 967.19 examples/s] 32004 examples [00:32, 967.25 examples/s]32105 examples [00:32, 978.76 examples/s]32209 examples [00:32, 995.51 examples/s]32313 examples [00:32, 1007.40 examples/s]32414 examples [00:32, 1005.35 examples/s]32515 examples [00:32, 1002.18 examples/s]32621 examples [00:32, 1018.11 examples/s]32727 examples [00:32, 1028.42 examples/s]32833 examples [00:33, 1036.18 examples/s]32942 examples [00:33, 1050.89 examples/s]33048 examples [00:33, 1042.64 examples/s]33156 examples [00:33, 1053.35 examples/s]33263 examples [00:33, 1056.33 examples/s]33372 examples [00:33, 1064.03 examples/s]33479 examples [00:33, 1064.30 examples/s]33586 examples [00:33, 1043.12 examples/s]33695 examples [00:33, 1055.81 examples/s]33805 examples [00:33, 1068.42 examples/s]33912 examples [00:34, 1053.65 examples/s]34020 examples [00:34, 1059.99 examples/s]34127 examples [00:34, 1041.97 examples/s]34232 examples [00:34, 1036.35 examples/s]34336 examples [00:34, 1008.49 examples/s]34443 examples [00:34, 1025.19 examples/s]34551 examples [00:34, 1039.01 examples/s]34656 examples [00:34, 1024.60 examples/s]34764 examples [00:34, 1039.76 examples/s]34871 examples [00:35, 1046.43 examples/s]34976 examples [00:35, 1041.95 examples/s]35081 examples [00:35, 1033.19 examples/s]35185 examples [00:35, 1005.83 examples/s]35286 examples [00:35, 989.69 examples/s] 35389 examples [00:35, 1000.77 examples/s]35490 examples [00:35, 997.48 examples/s] 35594 examples [00:35, 1009.42 examples/s]35696 examples [00:35, 974.21 examples/s] 35805 examples [00:35, 1004.59 examples/s]35908 examples [00:36, 1010.64 examples/s]36010 examples [00:36, 1005.99 examples/s]36111 examples [00:36, 1000.96 examples/s]36212 examples [00:36, 992.22 examples/s] 36313 examples [00:36, 997.10 examples/s]36413 examples [00:36, 984.40 examples/s]36521 examples [00:36, 1009.46 examples/s]36626 examples [00:36, 1019.00 examples/s]36734 examples [00:36, 1035.80 examples/s]36844 examples [00:36, 1053.93 examples/s]36952 examples [00:37, 1060.30 examples/s]37059 examples [00:37, 1034.22 examples/s]37165 examples [00:37, 1041.80 examples/s]37273 examples [00:37, 1052.90 examples/s]37379 examples [00:37, 1020.55 examples/s]37482 examples [00:37, 1022.25 examples/s]37585 examples [00:37, 979.08 examples/s] 37684 examples [00:37, 972.61 examples/s]37782 examples [00:37, 955.55 examples/s]37886 examples [00:38, 977.72 examples/s]37985 examples [00:38, 964.92 examples/s]38082 examples [00:38, 965.09 examples/s]38179 examples [00:38, 958.51 examples/s]38281 examples [00:38, 974.00 examples/s]38383 examples [00:38, 985.97 examples/s]38488 examples [00:38, 1003.00 examples/s]38589 examples [00:38, 997.83 examples/s] 38689 examples [00:38, 964.45 examples/s]38795 examples [00:38, 990.60 examples/s]38895 examples [00:39, 989.73 examples/s]38997 examples [00:39, 996.02 examples/s]39103 examples [00:39, 1011.77 examples/s]39205 examples [00:39, 1008.33 examples/s]39311 examples [00:39, 1022.57 examples/s]39414 examples [00:39, 1018.54 examples/s]39519 examples [00:39, 1025.66 examples/s]39629 examples [00:39, 1045.79 examples/s]39734 examples [00:39, 1026.78 examples/s]39843 examples [00:39, 1043.81 examples/s]39956 examples [00:40, 1067.99 examples/s]40064 examples [00:40, 1005.90 examples/s]40176 examples [00:40, 1037.31 examples/s]40281 examples [00:40, 1028.11 examples/s]40385 examples [00:40, 1030.33 examples/s]40492 examples [00:40, 1040.10 examples/s]40601 examples [00:40, 1052.08 examples/s]40707 examples [00:40, 1043.16 examples/s]40812 examples [00:40, 992.81 examples/s] 40912 examples [00:40, 988.63 examples/s]41012 examples [00:41, 975.36 examples/s]41112 examples [00:41, 980.02 examples/s]41211 examples [00:41, 955.29 examples/s]41307 examples [00:41, 942.54 examples/s]41403 examples [00:41, 945.61 examples/s]41504 examples [00:41, 962.88 examples/s]41608 examples [00:41, 983.91 examples/s]41712 examples [00:41, 998.57 examples/s]41813 examples [00:41, 997.25 examples/s]41921 examples [00:42, 1019.95 examples/s]42025 examples [00:42, 1025.57 examples/s]42132 examples [00:42, 1037.72 examples/s]42238 examples [00:42, 1043.51 examples/s]42343 examples [00:42, 1017.14 examples/s]42445 examples [00:42, 1007.81 examples/s]42553 examples [00:42, 1027.10 examples/s]42662 examples [00:42, 1043.29 examples/s]42768 examples [00:42, 1044.26 examples/s]42875 examples [00:42, 1050.83 examples/s]42981 examples [00:43, 1047.33 examples/s]43086 examples [00:43, 1042.97 examples/s]43191 examples [00:43, 1044.38 examples/s]43296 examples [00:43, 1035.24 examples/s]43400 examples [00:43, 1019.43 examples/s]43506 examples [00:43, 1030.33 examples/s]43612 examples [00:43, 1037.86 examples/s]43716 examples [00:43, 1022.78 examples/s]43819 examples [00:43, 1023.93 examples/s]43922 examples [00:43, 1017.00 examples/s]44028 examples [00:44, 1027.24 examples/s]44132 examples [00:44, 1029.73 examples/s]44242 examples [00:44, 1047.89 examples/s]44347 examples [00:44, 1045.84 examples/s]44452 examples [00:44, 1034.32 examples/s]44556 examples [00:44, 1033.92 examples/s]44660 examples [00:44, 1030.54 examples/s]44764 examples [00:44, 991.70 examples/s] 44864 examples [00:44, 980.64 examples/s]44967 examples [00:44, 992.85 examples/s]45072 examples [00:45, 1009.04 examples/s]45178 examples [00:45, 1022.70 examples/s]45289 examples [00:45, 1047.25 examples/s]45395 examples [00:45, 1040.82 examples/s]45504 examples [00:45, 1054.27 examples/s]45613 examples [00:45, 1062.81 examples/s]45720 examples [00:45, 1053.48 examples/s]45829 examples [00:45, 1063.01 examples/s]45936 examples [00:45, 1045.34 examples/s]46041 examples [00:46, 1038.93 examples/s]46147 examples [00:46, 1044.02 examples/s]46252 examples [00:46, 1042.66 examples/s]46362 examples [00:46, 1057.23 examples/s]46468 examples [00:46, 1032.91 examples/s]46572 examples [00:46, 1009.60 examples/s]46674 examples [00:46, 977.91 examples/s] 46773 examples [00:46, 954.71 examples/s]46876 examples [00:46, 973.58 examples/s]46974 examples [00:46, 970.07 examples/s]47079 examples [00:47, 992.44 examples/s]47179 examples [00:47, 992.04 examples/s]47283 examples [00:47, 1003.97 examples/s]47386 examples [00:47, 1011.64 examples/s]47488 examples [00:47, 1005.14 examples/s]47589 examples [00:47, 998.68 examples/s] 47696 examples [00:47, 1018.87 examples/s]47799 examples [00:47, 1015.33 examples/s]47906 examples [00:47, 1028.41 examples/s]48009 examples [00:47, 1018.05 examples/s]48111 examples [00:48, 1011.56 examples/s]48214 examples [00:48, 1016.15 examples/s]48318 examples [00:48, 1021.09 examples/s]48422 examples [00:48, 1024.71 examples/s]48525 examples [00:48, 1006.12 examples/s]48628 examples [00:48, 1011.64 examples/s]48731 examples [00:48, 1016.52 examples/s]48833 examples [00:48, 1013.52 examples/s]48935 examples [00:48, 983.11 examples/s] 49034 examples [00:48, 968.34 examples/s]49133 examples [00:49, 970.52 examples/s]49238 examples [00:49, 991.38 examples/s]49342 examples [00:49, 1003.73 examples/s]49444 examples [00:49, 1007.15 examples/s]49545 examples [00:49, 995.21 examples/s] 49650 examples [00:49, 1009.55 examples/s]49753 examples [00:49, 1015.00 examples/s]49857 examples [00:49, 1020.69 examples/s]49960 examples [00:49, 990.19 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▋        | 8227/50000 [00:00<00:00, 82266.74 examples/s] 44%|████▎     | 21811/50000 [00:00<00:00, 93305.80 examples/s] 70%|███████   | 35126/50000 [00:00<00:00, 102507.27 examples/s] 97%|█████████▋| 48534/50000 [00:00<00:00, 110297.60 examples/s]                                                                0 examples [00:00, ? examples/s]75 examples [00:00, 746.62 examples/s]185 examples [00:00, 825.41 examples/s]296 examples [00:00, 892.30 examples/s]405 examples [00:00, 942.75 examples/s]509 examples [00:00, 967.87 examples/s]613 examples [00:00, 987.19 examples/s]717 examples [00:00, 999.74 examples/s]822 examples [00:00, 1013.35 examples/s]927 examples [00:00, 1021.86 examples/s]1027 examples [00:01, 1005.27 examples/s]1126 examples [00:01, 974.05 examples/s] 1232 examples [00:01, 995.75 examples/s]1342 examples [00:01, 1023.36 examples/s]1453 examples [00:01, 1047.72 examples/s]1563 examples [00:01, 1061.98 examples/s]1670 examples [00:01, 1047.48 examples/s]1777 examples [00:01, 1052.13 examples/s]1883 examples [00:01, 1054.26 examples/s]1994 examples [00:01, 1069.15 examples/s]2102 examples [00:02, 1068.77 examples/s]2209 examples [00:02, 1043.39 examples/s]2318 examples [00:02, 1056.81 examples/s]2426 examples [00:02, 1061.97 examples/s]2534 examples [00:02, 1067.23 examples/s]2641 examples [00:02, 1058.84 examples/s]2747 examples [00:02, 1036.92 examples/s]2851 examples [00:02, 1035.30 examples/s]2963 examples [00:02, 1056.77 examples/s]3073 examples [00:02, 1066.57 examples/s]3182 examples [00:03, 1073.35 examples/s]3290 examples [00:03, 1060.40 examples/s]3397 examples [00:03, 1033.58 examples/s]3503 examples [00:03, 1041.12 examples/s]3608 examples [00:03, 1036.10 examples/s]3712 examples [00:03, 1033.02 examples/s]3816 examples [00:03, 974.61 examples/s] 3915 examples [00:03, 944.97 examples/s]4011 examples [00:03, 947.56 examples/s]4107 examples [00:04, 933.46 examples/s]4201 examples [00:04, 913.59 examples/s]4299 examples [00:04, 930.33 examples/s]4408 examples [00:04, 972.67 examples/s]4516 examples [00:04, 1000.98 examples/s]4621 examples [00:04, 1014.87 examples/s]4725 examples [00:04, 1019.41 examples/s]4828 examples [00:04, 1001.77 examples/s]4929 examples [00:04, 998.21 examples/s] 5031 examples [00:04, 1002.17 examples/s]5139 examples [00:05, 1022.05 examples/s]5243 examples [00:05, 1027.19 examples/s]5346 examples [00:05, 1020.45 examples/s]5449 examples [00:05, 845.52 examples/s] 5555 examples [00:05, 898.31 examples/s]5655 examples [00:05, 926.04 examples/s]5751 examples [00:05, 929.21 examples/s]5861 examples [00:05, 973.84 examples/s]5970 examples [00:05, 1003.91 examples/s]6080 examples [00:06, 1028.35 examples/s]6185 examples [00:06, 1010.46 examples/s]6288 examples [00:06, 981.13 examples/s] 6393 examples [00:06, 999.06 examples/s]6494 examples [00:06, 995.16 examples/s]6595 examples [00:06, 984.10 examples/s]6706 examples [00:06, 1017.78 examples/s]6809 examples [00:06, 1011.59 examples/s]6915 examples [00:06, 1022.75 examples/s]7026 examples [00:06, 1046.55 examples/s]7133 examples [00:07, 1052.08 examples/s]7239 examples [00:07, 1043.19 examples/s]7344 examples [00:07, 1025.29 examples/s]7451 examples [00:07, 1036.66 examples/s]7556 examples [00:07, 1038.61 examples/s]7663 examples [00:07, 1046.42 examples/s]7768 examples [00:07, 1035.96 examples/s]7872 examples [00:07, 1017.42 examples/s]7980 examples [00:07, 1033.90 examples/s]8084 examples [00:07, 1018.39 examples/s]8187 examples [00:08, 1015.76 examples/s]8290 examples [00:08, 1018.93 examples/s]8392 examples [00:08, 996.28 examples/s] 8496 examples [00:08, 1007.12 examples/s]8600 examples [00:08, 1016.60 examples/s]8703 examples [00:08, 1018.64 examples/s]8805 examples [00:08, 1009.20 examples/s]8907 examples [00:08, 1005.19 examples/s]9011 examples [00:08, 1014.45 examples/s]9114 examples [00:08, 1016.24 examples/s]9216 examples [00:09, 988.23 examples/s] 9316 examples [00:09, 960.53 examples/s]9414 examples [00:09, 963.56 examples/s]9519 examples [00:09, 985.93 examples/s]9622 examples [00:09, 996.65 examples/s]9722 examples [00:09, 994.04 examples/s]9824 examples [00:09, 1000.21 examples/s]9925 examples [00:09, 982.92 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteAQUC5I/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteAQUC5I/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f97443550d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f97443550d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f97443550d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f978fd33828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f96ce5bfda0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f97443550d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f97443550d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f97443550d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f96ce5ff0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f97424f4048>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 16.03 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.42 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.74 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.97 file/s]2020-08-22 00:08:24.250768: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-22 00:08:24.254466: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-22 00:08:24.254633: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55da09dceab0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-22 00:08:24.254649: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 159114.48it/s] 68%|██████▊   | 6725632/9912422 [00:00<00:14, 227071.03it/s]9920512it [00:00, 42590894.79it/s]                           
0it [00:00, ?it/s]32768it [00:00, 569485.05it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157054.42it/s] 92%|█████████▏| 1523712/1648877 [00:00<00:00, 223362.21it/s]1654784it [00:00, 8039739.78it/s]                            
0it [00:00, ?it/s]8192it [00:00, 215422.91it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
