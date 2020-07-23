
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f9fdd47b9d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f9fdd47b9d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fa0480e7510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fa0480e7510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fa067315ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fa067315ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9ff541a950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9ff541a950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9ff541a950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:15, 131374.45it/s] 60%|█████▉    | 5947392/9912422 [00:00<00:21, 187494.33it/s]9920512it [00:00, 35710025.55it/s]                           
0it [00:00, ?it/s]32768it [00:00, 562591.91it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160390.89it/s]1654784it [00:00, 11307922.52it/s]                         
0it [00:00, ?it/s]8192it [00:00, 160206.92it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9fdd3366a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9fdd3469b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f9ff541a598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f9ff541a598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f9ff541a598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:28,  1.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:28,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:28,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:27,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:27,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:26,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:26,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:25,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:00,  2.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:00,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:59,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:59,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:58,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:58,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:58,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:57,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:57,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:57,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:56,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:39,  3.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:39,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:39,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:39,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:39,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:38,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:38,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:38,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:37,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:37,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:37,  3.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:26,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:26,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:25,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:25,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:25,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:25,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:25,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:24,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:24,  5.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:17,  7.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:17,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:17,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:17,  7.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:17,  7.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:16,  7.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:16,  7.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:16,  7.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:16,  7.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:16,  7.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11,  9.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:10,  9.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 13.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 30.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 30.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 38.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 46.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 46.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 54.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 54.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 61.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 68.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.33s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.96 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.69 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.42s/ url]
0 examples [00:00, ? examples/s]2020-07-23 00:08:49.585432: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-23 00:08:49.624876: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-23 00:08:49.625114: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560f2904afd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-23 00:08:49.625135: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.93 examples/s]112 examples [00:00,  4.18 examples/s]216 examples [00:00,  5.96 examples/s]324 examples [00:00,  8.49 examples/s]430 examples [00:00, 12.09 examples/s]542 examples [00:00, 17.19 examples/s]652 examples [00:00, 24.39 examples/s]756 examples [00:01, 34.50 examples/s]863 examples [00:01, 48.61 examples/s]974 examples [00:01, 68.16 examples/s]1078 examples [00:01, 94.62 examples/s]1183 examples [00:01, 130.11 examples/s]1287 examples [00:01, 176.02 examples/s]1398 examples [00:01, 235.32 examples/s]1508 examples [00:01, 307.88 examples/s]1621 examples [00:01, 393.57 examples/s]1733 examples [00:01, 488.43 examples/s]1843 examples [00:02, 579.53 examples/s]1955 examples [00:02, 676.69 examples/s]2068 examples [00:02, 768.22 examples/s]2180 examples [00:02, 847.97 examples/s]2292 examples [00:02, 913.68 examples/s]2404 examples [00:02, 965.02 examples/s]2515 examples [00:02, 993.38 examples/s]2627 examples [00:02, 1026.15 examples/s]2738 examples [00:02, 1046.22 examples/s]2848 examples [00:02, 1058.37 examples/s]2958 examples [00:03, 1067.86 examples/s]3069 examples [00:03, 1078.77 examples/s]3179 examples [00:03, 1058.54 examples/s]3292 examples [00:03, 1076.04 examples/s]3403 examples [00:03, 1083.31 examples/s]3513 examples [00:03, 1078.69 examples/s]3624 examples [00:03, 1086.60 examples/s]3734 examples [00:03, 1083.57 examples/s]3843 examples [00:03, 1066.86 examples/s]3955 examples [00:03, 1080.06 examples/s]4064 examples [00:04, 1025.57 examples/s]4172 examples [00:04, 1040.05 examples/s]4282 examples [00:04, 1054.83 examples/s]4388 examples [00:04, 1054.41 examples/s]4496 examples [00:04, 1060.13 examples/s]4603 examples [00:04, 1042.23 examples/s]4714 examples [00:04, 1060.30 examples/s]4821 examples [00:04, 1033.68 examples/s]4925 examples [00:04, 1023.22 examples/s]5028 examples [00:05, 985.88 examples/s] 5129 examples [00:05, 992.80 examples/s]5239 examples [00:05, 1021.03 examples/s]5345 examples [00:05, 1029.11 examples/s]5452 examples [00:05, 1039.55 examples/s]5558 examples [00:05, 1043.00 examples/s]5663 examples [00:05, 1040.00 examples/s]5775 examples [00:05, 1061.34 examples/s]5887 examples [00:05, 1076.06 examples/s]5995 examples [00:05, 1061.14 examples/s]6102 examples [00:06, 1058.63 examples/s]6208 examples [00:06, 1056.06 examples/s]6318 examples [00:06, 1068.53 examples/s]6429 examples [00:06, 1080.41 examples/s]6538 examples [00:06, 1078.76 examples/s]6646 examples [00:06, 1063.50 examples/s]6755 examples [00:06, 1069.57 examples/s]6864 examples [00:06, 1072.57 examples/s]6973 examples [00:06, 1074.89 examples/s]7081 examples [00:06, 1022.22 examples/s]7184 examples [00:07, 1024.22 examples/s]7287 examples [00:07, 1018.59 examples/s]7394 examples [00:07, 1032.99 examples/s]7504 examples [00:07, 1051.22 examples/s]7610 examples [00:07, 1038.87 examples/s]7715 examples [00:07, 1018.17 examples/s]7827 examples [00:07, 1045.01 examples/s]7939 examples [00:07, 1065.85 examples/s]8050 examples [00:07, 1077.28 examples/s]8158 examples [00:08, 1078.02 examples/s]8266 examples [00:08, 1075.64 examples/s]8374 examples [00:08, 1071.62 examples/s]8482 examples [00:08, 1067.82 examples/s]8591 examples [00:08, 1073.01 examples/s]8701 examples [00:08, 1078.21 examples/s]8810 examples [00:08, 1079.52 examples/s]8921 examples [00:08, 1086.53 examples/s]9030 examples [00:08, 1073.85 examples/s]9140 examples [00:08, 1079.13 examples/s]9248 examples [00:09, 1066.94 examples/s]9356 examples [00:09, 1068.58 examples/s]9464 examples [00:09, 1070.31 examples/s]9574 examples [00:09, 1078.85 examples/s]9682 examples [00:09, 1052.99 examples/s]9788 examples [00:09, 1051.89 examples/s]9898 examples [00:09, 1063.78 examples/s]10005 examples [00:09, 1014.86 examples/s]10112 examples [00:09, 1028.94 examples/s]10223 examples [00:09, 1049.86 examples/s]10329 examples [00:10, 1029.31 examples/s]10436 examples [00:10, 1040.77 examples/s]10541 examples [00:10, 1029.54 examples/s]10645 examples [00:10, 1010.59 examples/s]10751 examples [00:10, 1024.10 examples/s]10854 examples [00:10, 1019.35 examples/s]10957 examples [00:10, 1011.89 examples/s]11068 examples [00:10, 1038.36 examples/s]11173 examples [00:10, 1026.21 examples/s]11276 examples [00:10, 1003.20 examples/s]11382 examples [00:11, 1017.71 examples/s]11491 examples [00:11, 1036.10 examples/s]11601 examples [00:11, 1052.71 examples/s]11707 examples [00:11, 1032.61 examples/s]11820 examples [00:11, 1059.71 examples/s]11927 examples [00:11, 1043.97 examples/s]12036 examples [00:11, 1056.39 examples/s]12142 examples [00:11, 1019.05 examples/s]12245 examples [00:11, 1019.18 examples/s]12350 examples [00:12, 1027.44 examples/s]12458 examples [00:12, 1040.24 examples/s]12563 examples [00:12, 1010.06 examples/s]12665 examples [00:12, 994.79 examples/s] 12773 examples [00:12, 1017.65 examples/s]12880 examples [00:12, 1031.99 examples/s]12984 examples [00:12, 1020.96 examples/s]13092 examples [00:12, 1036.74 examples/s]13201 examples [00:12, 1050.61 examples/s]13307 examples [00:12, 1025.43 examples/s]13410 examples [00:13, 951.25 examples/s] 13507 examples [00:13, 928.52 examples/s]13601 examples [00:13, 917.30 examples/s]13694 examples [00:13, 895.04 examples/s]13785 examples [00:13, 899.33 examples/s]13876 examples [00:13, 902.02 examples/s]13967 examples [00:13, 892.28 examples/s]14057 examples [00:13, 850.46 examples/s]14151 examples [00:13, 875.14 examples/s]14243 examples [00:14, 886.50 examples/s]14352 examples [00:14, 938.90 examples/s]14463 examples [00:14, 982.16 examples/s]14573 examples [00:14, 1013.97 examples/s]14680 examples [00:14, 1028.04 examples/s]14792 examples [00:14, 1053.83 examples/s]14899 examples [00:14, 1056.32 examples/s]15011 examples [00:14, 1072.72 examples/s]15122 examples [00:14, 1082.33 examples/s]15234 examples [00:14, 1091.24 examples/s]15344 examples [00:15, 1082.69 examples/s]15453 examples [00:15, 1059.09 examples/s]15563 examples [00:15, 1069.44 examples/s]15673 examples [00:15, 1076.86 examples/s]15783 examples [00:15, 1081.32 examples/s]15892 examples [00:15, 1081.55 examples/s]16001 examples [00:15, 1077.45 examples/s]16109 examples [00:15, 1067.39 examples/s]16216 examples [00:15, 1036.44 examples/s]16322 examples [00:15, 1042.44 examples/s]16433 examples [00:16, 1061.81 examples/s]16540 examples [00:16, 1006.43 examples/s]16649 examples [00:16, 1028.54 examples/s]16759 examples [00:16, 1048.16 examples/s]16866 examples [00:16, 1052.21 examples/s]16975 examples [00:16, 1060.72 examples/s]17086 examples [00:16, 1073.99 examples/s]17196 examples [00:16, 1079.44 examples/s]17306 examples [00:16, 1084.18 examples/s]17415 examples [00:16, 1069.55 examples/s]17523 examples [00:17, 1068.34 examples/s]17632 examples [00:17, 1074.37 examples/s]17740 examples [00:17, 1028.32 examples/s]17847 examples [00:17, 1038.08 examples/s]17959 examples [00:17, 1059.07 examples/s]18071 examples [00:17, 1075.48 examples/s]18179 examples [00:17, 1068.83 examples/s]18287 examples [00:17, 1059.43 examples/s]18399 examples [00:17, 1074.57 examples/s]18511 examples [00:18, 1085.28 examples/s]18624 examples [00:18, 1096.17 examples/s]18736 examples [00:18, 1100.65 examples/s]18847 examples [00:18, 1099.40 examples/s]18958 examples [00:18, 1101.58 examples/s]19070 examples [00:18, 1106.23 examples/s]19181 examples [00:18, 1090.67 examples/s]19293 examples [00:18, 1099.18 examples/s]19404 examples [00:18, 1101.40 examples/s]19516 examples [00:18, 1104.57 examples/s]19627 examples [00:19, 1091.78 examples/s]19737 examples [00:19, 1091.56 examples/s]19847 examples [00:19, 1091.61 examples/s]19957 examples [00:19, 1088.74 examples/s]20066 examples [00:19, 1037.69 examples/s]20175 examples [00:19, 1052.10 examples/s]20285 examples [00:19, 1063.49 examples/s]20397 examples [00:19, 1077.21 examples/s]20505 examples [00:19, 1073.82 examples/s]20617 examples [00:19, 1086.29 examples/s]20727 examples [00:20, 1089.08 examples/s]20837 examples [00:20, 1092.00 examples/s]20947 examples [00:20, 1091.05 examples/s]21057 examples [00:20, 1078.76 examples/s]21165 examples [00:20, 1018.36 examples/s]21276 examples [00:20, 1043.95 examples/s]21389 examples [00:20, 1066.33 examples/s]21501 examples [00:20, 1079.46 examples/s]21610 examples [00:20, 1080.56 examples/s]21719 examples [00:20, 1068.90 examples/s]21828 examples [00:21, 1072.18 examples/s]21936 examples [00:21, 1065.55 examples/s]22049 examples [00:21, 1081.49 examples/s]22162 examples [00:21, 1095.50 examples/s]22272 examples [00:21, 1092.52 examples/s]22382 examples [00:21, 1078.81 examples/s]22493 examples [00:21, 1086.56 examples/s]22602 examples [00:21, 1035.62 examples/s]22712 examples [00:21, 1053.57 examples/s]22823 examples [00:22, 1068.01 examples/s]22931 examples [00:22, 1053.72 examples/s]23043 examples [00:22, 1070.30 examples/s]23151 examples [00:22, 1072.69 examples/s]23260 examples [00:22, 1077.46 examples/s]23368 examples [00:22, 1076.06 examples/s]23476 examples [00:22, 1071.54 examples/s]23584 examples [00:22, 1071.04 examples/s]23698 examples [00:22, 1089.90 examples/s]23811 examples [00:22, 1098.88 examples/s]23924 examples [00:23, 1105.92 examples/s]24035 examples [00:23, 1087.37 examples/s]24146 examples [00:23, 1091.99 examples/s]24258 examples [00:23, 1100.04 examples/s]24369 examples [00:23, 1102.98 examples/s]24480 examples [00:23, 1088.96 examples/s]24589 examples [00:23, 1083.67 examples/s]24699 examples [00:23, 1087.80 examples/s]24810 examples [00:23, 1093.00 examples/s]24924 examples [00:23, 1105.84 examples/s]25035 examples [00:24, 1106.28 examples/s]25146 examples [00:24, 1100.15 examples/s]25259 examples [00:24, 1106.59 examples/s]25371 examples [00:24, 1108.98 examples/s]25483 examples [00:24, 1109.54 examples/s]25595 examples [00:24, 1110.61 examples/s]25707 examples [00:24, 1088.98 examples/s]25817 examples [00:24, 1076.58 examples/s]25929 examples [00:24, 1087.71 examples/s]26041 examples [00:24, 1096.18 examples/s]26151 examples [00:25, 1095.80 examples/s]26263 examples [00:25, 1101.34 examples/s]26375 examples [00:25, 1105.58 examples/s]26490 examples [00:25, 1116.39 examples/s]26604 examples [00:25, 1120.93 examples/s]26717 examples [00:25, 1109.19 examples/s]26828 examples [00:25, 1107.20 examples/s]26939 examples [00:25, 1104.80 examples/s]27050 examples [00:25, 1098.21 examples/s]27160 examples [00:25, 1039.90 examples/s]27270 examples [00:26, 1055.40 examples/s]27380 examples [00:26, 1066.01 examples/s]27493 examples [00:26, 1081.68 examples/s]27606 examples [00:26, 1094.98 examples/s]27717 examples [00:26, 1098.65 examples/s]27828 examples [00:26, 1101.78 examples/s]27941 examples [00:26, 1109.63 examples/s]28053 examples [00:26, 1081.23 examples/s]28168 examples [00:26, 1098.40 examples/s]28279 examples [00:27, 1071.23 examples/s]28387 examples [00:27, 1068.72 examples/s]28501 examples [00:27, 1086.93 examples/s]28613 examples [00:27, 1096.39 examples/s]28727 examples [00:27, 1107.32 examples/s]28840 examples [00:27, 1113.43 examples/s]28952 examples [00:27, 1098.75 examples/s]29063 examples [00:27, 1100.53 examples/s]29178 examples [00:27, 1112.42 examples/s]29290 examples [00:27, 1102.51 examples/s]29404 examples [00:28, 1112.48 examples/s]29516 examples [00:28, 1097.59 examples/s]29629 examples [00:28, 1106.47 examples/s]29740 examples [00:28, 1107.06 examples/s]29851 examples [00:28, 1104.68 examples/s]29962 examples [00:28, 1076.11 examples/s]30070 examples [00:28, 1031.65 examples/s]30182 examples [00:28, 1054.98 examples/s]30295 examples [00:28, 1074.08 examples/s]30403 examples [00:28, 1065.98 examples/s]30511 examples [00:29, 1068.59 examples/s]30624 examples [00:29, 1086.24 examples/s]30739 examples [00:29, 1101.98 examples/s]30852 examples [00:29, 1110.00 examples/s]30966 examples [00:29, 1118.21 examples/s]31078 examples [00:29, 1109.48 examples/s]31191 examples [00:29, 1114.51 examples/s]31306 examples [00:29, 1122.63 examples/s]31419 examples [00:29, 1121.07 examples/s]31532 examples [00:29, 1111.02 examples/s]31644 examples [00:30, 1113.18 examples/s]31757 examples [00:30, 1117.23 examples/s]31871 examples [00:30, 1121.38 examples/s]31984 examples [00:30, 1120.17 examples/s]32098 examples [00:30, 1124.73 examples/s]32212 examples [00:30, 1128.61 examples/s]32325 examples [00:30, 1091.51 examples/s]32436 examples [00:30, 1095.22 examples/s]32547 examples [00:30, 1097.93 examples/s]32657 examples [00:31, 1067.36 examples/s]32765 examples [00:31, 1044.57 examples/s]32874 examples [00:31, 1056.98 examples/s]32983 examples [00:31, 1066.44 examples/s]33095 examples [00:31, 1080.87 examples/s]33208 examples [00:31, 1093.86 examples/s]33323 examples [00:31, 1107.78 examples/s]33436 examples [00:31, 1111.72 examples/s]33550 examples [00:31, 1117.56 examples/s]33662 examples [00:31, 1114.75 examples/s]33777 examples [00:32, 1124.48 examples/s]33891 examples [00:32, 1127.09 examples/s]34004 examples [00:32, 1122.14 examples/s]34117 examples [00:32, 1120.06 examples/s]34230 examples [00:32, 1118.17 examples/s]34345 examples [00:32, 1125.90 examples/s]34459 examples [00:32, 1128.08 examples/s]34572 examples [00:32, 1124.40 examples/s]34686 examples [00:32, 1126.55 examples/s]34799 examples [00:32, 1125.32 examples/s]34914 examples [00:33, 1131.79 examples/s]35028 examples [00:33, 1132.84 examples/s]35142 examples [00:33, 1130.03 examples/s]35256 examples [00:33, 1119.57 examples/s]35368 examples [00:33, 1115.38 examples/s]35482 examples [00:33, 1121.94 examples/s]35596 examples [00:33, 1125.67 examples/s]35709 examples [00:33, 1112.60 examples/s]35822 examples [00:33, 1116.38 examples/s]35934 examples [00:33, 1117.14 examples/s]36048 examples [00:34, 1121.94 examples/s]36161 examples [00:34, 1104.51 examples/s]36272 examples [00:34, 1105.92 examples/s]36385 examples [00:34, 1112.91 examples/s]36497 examples [00:34, 1111.76 examples/s]36610 examples [00:34, 1114.74 examples/s]36722 examples [00:34, 1115.99 examples/s]36834 examples [00:34, 1111.38 examples/s]36946 examples [00:34, 1112.05 examples/s]37058 examples [00:34, 1107.81 examples/s]37169 examples [00:35, 1105.05 examples/s]37280 examples [00:35, 1060.09 examples/s]37389 examples [00:35, 1067.94 examples/s]37503 examples [00:35, 1086.75 examples/s]37614 examples [00:35, 1092.13 examples/s]37727 examples [00:35, 1102.25 examples/s]37838 examples [00:35, 1097.90 examples/s]37951 examples [00:35, 1106.15 examples/s]38062 examples [00:35, 1097.40 examples/s]38172 examples [00:35, 1058.95 examples/s]38283 examples [00:36, 1071.13 examples/s]38394 examples [00:36, 1081.63 examples/s]38503 examples [00:36, 1077.90 examples/s]38615 examples [00:36, 1088.71 examples/s]38726 examples [00:36, 1093.41 examples/s]38836 examples [00:36, 1090.37 examples/s]38949 examples [00:36, 1099.96 examples/s]39060 examples [00:36, 1080.55 examples/s]39171 examples [00:36, 1086.49 examples/s]39282 examples [00:37, 1092.04 examples/s]39394 examples [00:37, 1098.00 examples/s]39504 examples [00:37, 1079.89 examples/s]39613 examples [00:37, 1058.10 examples/s]39724 examples [00:37, 1071.93 examples/s]39835 examples [00:37, 1082.42 examples/s]39944 examples [00:37, 1083.21 examples/s]40053 examples [00:37, 1031.88 examples/s]40163 examples [00:37, 1049.19 examples/s]40276 examples [00:37, 1071.26 examples/s]40386 examples [00:38, 1079.20 examples/s]40499 examples [00:38, 1093.83 examples/s]40610 examples [00:38, 1096.34 examples/s]40720 examples [00:38, 1081.12 examples/s]40831 examples [00:38, 1088.81 examples/s]40942 examples [00:38, 1093.21 examples/s]41053 examples [00:38, 1095.59 examples/s]41164 examples [00:38, 1099.57 examples/s]41275 examples [00:38, 1096.43 examples/s]41387 examples [00:38, 1100.85 examples/s]41500 examples [00:39, 1106.84 examples/s]41612 examples [00:39, 1109.65 examples/s]41723 examples [00:39, 1093.27 examples/s]41833 examples [00:39, 1085.13 examples/s]41942 examples [00:39, 1062.37 examples/s]42055 examples [00:39, 1080.09 examples/s]42164 examples [00:39, 1081.76 examples/s]42273 examples [00:39, 1069.76 examples/s]42381 examples [00:39, 1042.00 examples/s]42486 examples [00:39, 1033.83 examples/s]42597 examples [00:40, 1055.07 examples/s]42706 examples [00:40, 1064.31 examples/s]42816 examples [00:40, 1072.52 examples/s]42927 examples [00:40, 1082.89 examples/s]43039 examples [00:40, 1092.63 examples/s]43154 examples [00:40, 1107.59 examples/s]43270 examples [00:40, 1120.81 examples/s]43383 examples [00:40, 1120.46 examples/s]43496 examples [00:40, 1121.10 examples/s]43610 examples [00:40, 1125.46 examples/s]43723 examples [00:41, 1123.49 examples/s]43836 examples [00:41, 1107.87 examples/s]43948 examples [00:41, 1108.62 examples/s]44060 examples [00:41, 1110.70 examples/s]44172 examples [00:41, 1074.91 examples/s]44281 examples [00:41, 1079.11 examples/s]44390 examples [00:41, 1058.92 examples/s]44497 examples [00:41, 1056.11 examples/s]44607 examples [00:41, 1067.66 examples/s]44720 examples [00:42, 1083.25 examples/s]44832 examples [00:42, 1092.43 examples/s]44942 examples [00:42, 1084.10 examples/s]45054 examples [00:42, 1093.59 examples/s]45164 examples [00:42, 1077.35 examples/s]45277 examples [00:42, 1090.65 examples/s]45388 examples [00:42, 1095.50 examples/s]45498 examples [00:42, 1095.45 examples/s]45608 examples [00:42, 1076.06 examples/s]45717 examples [00:42, 1078.52 examples/s]45828 examples [00:43, 1086.67 examples/s]45937 examples [00:43, 1082.27 examples/s]46051 examples [00:43, 1097.07 examples/s]46163 examples [00:43, 1102.70 examples/s]46274 examples [00:43, 1097.19 examples/s]46388 examples [00:43, 1107.02 examples/s]46499 examples [00:43, 1081.11 examples/s]46611 examples [00:43, 1090.46 examples/s]46722 examples [00:43, 1096.01 examples/s]46832 examples [00:43, 1091.17 examples/s]46945 examples [00:44, 1100.16 examples/s]47057 examples [00:44, 1104.82 examples/s]47172 examples [00:44, 1114.97 examples/s]47284 examples [00:44, 1115.07 examples/s]47396 examples [00:44, 1112.86 examples/s]47508 examples [00:44, 1110.43 examples/s]47620 examples [00:44, 1107.93 examples/s]47731 examples [00:44, 1105.06 examples/s]47842 examples [00:44, 1103.90 examples/s]47953 examples [00:44, 1096.15 examples/s]48064 examples [00:45, 1098.29 examples/s]48174 examples [00:45, 1076.38 examples/s]48288 examples [00:45, 1092.86 examples/s]48402 examples [00:45, 1103.90 examples/s]48513 examples [00:45, 1104.55 examples/s]48627 examples [00:45, 1112.69 examples/s]48739 examples [00:45, 1082.24 examples/s]48848 examples [00:45, 1044.80 examples/s]48959 examples [00:45, 1061.81 examples/s]49068 examples [00:45, 1068.50 examples/s]49176 examples [00:46, 1052.04 examples/s]49282 examples [00:46, 1050.58 examples/s]49388 examples [00:46, 1005.31 examples/s]49490 examples [00:46, 985.62 examples/s] 49593 examples [00:46, 996.88 examples/s]49702 examples [00:46, 1022.63 examples/s]49809 examples [00:46, 1035.04 examples/s]49919 examples [00:46, 1051.66 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6381/50000 [00:00<00:00, 63806.87 examples/s] 38%|███▊      | 19233/50000 [00:00<00:00, 75159.97 examples/s] 65%|██████▍   | 32268/50000 [00:00<00:00, 86094.51 examples/s] 91%|█████████▏| 45674/50000 [00:00<00:00, 96444.99 examples/s]                                                               0 examples [00:00, ? examples/s]87 examples [00:00, 868.43 examples/s]189 examples [00:00, 908.87 examples/s]297 examples [00:00, 953.42 examples/s]409 examples [00:00, 996.43 examples/s]519 examples [00:00, 1024.71 examples/s]630 examples [00:00, 1047.27 examples/s]741 examples [00:00, 1063.87 examples/s]853 examples [00:00, 1079.56 examples/s]965 examples [00:00, 1089.69 examples/s]1078 examples [00:01, 1099.76 examples/s]1186 examples [00:01, 1038.16 examples/s]1296 examples [00:01, 1053.63 examples/s]1408 examples [00:01, 1070.37 examples/s]1521 examples [00:01, 1085.16 examples/s]1630 examples [00:01, 1073.87 examples/s]1742 examples [00:01, 1086.36 examples/s]1854 examples [00:01, 1095.70 examples/s]1964 examples [00:01, 1082.80 examples/s]2074 examples [00:01, 1087.30 examples/s]2185 examples [00:02, 1092.18 examples/s]2296 examples [00:02, 1096.44 examples/s]2407 examples [00:02, 1099.28 examples/s]2520 examples [00:02, 1106.34 examples/s]2631 examples [00:02, 1091.12 examples/s]2745 examples [00:02, 1102.79 examples/s]2858 examples [00:02, 1107.87 examples/s]2969 examples [00:02, 1106.59 examples/s]3080 examples [00:02, 1077.74 examples/s]3188 examples [00:02, 1023.66 examples/s]3299 examples [00:03, 1047.72 examples/s]3410 examples [00:03, 1064.52 examples/s]3518 examples [00:03, 1068.13 examples/s]3631 examples [00:03, 1085.29 examples/s]3743 examples [00:03, 1095.40 examples/s]3856 examples [00:03, 1103.38 examples/s]3969 examples [00:03, 1109.60 examples/s]4083 examples [00:03, 1116.51 examples/s]4195 examples [00:03, 1100.47 examples/s]4306 examples [00:03, 1101.48 examples/s]4420 examples [00:04, 1109.67 examples/s]4532 examples [00:04, 1100.25 examples/s]4644 examples [00:04, 1103.97 examples/s]4755 examples [00:04, 1079.14 examples/s]4864 examples [00:04, 1060.90 examples/s]4971 examples [00:04, 1032.24 examples/s]5079 examples [00:04, 1046.00 examples/s]5189 examples [00:04, 1061.07 examples/s]5296 examples [00:04, 1054.08 examples/s]5410 examples [00:05, 1076.02 examples/s]5525 examples [00:05, 1094.37 examples/s]5638 examples [00:05, 1104.65 examples/s]5751 examples [00:05, 1110.33 examples/s]5863 examples [00:05, 1090.66 examples/s]5974 examples [00:05, 1094.98 examples/s]6086 examples [00:05, 1099.61 examples/s]6197 examples [00:05, 1098.54 examples/s]6307 examples [00:05, 1097.33 examples/s]6417 examples [00:05, 1095.88 examples/s]6527 examples [00:06, 1091.46 examples/s]6637 examples [00:06, 1093.59 examples/s]6750 examples [00:06, 1102.00 examples/s]6861 examples [00:06, 1099.11 examples/s]6974 examples [00:06, 1107.02 examples/s]7086 examples [00:06, 1109.98 examples/s]7199 examples [00:06, 1113.76 examples/s]7311 examples [00:06, 1111.31 examples/s]7425 examples [00:06, 1118.22 examples/s]7537 examples [00:06, 1112.94 examples/s]7650 examples [00:07, 1115.60 examples/s]7762 examples [00:07, 1116.15 examples/s]7874 examples [00:07, 1114.79 examples/s]7986 examples [00:07, 1088.69 examples/s]8096 examples [00:07, 1081.14 examples/s]8205 examples [00:07, 1065.15 examples/s]8316 examples [00:07, 1077.70 examples/s]8424 examples [00:07, 1073.73 examples/s]8532 examples [00:07, 1065.98 examples/s]8639 examples [00:07, 1060.59 examples/s]8752 examples [00:08, 1079.48 examples/s]8865 examples [00:08, 1092.95 examples/s]8977 examples [00:08, 1098.95 examples/s]9087 examples [00:08, 1082.07 examples/s]9199 examples [00:08, 1091.49 examples/s]9311 examples [00:08, 1098.89 examples/s]9424 examples [00:08, 1106.01 examples/s]9535 examples [00:08, 1059.45 examples/s]9645 examples [00:08, 1071.11 examples/s]9753 examples [00:08, 1065.43 examples/s]9860 examples [00:09, 1036.36 examples/s]9972 examples [00:09, 1059.27 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteEX7L2Z/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteEX7L2Z/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9ff541a950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9ff541a950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9ff541a950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9fdf1d5da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9ff34f5c18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9ff541a950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9ff541a950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9ff541a950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9fdd346240>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9fdd346358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f9f6ddc8ea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f9f6ddc8ea0> 

  function with postional parmater data_info <function split_train_valid at 0x7f9f6ddc8ea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f9f6e097048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f9f6e097048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f9f6e097048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=3d0e7dacc708df09b9577689fe8732110f63229ed6d1dc1485b4a687bb36973b
  Stored in directory: /tmp/pip-ephem-wheel-cache-t4sx7vof/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:43:31, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:44:29, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:22:20, 23.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:16:08, 32.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:04:32, 47.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.86M/862M [00:01<3:31:43, 67.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.6M/862M [00:01<2:27:18, 95.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.3M/862M [00:01<1:42:31, 137kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.0M/862M [00:01<1:11:22, 195kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.7M/862M [00:01<49:46, 278kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:01<34:47, 396kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.2M/862M [00:02<24:18, 564kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.2M/862M [00:02<17:01, 800kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.7M/862M [00:02<11:56, 1.13MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.1M/862M [00:02<08:24, 1.60MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.4M/862M [00:03<16:58, 794kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:03<12:04, 1.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:05<12:38, 1.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<11:48, 1.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:05<08:52, 1.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:05<06:21, 2.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:07<13:28, 990kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:07<10:48, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:07<07:51, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<08:33, 1.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:09<07:20, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:09<05:25, 2.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<06:53, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:11<07:29, 1.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<05:54, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:13<06:15, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:13<05:44, 2.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:13<04:17, 3.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:15<06:02, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:15<05:32, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:15<04:10, 3.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<05:57, 2.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:17<06:43, 1.93MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:17<05:17, 2.46MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:17<03:51, 3.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:19<09:34, 1.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:19<08:02, 1.61MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.3M/862M [00:19<05:53, 2.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:21<07:08, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:21<06:17, 2.04MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:21<04:43, 2.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:23<06:19, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:23<07:03, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<05:34, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<05:57, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:25<05:27, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:08, 3.06MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:50, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<06:45, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:17, 2.39MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<03:54, 3.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:09, 1.76MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<06:07, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<04:36, 2.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:07, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<07:12, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<05:45, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<04:09, 2.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<12:00, 1.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<09:48, 1.27MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<07:12, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:42, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<08:07, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<06:18, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<04:33, 2.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<08:50, 1.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:34, 1.63MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<05:38, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:35, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<07:21, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:51, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<04:15, 2.86MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<15:08, 803kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<11:56, 1.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<08:40, 1.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<08:43, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<08:47, 1.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<06:51, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<04:55, 2.44MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<13:35, 886kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<10:50, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<07:54, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:06, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<08:26, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:28, 1.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<04:45, 2.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:39, 1.79MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:00, 1.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<04:28, 2.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:40, 2.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:42, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<05:16, 2.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<03:51, 3.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:15, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:25, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<04:45, 2.46MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:49, 2.01MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:48, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<05:21, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<03:54, 2.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<13:03, 889kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<10:27, 1.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<07:37, 1.52MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:49, 1.48MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<08:02, 1.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<06:11, 1.86MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:34, 2.52MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:11, 1.85MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:38, 2.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<04:15, 2.69MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:26, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:03, 2.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:48, 2.98MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:06, 2.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:07, 1.85MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<04:51, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<03:31, 3.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<09:04, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<07:25, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<05:29, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:15, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:00, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:27, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:57, 2.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<07:12, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:17, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:40, 2.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:38, 1.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:11, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:53, 2.84MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:06, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:02, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<04:45, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<03:28, 3.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:07, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:13, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:39, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:35, 1.94MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<06:20, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<05:00, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<03:36, 2.99MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<10:17, 1.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<08:24, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<06:10, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:36, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<05:50, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:22, 2.44MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:21, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:12, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:51, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:24<03:32, 2.99MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:25, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:40, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<04:15, 2.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:15, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:07, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:48, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<03:32, 2.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:45, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:12, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:53, 2.68MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:57, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:52, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:36, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<03:20, 3.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:49, 1.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:38, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<04:53, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:37, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<06:11, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:50, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<03:30, 2.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:58, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:52, 1.73MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<04:23, 2.31MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:15, 1.92MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:55, 1.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:37, 2.17MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<03:21, 2.98MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<06:57, 1.44MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:47, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<04:19, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:10, 1.92MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:50, 1.70MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:34, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:20, 2.96MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:47, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:08, 1.92MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:52, 2.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:49, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:38, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:24, 2.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<03:11, 3.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<07:52, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<06:36, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<04:53, 1.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:29, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<06:04, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:47, 2.01MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<03:28, 2.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<11:53, 807kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:24, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<06:47, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:46, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:56, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:19, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<03:49, 2.48MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:05, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:44, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:55, 1.91MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:28, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:48, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:29, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:00<03:15, 2.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:41, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:42, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:14, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:57, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:36, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:22, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<03:11, 2.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:00, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:13, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:54, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:41, 1.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:11, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:06, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<02:58, 3.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<20:46, 436kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<15:30, 583kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<11:03, 815kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<09:43, 923kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:49, 1.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:42, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:53, 1.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:12, 1.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:47, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<03:27, 2.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<07:01, 1.26MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:52, 1.50MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<04:20, 2.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:00, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:31, 1.59MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:16, 2.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<03:07, 2.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:21, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:44, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:33, 2.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:20, 1.99MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:02, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:00, 2.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:54, 2.94MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<10:29, 816kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<08:16, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<06:00, 1.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:04, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:15, 1.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:51, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<03:30, 2.41MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<09:46, 862kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<07:43, 1.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:36, 1.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:47, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:49, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:27, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:12, 2.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<06:50, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:42, 1.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:13, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:42, 1.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:06, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:58, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<02:53, 2.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<09:30, 856kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<07:32, 1.08MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:29, 1.48MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:34, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:45, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:25, 1.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<03:10, 2.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<06:20, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:20, 1.50MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<03:57, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:27, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:47, 1.65MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:42, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:42, 2.91MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:41, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:09, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:05, 2.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:51, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:25, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:29, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<02:33, 3.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:20, 1.78MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<03:46, 2.05MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:48, 2.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<02:04, 3.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<07:24, 1.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:57, 1.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:17, 1.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:47, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<06:08, 1.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:05, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:43, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:41, 2.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<7:34:41, 16.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<5:19:43, 23.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<3:43:50, 33.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<2:35:43, 47.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<2:05:41, 59.3kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<1:28:44, 83.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<1:02:07, 119kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<44:51, 165kB/s]  .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<32:56, 224kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<23:21, 315kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<16:22, 448kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<14:35, 501kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<10:59, 665kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<07:51, 926kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<07:05, 1.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:33, 1.10MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:59, 1.45MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<03:34, 2.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<09:34, 749kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<07:27, 962kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:23, 1.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:20, 1.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:14, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:58, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:53, 2.44MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:22, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:51, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<02:52, 2.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:30, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:55, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:05, 2.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:13, 3.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<11:12, 616kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<08:33, 805kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<06:09, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:49, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<05:31, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:10, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:59, 2.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<05:51, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:48, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<03:30, 1.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:56, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:18, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:22, 1.97MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<02:26, 2.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<07:47, 849kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:11, 1.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<04:30, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:33, 1.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:37, 1.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:34, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<02:34, 2.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<07:45, 837kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<06:09, 1.05MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<04:28, 1.44MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:29, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:51, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:50, 2.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:21, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:48, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:58, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:27<02:08, 2.93MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<05:59, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:52, 1.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<03:34, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:49, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:06, 1.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:10, 1.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<02:18, 2.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:44, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:18, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:27, 2.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:01, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:45, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:04, 2.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:46, 2.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:15, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:34, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<01:52, 3.18MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 507M/862M [03:39<03:45, 1.58MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:16, 1.81MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:27, 2.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:58, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:25, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:42, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<01:57, 2.96MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<06:13, 930kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:53, 1.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<03:34, 1.61MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:43, 1.53MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:55, 1.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:02, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<02:11, 2.58MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<06:42, 841kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<05:19, 1.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<03:50, 1.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:53, 1.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<04:00, 1.39MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:06, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<02:14, 2.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<06:37, 833kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<05:14, 1.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:48, 1.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:49, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:49, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:57, 1.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<02:07, 2.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<18:32, 290kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<13:32, 397kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<09:34, 558kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<07:50, 676kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<06:43, 789kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:56, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<03:30, 1.50MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:37, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:49, 1.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:47, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:03, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:20, 1.55MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:34, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<01:51, 2.76MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<04:35, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:45, 1.35MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:43, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:01, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:16, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:34, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:50, 2.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<05:55, 838kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:39, 1.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<03:22, 1.46MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:26, 1.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:32, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:43, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<01:57, 2.45MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<05:22, 895kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:18, 1.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<03:07, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:12, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:17, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:33, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<01:50, 2.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<05:30, 849kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:21, 1.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:08, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:12, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:18, 1.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:32, 1.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:49, 2.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<05:05, 890kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:03, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:57, 1.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:00, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:36, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:55, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:17, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:04, 2.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:33, 2.81MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:03, 2.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:26, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:55, 2.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:23, 3.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:46, 893kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:48, 1.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:45, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:48, 1.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:56, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:17, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:29<01:38, 2.52MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:55, 836kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:54, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:50, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:50, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:25, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:48, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:06, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:54, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:26, 2.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:50, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:12, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:45, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:15, 3.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:25, 870kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:31, 1.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:33, 1.49MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:35, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:38, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:03, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:28, 2.51MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<04:40, 791kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:40, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:38, 1.39MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:38, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:15, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:40, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:55, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:06, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:37, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:10, 3.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<03:13, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:37, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:54, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:05, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:15, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:44, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:14, 2.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:54, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:24, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:46, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:55, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:41, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:15, 2.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:35, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:49, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:26, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:01, 3.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:30, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:06, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:32, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [04:59<01:06, 2.77MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:23, 906kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:01, 1.02MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:15, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:36, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:36, 1.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:10, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:34, 1.89MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:43, 1.71MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:53, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:29, 1.97MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:03, 2.72MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<03:22, 850kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<02:41, 1.07MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:56, 1.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:56, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:59, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:31, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:05, 2.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:01, 904kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:25, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:45, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:47, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:50, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:24, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:01, 2.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:28, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:16, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<00:59, 2.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:43, 3.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:55, 863kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:38, 957kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:58, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:23, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<03:23, 724kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:38, 928kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:53, 1.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:49, 1.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:49, 1.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:24, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:59, 2.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<02:51, 812kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<02:15, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:37, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:36, 1.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:35, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:13, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:52, 2.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<07:10, 304kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<05:15, 413kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<03:41, 583kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:59, 704kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:32, 830kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<01:52, 1.12MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<01:18, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<07:03, 289kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<05:06, 399kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<03:35, 561kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:54, 679kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:26, 805kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:47, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:32<01:14, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<04:16, 446kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<03:10, 597kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<02:14, 834kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:57, 940kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:34, 1.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<01:08, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:09, 1.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:11, 1.47MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:55, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:39, 2.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:16, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:04, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:47, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:53, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:59, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:46, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:33, 2.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:54, 812kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:30, 1.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:04, 1.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:03, 1.40MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:03, 1.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:48, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:46<00:33, 2.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<04:39, 304kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<03:24, 414kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<02:22, 583kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:54, 706kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:37, 824kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:11, 1.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<00:49, 1.56MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:35, 807kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:14, 1.02MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:53, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:51, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:42, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:31, 2.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:36, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:41, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:31, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:22, 2.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:57, 1.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:46, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:33, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:35, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:38, 1.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:29, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:20, 2.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:42, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:36, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:26, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:28, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:30, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:23, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:16, 2.91MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<02:42, 292kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<01:58, 399kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<01:21, 561kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<01:03, 684kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:54, 800kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:39, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:26, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:54, 722kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:42, 926kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:29, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:20, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:13, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:19, 1.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:17, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:13, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:15, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:07, 3.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:19, 1.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:15, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:10, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:11, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:08, 2.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:05, 2.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:22, 633kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:17, 820kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:11, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:08, 1.20MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:08, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.61MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:02, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 831kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:05, 1.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.42MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.82MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<97:36:12,  1.14it/s]  0%|          | 808/400000 [00:00<68:11:19,  1.63it/s]  0%|          | 1680/400000 [00:01<47:37:53,  2.32it/s]  1%|          | 2540/400000 [00:01<33:16:26,  3.32it/s]  1%|          | 3408/400000 [00:01<23:14:41,  4.74it/s]  1%|          | 4271/400000 [00:01<16:14:23,  6.77it/s]  1%|▏         | 5140/400000 [00:01<11:20:47,  9.67it/s]  2%|▏         | 6012/400000 [00:01<7:55:43, 13.80it/s]   2%|▏         | 6878/400000 [00:01<5:32:30, 19.70it/s]  2%|▏         | 7737/400000 [00:01<3:52:28, 28.12it/s]  2%|▏         | 8566/400000 [00:01<2:42:37, 40.12it/s]  2%|▏         | 9402/400000 [00:01<1:53:49, 57.19it/s]  3%|▎         | 10272/400000 [00:02<1:19:43, 81.47it/s]  3%|▎         | 11126/400000 [00:02<55:54, 115.91it/s]   3%|▎         | 12000/400000 [00:02<39:16, 164.65it/s]  3%|▎         | 12868/400000 [00:02<27:39, 233.32it/s]  3%|▎         | 13726/400000 [00:02<19:33, 329.23it/s]  4%|▎         | 14588/400000 [00:02<13:52, 462.75it/s]  4%|▍         | 15453/400000 [00:02<09:55, 646.25it/s]  4%|▍         | 16306/400000 [00:02<07:09, 892.69it/s]  4%|▍         | 17147/400000 [00:02<05:13, 1219.73it/s]  4%|▍         | 17990/400000 [00:02<03:52, 1640.63it/s]  5%|▍         | 18864/400000 [00:03<02:55, 2169.19it/s]  5%|▍         | 19743/400000 [00:03<02:15, 2802.38it/s]  5%|▌         | 20623/400000 [00:03<01:47, 3522.23it/s]  5%|▌         | 21493/400000 [00:03<01:28, 4287.11it/s]  6%|▌         | 22361/400000 [00:03<01:14, 5053.61it/s]  6%|▌         | 23231/400000 [00:03<01:05, 5780.30it/s]  6%|▌         | 24112/400000 [00:03<00:58, 6444.38it/s]  6%|▌         | 24984/400000 [00:03<00:54, 6913.66it/s]  6%|▋         | 25852/400000 [00:03<00:50, 7362.63it/s]  7%|▋         | 26715/400000 [00:03<00:48, 7670.42it/s]  7%|▋         | 27574/400000 [00:04<00:47, 7880.90it/s]  7%|▋         | 28443/400000 [00:04<00:45, 8105.52it/s]  7%|▋         | 29308/400000 [00:04<00:44, 8261.15it/s]  8%|▊         | 30181/400000 [00:04<00:44, 8394.03it/s]  8%|▊         | 31045/400000 [00:04<00:43, 8387.17it/s]  8%|▊         | 31901/400000 [00:04<00:44, 8362.07it/s]  8%|▊         | 32770/400000 [00:04<00:43, 8457.33it/s]  8%|▊         | 33642/400000 [00:04<00:42, 8532.97it/s]  9%|▊         | 34502/400000 [00:04<00:42, 8529.21it/s]  9%|▉         | 35374/400000 [00:05<00:42, 8584.65it/s]  9%|▉         | 36248/400000 [00:05<00:42, 8628.06it/s]  9%|▉         | 37118/400000 [00:05<00:41, 8647.85it/s]  9%|▉         | 37985/400000 [00:05<00:42, 8496.68it/s] 10%|▉         | 38837/400000 [00:05<00:43, 8327.74it/s] 10%|▉         | 39678/400000 [00:05<00:43, 8351.15it/s] 10%|█         | 40539/400000 [00:05<00:42, 8426.06it/s] 10%|█         | 41408/400000 [00:05<00:42, 8502.48it/s] 11%|█         | 42285/400000 [00:05<00:41, 8580.66it/s] 11%|█         | 43162/400000 [00:05<00:41, 8635.53it/s] 11%|█         | 44027/400000 [00:06<00:41, 8580.94it/s] 11%|█         | 44886/400000 [00:06<00:41, 8568.78it/s] 11%|█▏        | 45758/400000 [00:06<00:41, 8610.87it/s] 12%|█▏        | 46630/400000 [00:06<00:40, 8642.41it/s] 12%|█▏        | 47507/400000 [00:06<00:40, 8679.52it/s] 12%|█▏        | 48376/400000 [00:06<00:40, 8588.06it/s] 12%|█▏        | 49236/400000 [00:06<00:41, 8453.67it/s] 13%|█▎        | 50083/400000 [00:06<00:41, 8384.72it/s] 13%|█▎        | 50937/400000 [00:06<00:41, 8428.61it/s] 13%|█▎        | 51798/400000 [00:06<00:41, 8481.21it/s] 13%|█▎        | 52654/400000 [00:07<00:40, 8504.59it/s] 13%|█▎        | 53505/400000 [00:07<00:40, 8475.56it/s] 14%|█▎        | 54353/400000 [00:07<00:40, 8474.65it/s] 14%|█▍        | 55220/400000 [00:07<00:40, 8531.73it/s] 14%|█▍        | 56086/400000 [00:07<00:40, 8569.42it/s] 14%|█▍        | 56944/400000 [00:07<00:40, 8560.48it/s] 14%|█▍        | 57801/400000 [00:07<00:40, 8552.83it/s] 15%|█▍        | 58665/400000 [00:07<00:39, 8578.12it/s] 15%|█▍        | 59531/400000 [00:07<00:39, 8602.35it/s] 15%|█▌        | 60399/400000 [00:07<00:39, 8625.00it/s] 15%|█▌        | 61262/400000 [00:08<00:39, 8609.36it/s] 16%|█▌        | 62123/400000 [00:08<00:40, 8312.95it/s] 16%|█▌        | 62996/400000 [00:08<00:39, 8432.11it/s] 16%|█▌        | 63871/400000 [00:08<00:39, 8522.56it/s] 16%|█▌        | 64733/400000 [00:08<00:39, 8549.21it/s] 16%|█▋        | 65590/400000 [00:08<00:39, 8468.18it/s] 17%|█▋        | 66438/400000 [00:08<00:39, 8404.80it/s] 17%|█▋        | 67280/400000 [00:08<00:39, 8400.45it/s] 17%|█▋        | 68145/400000 [00:08<00:39, 8472.46it/s] 17%|█▋        | 69012/400000 [00:08<00:38, 8530.15it/s] 17%|█▋        | 69878/400000 [00:09<00:38, 8568.51it/s] 18%|█▊        | 70741/400000 [00:09<00:38, 8585.15it/s] 18%|█▊        | 71611/400000 [00:09<00:38, 8618.18it/s] 18%|█▊        | 72475/400000 [00:09<00:37, 8623.11it/s] 18%|█▊        | 73345/400000 [00:09<00:37, 8646.00it/s] 19%|█▊        | 74210/400000 [00:09<00:37, 8617.05it/s] 19%|█▉        | 75072/400000 [00:09<00:37, 8587.23it/s] 19%|█▉        | 75931/400000 [00:09<00:38, 8421.08it/s] 19%|█▉        | 76774/400000 [00:09<00:38, 8421.77it/s] 19%|█▉        | 77635/400000 [00:09<00:38, 8476.22it/s] 20%|█▉        | 78490/400000 [00:10<00:37, 8495.83it/s] 20%|█▉        | 79340/400000 [00:10<00:37, 8484.40it/s] 20%|██        | 80205/400000 [00:10<00:37, 8532.57it/s] 20%|██        | 81059/400000 [00:10<00:37, 8505.78it/s] 20%|██        | 81910/400000 [00:10<00:37, 8478.89it/s] 21%|██        | 82780/400000 [00:10<00:37, 8543.06it/s] 21%|██        | 83635/400000 [00:10<00:38, 8320.37it/s] 21%|██        | 84498/400000 [00:10<00:37, 8409.32it/s] 21%|██▏       | 85360/400000 [00:10<00:37, 8471.05it/s] 22%|██▏       | 86209/400000 [00:10<00:37, 8312.67it/s] 22%|██▏       | 87064/400000 [00:11<00:37, 8381.96it/s] 22%|██▏       | 87917/400000 [00:11<00:37, 8425.82it/s] 22%|██▏       | 88761/400000 [00:11<00:37, 8379.35it/s] 22%|██▏       | 89610/400000 [00:11<00:36, 8409.46it/s] 23%|██▎       | 90468/400000 [00:11<00:36, 8458.67it/s] 23%|██▎       | 91318/400000 [00:11<00:36, 8469.12it/s] 23%|██▎       | 92172/400000 [00:11<00:36, 8487.56it/s] 23%|██▎       | 93032/400000 [00:11<00:36, 8518.48it/s] 23%|██▎       | 93900/400000 [00:11<00:35, 8565.05it/s] 24%|██▎       | 94770/400000 [00:11<00:35, 8604.88it/s] 24%|██▍       | 95633/400000 [00:12<00:35, 8611.76it/s] 24%|██▍       | 96495/400000 [00:12<00:35, 8570.43it/s] 24%|██▍       | 97353/400000 [00:12<00:35, 8555.65it/s] 25%|██▍       | 98221/400000 [00:12<00:35, 8590.08it/s] 25%|██▍       | 99081/400000 [00:12<00:35, 8528.69it/s] 25%|██▍       | 99955/400000 [00:12<00:34, 8590.41it/s] 25%|██▌       | 100815/400000 [00:12<00:35, 8501.63it/s] 25%|██▌       | 101666/400000 [00:12<00:35, 8438.42it/s] 26%|██▌       | 102547/400000 [00:12<00:34, 8546.17it/s] 26%|██▌       | 103425/400000 [00:13<00:34, 8613.22it/s] 26%|██▌       | 104297/400000 [00:13<00:34, 8642.11it/s] 26%|██▋       | 105162/400000 [00:13<00:34, 8465.56it/s] 27%|██▋       | 106030/400000 [00:13<00:34, 8527.53it/s] 27%|██▋       | 106905/400000 [00:13<00:34, 8592.33it/s] 27%|██▋       | 107778/400000 [00:13<00:33, 8632.27it/s] 27%|██▋       | 108661/400000 [00:13<00:33, 8690.48it/s] 27%|██▋       | 109531/400000 [00:13<00:33, 8653.49it/s] 28%|██▊       | 110415/400000 [00:13<00:33, 8705.69it/s] 28%|██▊       | 111297/400000 [00:13<00:33, 8739.23it/s] 28%|██▊       | 112172/400000 [00:14<00:33, 8697.57it/s] 28%|██▊       | 113059/400000 [00:14<00:32, 8747.19it/s] 28%|██▊       | 113934/400000 [00:14<00:33, 8636.64it/s] 29%|██▊       | 114812/400000 [00:14<00:32, 8678.79it/s] 29%|██▉       | 115681/400000 [00:14<00:32, 8656.57it/s] 29%|██▉       | 116553/400000 [00:14<00:32, 8675.14it/s] 29%|██▉       | 117421/400000 [00:14<00:32, 8641.13it/s] 30%|██▉       | 118286/400000 [00:14<00:32, 8613.35it/s] 30%|██▉       | 119148/400000 [00:14<00:32, 8559.69it/s] 30%|███       | 120025/400000 [00:14<00:32, 8618.88it/s] 30%|███       | 120888/400000 [00:15<00:32, 8490.67it/s] 30%|███       | 121774/400000 [00:15<00:32, 8597.47it/s] 31%|███       | 122635/400000 [00:15<00:32, 8601.06it/s] 31%|███       | 123522/400000 [00:15<00:31, 8679.47it/s] 31%|███       | 124400/400000 [00:15<00:31, 8707.56it/s] 31%|███▏      | 125278/400000 [00:15<00:31, 8727.18it/s] 32%|███▏      | 126154/400000 [00:15<00:31, 8729.15it/s] 32%|███▏      | 127028/400000 [00:15<00:31, 8615.63it/s] 32%|███▏      | 127891/400000 [00:15<00:31, 8544.00it/s] 32%|███▏      | 128762/400000 [00:15<00:31, 8591.43it/s] 32%|███▏      | 129631/400000 [00:16<00:31, 8618.87it/s] 33%|███▎      | 130512/400000 [00:16<00:31, 8673.64it/s] 33%|███▎      | 131380/400000 [00:16<00:31, 8664.22it/s] 33%|███▎      | 132268/400000 [00:16<00:30, 8727.74it/s] 33%|███▎      | 133165/400000 [00:16<00:30, 8798.79it/s] 34%|███▎      | 134058/400000 [00:16<00:30, 8835.89it/s] 34%|███▎      | 134942/400000 [00:16<00:30, 8738.20it/s] 34%|███▍      | 135818/400000 [00:16<00:30, 8743.42it/s] 34%|███▍      | 136693/400000 [00:16<00:30, 8697.19it/s] 34%|███▍      | 137578/400000 [00:16<00:30, 8742.44it/s] 35%|███▍      | 138453/400000 [00:17<00:30, 8661.53it/s] 35%|███▍      | 139330/400000 [00:17<00:29, 8692.68it/s] 35%|███▌      | 140207/400000 [00:17<00:29, 8714.75it/s] 35%|███▌      | 141079/400000 [00:17<00:29, 8702.92it/s] 35%|███▌      | 141961/400000 [00:17<00:29, 8736.53it/s] 36%|███▌      | 142835/400000 [00:17<00:29, 8689.63it/s] 36%|███▌      | 143716/400000 [00:17<00:29, 8723.01it/s] 36%|███▌      | 144589/400000 [00:17<00:29, 8688.68it/s] 36%|███▋      | 145468/400000 [00:17<00:29, 8716.90it/s] 37%|███▋      | 146366/400000 [00:17<00:28, 8792.36it/s] 37%|███▋      | 147251/400000 [00:18<00:28, 8806.65it/s] 37%|███▋      | 148133/400000 [00:18<00:28, 8809.50it/s] 37%|███▋      | 149015/400000 [00:18<00:28, 8808.33it/s] 37%|███▋      | 149906/400000 [00:18<00:28, 8837.98it/s] 38%|███▊      | 150790/400000 [00:18<00:28, 8837.06it/s] 38%|███▊      | 151674/400000 [00:18<00:28, 8834.33it/s] 38%|███▊      | 152558/400000 [00:18<00:28, 8825.10it/s] 38%|███▊      | 153441/400000 [00:18<00:27, 8819.89it/s] 39%|███▊      | 154324/400000 [00:18<00:28, 8529.85it/s] 39%|███▉      | 155198/400000 [00:18<00:28, 8589.30it/s] 39%|███▉      | 156080/400000 [00:19<00:28, 8655.22it/s] 39%|███▉      | 156954/400000 [00:19<00:28, 8678.56it/s] 39%|███▉      | 157823/400000 [00:19<00:28, 8561.18it/s] 40%|███▉      | 158698/400000 [00:19<00:28, 8614.11it/s] 40%|███▉      | 159579/400000 [00:19<00:27, 8669.89it/s] 40%|████      | 160450/400000 [00:19<00:27, 8681.54it/s] 40%|████      | 161319/400000 [00:19<00:27, 8597.41it/s] 41%|████      | 162180/400000 [00:19<00:28, 8457.36it/s] 41%|████      | 163036/400000 [00:19<00:27, 8487.71it/s] 41%|████      | 163912/400000 [00:19<00:27, 8566.20it/s] 41%|████      | 164799/400000 [00:20<00:27, 8653.87it/s] 41%|████▏     | 165684/400000 [00:20<00:26, 8709.85it/s] 42%|████▏     | 166556/400000 [00:20<00:26, 8699.15it/s] 42%|████▏     | 167434/400000 [00:20<00:26, 8720.58it/s] 42%|████▏     | 168321/400000 [00:20<00:26, 8763.24it/s] 42%|████▏     | 169207/400000 [00:20<00:26, 8790.60it/s] 43%|████▎     | 170089/400000 [00:20<00:26, 8799.33it/s] 43%|████▎     | 170970/400000 [00:20<00:26, 8759.83it/s] 43%|████▎     | 171847/400000 [00:20<00:26, 8706.92it/s] 43%|████▎     | 172718/400000 [00:20<00:26, 8603.49it/s] 43%|████▎     | 173582/400000 [00:21<00:26, 8613.27it/s] 44%|████▎     | 174452/400000 [00:21<00:26, 8636.35it/s] 44%|████▍     | 175326/400000 [00:21<00:25, 8666.88it/s] 44%|████▍     | 176204/400000 [00:21<00:25, 8700.16it/s] 44%|████▍     | 177091/400000 [00:21<00:25, 8747.81it/s] 44%|████▍     | 177972/400000 [00:21<00:25, 8764.07it/s] 45%|████▍     | 178849/400000 [00:21<00:25, 8637.20it/s] 45%|████▍     | 179714/400000 [00:21<00:25, 8495.03it/s] 45%|████▌     | 180566/400000 [00:21<00:25, 8500.58it/s] 45%|████▌     | 181440/400000 [00:21<00:25, 8570.05it/s] 46%|████▌     | 182309/400000 [00:22<00:25, 8605.14it/s] 46%|████▌     | 183193/400000 [00:22<00:24, 8672.29it/s] 46%|████▌     | 184064/400000 [00:22<00:24, 8680.69it/s] 46%|████▌     | 184940/400000 [00:22<00:24, 8703.50it/s] 46%|████▋     | 185811/400000 [00:22<00:24, 8667.92it/s] 47%|████▋     | 186678/400000 [00:22<00:24, 8614.83it/s] 47%|████▋     | 187540/400000 [00:22<00:24, 8611.83it/s] 47%|████▋     | 188402/400000 [00:22<00:24, 8607.01it/s] 47%|████▋     | 189270/400000 [00:22<00:24, 8626.92it/s] 48%|████▊     | 190139/400000 [00:23<00:24, 8644.51it/s] 48%|████▊     | 191004/400000 [00:23<00:24, 8613.18it/s] 48%|████▊     | 191872/400000 [00:23<00:24, 8630.26it/s] 48%|████▊     | 192736/400000 [00:23<00:24, 8627.63it/s] 48%|████▊     | 193607/400000 [00:23<00:23, 8651.27it/s] 49%|████▊     | 194473/400000 [00:23<00:23, 8606.69it/s] 49%|████▉     | 195334/400000 [00:23<00:23, 8563.81it/s] 49%|████▉     | 196204/400000 [00:23<00:23, 8602.58it/s] 49%|████▉     | 197065/400000 [00:23<00:23, 8593.14it/s] 49%|████▉     | 197925/400000 [00:23<00:23, 8573.74it/s] 50%|████▉     | 198786/400000 [00:24<00:23, 8582.40it/s] 50%|████▉     | 199653/400000 [00:24<00:23, 8606.69it/s] 50%|█████     | 200530/400000 [00:24<00:23, 8652.40it/s] 50%|█████     | 201399/400000 [00:24<00:22, 8662.33it/s] 51%|█████     | 202279/400000 [00:24<00:22, 8701.74it/s] 51%|█████     | 203150/400000 [00:24<00:22, 8609.15it/s] 51%|█████     | 204012/400000 [00:24<00:23, 8375.73it/s] 51%|█████     | 204890/400000 [00:24<00:22, 8490.47it/s] 51%|█████▏    | 205746/400000 [00:24<00:22, 8508.38it/s] 52%|█████▏    | 206619/400000 [00:24<00:22, 8570.73it/s] 52%|█████▏    | 207496/400000 [00:25<00:22, 8629.24it/s] 52%|█████▏    | 208361/400000 [00:25<00:22, 8634.73it/s] 52%|█████▏    | 209230/400000 [00:25<00:22, 8650.35it/s] 53%|█████▎    | 210103/400000 [00:25<00:21, 8671.29it/s] 53%|█████▎    | 210971/400000 [00:25<00:21, 8655.89it/s] 53%|█████▎    | 211849/400000 [00:25<00:21, 8691.42it/s] 53%|█████▎    | 212719/400000 [00:25<00:21, 8667.83it/s] 53%|█████▎    | 213586/400000 [00:25<00:21, 8656.54it/s] 54%|█████▎    | 214452/400000 [00:25<00:21, 8650.63it/s] 54%|█████▍    | 215322/400000 [00:25<00:21, 8662.64it/s] 54%|█████▍    | 216202/400000 [00:26<00:21, 8700.65it/s] 54%|█████▍    | 217073/400000 [00:26<00:21, 8480.18it/s] 54%|█████▍    | 217946/400000 [00:26<00:21, 8552.80it/s] 55%|█████▍    | 218812/400000 [00:26<00:21, 8583.11it/s] 55%|█████▍    | 219682/400000 [00:26<00:20, 8617.61it/s] 55%|█████▌    | 220555/400000 [00:26<00:20, 8650.06it/s] 55%|█████▌    | 221421/400000 [00:26<00:20, 8543.90it/s] 56%|█████▌    | 222276/400000 [00:26<00:20, 8544.46it/s] 56%|█████▌    | 223134/400000 [00:26<00:20, 8552.66it/s] 56%|█████▌    | 223996/400000 [00:26<00:20, 8569.86it/s] 56%|█████▌    | 224863/400000 [00:27<00:20, 8598.08it/s] 56%|█████▋    | 225732/400000 [00:27<00:20, 8624.64it/s] 57%|█████▋    | 226598/400000 [00:27<00:20, 8630.41it/s] 57%|█████▋    | 227462/400000 [00:27<00:20, 8571.02it/s] 57%|█████▋    | 228334/400000 [00:27<00:19, 8613.45it/s] 57%|█████▋    | 229200/400000 [00:27<00:19, 8625.51it/s] 58%|█████▊    | 230077/400000 [00:27<00:19, 8667.92it/s] 58%|█████▊    | 230950/400000 [00:27<00:19, 8684.45it/s] 58%|█████▊    | 231819/400000 [00:27<00:19, 8544.24it/s] 58%|█████▊    | 232675/400000 [00:27<00:19, 8543.15it/s] 58%|█████▊    | 233537/400000 [00:28<00:19, 8564.04it/s] 59%|█████▊    | 234406/400000 [00:28<00:19, 8601.20it/s] 59%|█████▉    | 235276/400000 [00:28<00:19, 8630.23it/s] 59%|█████▉    | 236140/400000 [00:28<00:19, 8563.38it/s] 59%|█████▉    | 237001/400000 [00:28<00:19, 8575.39it/s] 59%|█████▉    | 237859/400000 [00:28<00:19, 8429.49it/s] 60%|█████▉    | 238724/400000 [00:28<00:18, 8494.36it/s] 60%|█████▉    | 239590/400000 [00:28<00:18, 8543.16it/s] 60%|██████    | 240459/400000 [00:28<00:18, 8585.64it/s] 60%|██████    | 241326/400000 [00:28<00:18, 8609.16it/s] 61%|██████    | 242192/400000 [00:29<00:18, 8621.44it/s] 61%|██████    | 243064/400000 [00:29<00:18, 8648.73it/s] 61%|██████    | 243945/400000 [00:29<00:17, 8694.86it/s] 61%|██████    | 244815/400000 [00:29<00:17, 8675.79it/s] 61%|██████▏   | 245683/400000 [00:29<00:17, 8658.90it/s] 62%|██████▏   | 246552/400000 [00:29<00:17, 8665.28it/s] 62%|██████▏   | 247425/400000 [00:29<00:17, 8683.26it/s] 62%|██████▏   | 248294/400000 [00:29<00:17, 8574.92it/s] 62%|██████▏   | 249152/400000 [00:29<00:17, 8572.87it/s] 63%|██████▎   | 250014/400000 [00:29<00:17, 8584.87it/s] 63%|██████▎   | 250873/400000 [00:30<00:17, 8539.20it/s] 63%|██████▎   | 251750/400000 [00:30<00:17, 8606.28it/s] 63%|██████▎   | 252621/400000 [00:30<00:17, 8636.85it/s] 63%|██████▎   | 253485/400000 [00:30<00:16, 8628.78it/s] 64%|██████▎   | 254350/400000 [00:30<00:16, 8633.18it/s] 64%|██████▍   | 255214/400000 [00:30<00:16, 8592.86it/s] 64%|██████▍   | 256074/400000 [00:30<00:16, 8486.09it/s] 64%|██████▍   | 256937/400000 [00:30<00:16, 8526.49it/s] 64%|██████▍   | 257795/400000 [00:30<00:16, 8541.58it/s] 65%|██████▍   | 258659/400000 [00:30<00:16, 8569.00it/s] 65%|██████▍   | 259524/400000 [00:31<00:16, 8592.79it/s] 65%|██████▌   | 260387/400000 [00:31<00:16, 8602.48it/s] 65%|██████▌   | 261259/400000 [00:31<00:16, 8636.65it/s] 66%|██████▌   | 262130/400000 [00:31<00:15, 8657.33it/s] 66%|██████▌   | 262997/400000 [00:31<00:15, 8658.60it/s] 66%|██████▌   | 263863/400000 [00:31<00:15, 8653.73it/s] 66%|██████▌   | 264733/400000 [00:31<00:15, 8667.16it/s] 66%|██████▋   | 265601/400000 [00:31<00:15, 8669.29it/s] 67%|██████▋   | 266471/400000 [00:31<00:15, 8676.63it/s] 67%|██████▋   | 267339/400000 [00:31<00:15, 8672.65it/s] 67%|██████▋   | 268211/400000 [00:32<00:15, 8684.48it/s] 67%|██████▋   | 269080/400000 [00:32<00:15, 8576.34it/s] 67%|██████▋   | 269938/400000 [00:32<00:15, 8562.00it/s] 68%|██████▊   | 270802/400000 [00:32<00:15, 8584.71it/s] 68%|██████▊   | 271666/400000 [00:32<00:14, 8599.44it/s] 68%|██████▊   | 272527/400000 [00:32<00:14, 8589.12it/s] 68%|██████▊   | 273393/400000 [00:32<00:14, 8608.54it/s] 69%|██████▊   | 274265/400000 [00:32<00:14, 8639.19it/s] 69%|██████▉   | 275131/400000 [00:32<00:14, 8644.43it/s] 69%|██████▉   | 275996/400000 [00:32<00:14, 8621.54it/s] 69%|██████▉   | 276866/400000 [00:33<00:14, 8642.94it/s] 69%|██████▉   | 277731/400000 [00:33<00:14, 8639.82it/s] 70%|██████▉   | 278602/400000 [00:33<00:14, 8658.75it/s] 70%|██████▉   | 279468/400000 [00:33<00:13, 8631.70it/s] 70%|███████   | 280332/400000 [00:33<00:13, 8592.68it/s] 70%|███████   | 281199/400000 [00:33<00:13, 8614.30it/s] 71%|███████   | 282069/400000 [00:33<00:13, 8639.44it/s] 71%|███████   | 282940/400000 [00:33<00:13, 8660.10it/s] 71%|███████   | 283807/400000 [00:33<00:13, 8638.93it/s] 71%|███████   | 284671/400000 [00:33<00:13, 8531.97it/s] 71%|███████▏  | 285534/400000 [00:34<00:13, 8560.98it/s] 72%|███████▏  | 286400/400000 [00:34<00:13, 8589.28it/s] 72%|███████▏  | 287265/400000 [00:34<00:13, 8601.05it/s] 72%|███████▏  | 288126/400000 [00:34<00:13, 8570.06it/s] 72%|███████▏  | 288984/400000 [00:34<00:13, 8388.63it/s] 72%|███████▏  | 289856/400000 [00:34<00:12, 8485.37it/s] 73%|███████▎  | 290728/400000 [00:34<00:12, 8552.20it/s] 73%|███████▎  | 291590/400000 [00:34<00:12, 8570.05it/s] 73%|███████▎  | 292467/400000 [00:34<00:12, 8627.13it/s] 73%|███████▎  | 293336/400000 [00:34<00:12, 8643.46it/s] 74%|███████▎  | 294209/400000 [00:35<00:12, 8666.38it/s] 74%|███████▍  | 295076/400000 [00:35<00:12, 8641.78it/s] 74%|███████▍  | 295944/400000 [00:35<00:12, 8652.79it/s] 74%|███████▍  | 296810/400000 [00:35<00:12, 8535.72it/s] 74%|███████▍  | 297668/400000 [00:35<00:11, 8547.38it/s] 75%|███████▍  | 298542/400000 [00:35<00:11, 8602.07it/s] 75%|███████▍  | 299403/400000 [00:35<00:11, 8581.45it/s] 75%|███████▌  | 300262/400000 [00:35<00:11, 8535.06it/s] 75%|███████▌  | 301139/400000 [00:35<00:11, 8602.92it/s] 76%|███████▌  | 302008/400000 [00:36<00:11, 8628.48it/s] 76%|███████▌  | 302872/400000 [00:36<00:11, 8570.30it/s] 76%|███████▌  | 303730/400000 [00:36<00:11, 8563.64it/s] 76%|███████▌  | 304587/400000 [00:36<00:11, 8519.42it/s] 76%|███████▋  | 305455/400000 [00:36<00:11, 8565.65it/s] 77%|███████▋  | 306314/400000 [00:36<00:10, 8571.97it/s] 77%|███████▋  | 307190/400000 [00:36<00:10, 8626.92it/s] 77%|███████▋  | 308053/400000 [00:36<00:10, 8615.15it/s] 77%|███████▋  | 308920/400000 [00:36<00:10, 8628.62it/s] 77%|███████▋  | 309783/400000 [00:36<00:10, 8437.80it/s] 78%|███████▊  | 310628/400000 [00:37<00:10, 8410.28it/s] 78%|███████▊  | 311489/400000 [00:37<00:10, 8467.88it/s] 78%|███████▊  | 312363/400000 [00:37<00:10, 8545.23it/s] 78%|███████▊  | 313230/400000 [00:37<00:10, 8581.03it/s] 79%|███████▊  | 314111/400000 [00:37<00:09, 8645.70it/s] 79%|███████▊  | 314976/400000 [00:37<00:09, 8589.10it/s] 79%|███████▉  | 315859/400000 [00:37<00:09, 8659.53it/s] 79%|███████▉  | 316726/400000 [00:37<00:09, 8649.45it/s] 79%|███████▉  | 317592/400000 [00:37<00:09, 8648.33it/s] 80%|███████▉  | 318469/400000 [00:37<00:09, 8684.25it/s] 80%|███████▉  | 319338/400000 [00:38<00:09, 8632.53it/s] 80%|████████  | 320202/400000 [00:38<00:09, 8470.67it/s] 80%|████████  | 321065/400000 [00:38<00:09, 8515.41it/s] 80%|████████  | 321921/400000 [00:38<00:09, 8526.85it/s] 81%|████████  | 322788/400000 [00:38<00:09, 8567.29it/s] 81%|████████  | 323646/400000 [00:38<00:08, 8534.12it/s] 81%|████████  | 324500/400000 [00:38<00:08, 8525.78it/s] 81%|████████▏ | 325369/400000 [00:38<00:08, 8572.43it/s] 82%|████████▏ | 326232/400000 [00:38<00:08, 8585.25it/s] 82%|████████▏ | 327091/400000 [00:38<00:08, 8414.47it/s] 82%|████████▏ | 327934/400000 [00:39<00:08, 8363.63it/s] 82%|████████▏ | 328772/400000 [00:39<00:08, 8331.33it/s] 82%|████████▏ | 329611/400000 [00:39<00:08, 8346.99it/s] 83%|████████▎ | 330447/400000 [00:39<00:08, 8176.44it/s] 83%|████████▎ | 331310/400000 [00:39<00:08, 8305.15it/s] 83%|████████▎ | 332179/400000 [00:39<00:08, 8416.67it/s] 83%|████████▎ | 333036/400000 [00:39<00:07, 8460.78it/s] 83%|████████▎ | 333891/400000 [00:39<00:07, 8486.49it/s] 84%|████████▎ | 334741/400000 [00:39<00:07, 8484.28it/s] 84%|████████▍ | 335609/400000 [00:39<00:07, 8540.70it/s] 84%|████████▍ | 336464/400000 [00:40<00:07, 8500.85it/s] 84%|████████▍ | 337328/400000 [00:40<00:07, 8540.59it/s] 85%|████████▍ | 338197/400000 [00:40<00:07, 8582.53it/s] 85%|████████▍ | 339056/400000 [00:40<00:07, 8456.46it/s] 85%|████████▍ | 339913/400000 [00:40<00:07, 8487.80it/s] 85%|████████▌ | 340771/400000 [00:40<00:06, 8515.20it/s] 85%|████████▌ | 341640/400000 [00:40<00:06, 8565.98it/s] 86%|████████▌ | 342498/400000 [00:40<00:06, 8567.26it/s] 86%|████████▌ | 343364/400000 [00:40<00:06, 8592.20it/s] 86%|████████▌ | 344239/400000 [00:40<00:06, 8636.36it/s] 86%|████████▋ | 345103/400000 [00:41<00:06, 8587.23it/s] 86%|████████▋ | 345962/400000 [00:41<00:06, 8552.44it/s] 87%|████████▋ | 346819/400000 [00:41<00:06, 8556.19it/s] 87%|████████▋ | 347675/400000 [00:41<00:06, 8542.78it/s] 87%|████████▋ | 348535/400000 [00:41<00:06, 8556.80it/s] 87%|████████▋ | 349396/400000 [00:41<00:05, 8569.88it/s] 88%|████████▊ | 350254/400000 [00:41<00:05, 8569.88it/s] 88%|████████▊ | 351119/400000 [00:41<00:05, 8591.93it/s] 88%|████████▊ | 351980/400000 [00:41<00:05, 8595.72it/s] 88%|████████▊ | 352840/400000 [00:41<00:05, 8531.44it/s] 88%|████████▊ | 353703/400000 [00:42<00:05, 8559.86it/s] 89%|████████▊ | 354560/400000 [00:42<00:05, 8542.99it/s] 89%|████████▉ | 355427/400000 [00:42<00:05, 8579.63it/s] 89%|████████▉ | 356286/400000 [00:42<00:05, 8512.16it/s] 89%|████████▉ | 357151/400000 [00:42<00:05, 8552.90it/s] 90%|████████▉ | 358032/400000 [00:42<00:04, 8625.68it/s] 90%|████████▉ | 358900/400000 [00:42<00:04, 8641.25it/s] 90%|████████▉ | 359780/400000 [00:42<00:04, 8685.69it/s] 90%|█████████ | 360656/400000 [00:42<00:04, 8706.09it/s] 90%|█████████ | 361538/400000 [00:42<00:04, 8739.18it/s] 91%|█████████ | 362431/400000 [00:43<00:04, 8793.99it/s] 91%|█████████ | 363311/400000 [00:43<00:04, 8755.90it/s] 91%|█████████ | 364196/400000 [00:43<00:04, 8782.05it/s] 91%|█████████▏| 365075/400000 [00:43<00:03, 8732.33it/s] 91%|█████████▏| 365951/400000 [00:43<00:03, 8739.12it/s] 92%|█████████▏| 366833/400000 [00:43<00:03, 8760.77it/s] 92%|█████████▏| 367710/400000 [00:43<00:03, 8749.09it/s] 92%|█████████▏| 368585/400000 [00:43<00:03, 8742.70it/s] 92%|█████████▏| 369467/400000 [00:43<00:03, 8764.29it/s] 93%|█████████▎| 370346/400000 [00:43<00:03, 8770.59it/s] 93%|█████████▎| 371224/400000 [00:44<00:03, 8765.54it/s] 93%|█████████▎| 372101/400000 [00:44<00:03, 8719.98it/s] 93%|█████████▎| 372974/400000 [00:44<00:03, 8684.55it/s] 93%|█████████▎| 373843/400000 [00:44<00:03, 8670.90it/s] 94%|█████████▎| 374711/400000 [00:44<00:02, 8465.85it/s] 94%|█████████▍| 375580/400000 [00:44<00:02, 8530.09it/s] 94%|█████████▍| 376434/400000 [00:44<00:02, 8417.53it/s] 94%|█████████▍| 377300/400000 [00:44<00:02, 8487.31it/s] 95%|█████████▍| 378177/400000 [00:44<00:02, 8567.70it/s] 95%|█████████▍| 379056/400000 [00:44<00:02, 8631.32it/s] 95%|█████████▍| 379920/400000 [00:45<00:02, 8521.75it/s] 95%|█████████▌| 380773/400000 [00:45<00:02, 8375.88it/s] 95%|█████████▌| 381645/400000 [00:45<00:02, 8475.60it/s] 96%|█████████▌| 382516/400000 [00:45<00:02, 8544.11it/s] 96%|█████████▌| 383384/400000 [00:45<00:01, 8583.13it/s] 96%|█████████▌| 384243/400000 [00:45<00:01, 8502.35it/s] 96%|█████████▋| 385094/400000 [00:45<00:01, 8451.02it/s] 96%|█████████▋| 385940/400000 [00:45<00:01, 8324.08it/s] 97%|█████████▋| 386782/400000 [00:45<00:01, 8350.83it/s] 97%|█████████▋| 387655/400000 [00:46<00:01, 8458.87it/s] 97%|█████████▋| 388535/400000 [00:46<00:01, 8557.47it/s] 97%|█████████▋| 389398/400000 [00:46<00:01, 8578.69it/s] 98%|█████████▊| 390257/400000 [00:46<00:01, 8435.26it/s] 98%|█████████▊| 391127/400000 [00:46<00:01, 8512.92it/s] 98%|█████████▊| 392012/400000 [00:46<00:00, 8610.39it/s] 98%|█████████▊| 392874/400000 [00:46<00:00, 8567.21it/s] 98%|█████████▊| 393746/400000 [00:46<00:00, 8611.65it/s] 99%|█████████▊| 394608/400000 [00:46<00:00, 8554.42it/s] 99%|█████████▉| 395464/400000 [00:46<00:00, 8555.53it/s] 99%|█████████▉| 396343/400000 [00:47<00:00, 8622.82it/s] 99%|█████████▉| 397214/400000 [00:47<00:00, 8646.02it/s]100%|█████████▉| 398084/400000 [00:47<00:00, 8659.53it/s]100%|█████████▉| 398951/400000 [00:47<00:00, 8661.59it/s]100%|█████████▉| 399824/400000 [00:47<00:00, 8679.19it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8431.31it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f9f6e0b6e80>, <torchtext.data.dataset.TabularDataset object at 0x7f9f6e0b6fd0>, <torchtext.vocab.Vocab object at 0x7f9f6e0b6ef0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 28.33 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 55.06 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.17 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.17 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.59 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.59 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.67 file/s]2020-07-23 00:17:55.852666: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-23 00:17:55.856845: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-23 00:17:55.857002: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b3d5fad030 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-23 00:17:55.857016: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'test', 'fashion-mnist_small.npy', 'mnist2', 'mnist_dataset_small.npy', 'train'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 139074.75it/s] 66%|██████▋   | 6586368/9912422 [00:00<00:16, 198493.52it/s]9920512it [00:00, 37103563.76it/s]                           
0it [00:00, ?it/s]32768it [00:00, 527923.03it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162755.20it/s] 74%|███████▍  | 1220608/1648877 [00:00<00:01, 231099.02it/s]1654784it [00:00, 7621796.63it/s]                            
0it [00:00, ?it/s]8192it [00:00, 173111.74it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
