
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6fe8a3b73a576c75c565441d8ffa2353b386e363', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6fe8a3b73a576c75c565441d8ffa2353b386e363', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6fe8a3b73a576c75c565441d8ffa2353b386e363

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6fe8a3b73a576c75c565441d8ffa2353b386e363

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6fe8a3b73a576c75c565441d8ffa2353b386e363

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f276e688ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f276e688ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f27d92fd488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f27d92fd488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f27f8529ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f27f8529ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f27866a49d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f27866a49d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f27866a49d8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3096576/9912422 [00:00<00:00, 30948818.34it/s]9920512it [00:00, 37639435.51it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1157000.68it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158487.34it/s]1654784it [00:00, 11390223.98it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186583.57it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f27650c86d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f27650d5978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f27866a4620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f27866a4620> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f27866a4620> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:25,  1.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:25,  1.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:25,  1.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:24,  1.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:24,  1.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:23,  1.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:23,  1.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:22,  1.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:57,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:57,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:56,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:56,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:56,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:55,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:55,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:55,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:54,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:38,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:37,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:37,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:37,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:37,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:36,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:36,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:36,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:36,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:25,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:24,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:24,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:24,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:24,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:24,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:23,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:23,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:16,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:16,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:16,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:16,  7.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:15,  7.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:15,  7.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11, 10.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11, 10.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 24.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 24.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 24.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 24.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 24.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 30.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 30.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 30.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 30.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 30.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 30.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 30.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 30.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 30.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 37.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 37.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 37.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 37.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 37.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 37.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 37.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 37.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 37.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 37.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 45.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 45.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 45.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 45.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 45.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 45.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 45.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 45.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 52.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 52.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 52.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 52.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 52.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 52.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 52.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 58.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 58.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 58.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 58.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 58.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 58.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 58.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 58.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 58.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 58.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.24 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.29s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.24 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.29s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.29s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.29s/ url]
0 examples [00:00, ? examples/s]2020-07-26 00:07:53.604986: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-26 00:07:53.616932: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-26 00:07:53.617298: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561a668e8900 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-26 00:07:53.617443: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
47 examples [00:00, 468.07 examples/s]151 examples [00:00, 560.22 examples/s]265 examples [00:00, 660.13 examples/s]380 examples [00:00, 756.00 examples/s]497 examples [00:00, 844.71 examples/s]612 examples [00:00, 917.49 examples/s]714 examples [00:00, 944.51 examples/s]834 examples [00:00, 1008.44 examples/s]944 examples [00:00, 1032.99 examples/s]1055 examples [00:01, 1054.05 examples/s]1163 examples [00:01, 1054.13 examples/s]1270 examples [00:01, 1055.88 examples/s]1387 examples [00:01, 1085.28 examples/s]1497 examples [00:01, 1085.14 examples/s]1611 examples [00:01, 1100.58 examples/s]1726 examples [00:01, 1114.58 examples/s]1838 examples [00:01, 1095.43 examples/s]1957 examples [00:01, 1120.86 examples/s]2071 examples [00:01, 1125.40 examples/s]2184 examples [00:02, 1116.11 examples/s]2296 examples [00:02, 1101.97 examples/s]2407 examples [00:02, 1086.61 examples/s]2520 examples [00:02, 1098.69 examples/s]2632 examples [00:02, 1103.48 examples/s]2743 examples [00:02, 1087.80 examples/s]2855 examples [00:02, 1096.48 examples/s]2965 examples [00:02, 1069.74 examples/s]3073 examples [00:02, 1061.28 examples/s]3190 examples [00:02, 1089.18 examples/s]3301 examples [00:03, 1094.93 examples/s]3411 examples [00:03, 1084.28 examples/s]3520 examples [00:03, 1035.26 examples/s]3630 examples [00:03, 1051.53 examples/s]3747 examples [00:03, 1082.09 examples/s]3864 examples [00:03, 1104.94 examples/s]3981 examples [00:03, 1121.55 examples/s]4094 examples [00:03, 1113.51 examples/s]4206 examples [00:03, 1109.87 examples/s]4322 examples [00:03, 1121.91 examples/s]4435 examples [00:04, 1109.80 examples/s]4547 examples [00:04, 1104.58 examples/s]4658 examples [00:04, 1084.15 examples/s]4774 examples [00:04, 1104.26 examples/s]4889 examples [00:04, 1117.51 examples/s]5001 examples [00:04, 1105.86 examples/s]5112 examples [00:04, 1095.65 examples/s]5222 examples [00:04, 1082.84 examples/s]5332 examples [00:04, 1085.41 examples/s]5441 examples [00:05, 1079.46 examples/s]5550 examples [00:05, 1068.12 examples/s]5657 examples [00:05, 1044.77 examples/s]5767 examples [00:05, 1060.51 examples/s]5884 examples [00:05, 1088.67 examples/s]5997 examples [00:05, 1100.64 examples/s]6108 examples [00:05, 1097.34 examples/s]6218 examples [00:05, 1095.78 examples/s]6328 examples [00:05, 1094.12 examples/s]6442 examples [00:05, 1107.44 examples/s]6553 examples [00:06, 1102.47 examples/s]6666 examples [00:06, 1108.41 examples/s]6777 examples [00:06, 1104.66 examples/s]6888 examples [00:06, 1023.00 examples/s]6992 examples [00:06, 1011.87 examples/s]7095 examples [00:06, 1014.88 examples/s]7204 examples [00:06, 1034.46 examples/s]7312 examples [00:06, 1045.10 examples/s]7417 examples [00:06, 1024.94 examples/s]7531 examples [00:06, 1055.59 examples/s]7643 examples [00:07, 1072.41 examples/s]7754 examples [00:07, 1080.79 examples/s]7863 examples [00:07, 1070.14 examples/s]7971 examples [00:07, 1065.74 examples/s]8081 examples [00:07, 1074.89 examples/s]8192 examples [00:07, 1083.29 examples/s]8301 examples [00:07, 1084.78 examples/s]8414 examples [00:07, 1097.75 examples/s]8526 examples [00:07, 1104.18 examples/s]8637 examples [00:07, 1105.40 examples/s]8748 examples [00:08, 1099.03 examples/s]8858 examples [00:08, 1089.67 examples/s]8968 examples [00:08, 1055.96 examples/s]9074 examples [00:08, 1012.47 examples/s]9181 examples [00:08, 1027.79 examples/s]9288 examples [00:08, 1038.25 examples/s]9400 examples [00:08, 1059.36 examples/s]9507 examples [00:08, 1058.46 examples/s]9614 examples [00:08, 1057.75 examples/s]9722 examples [00:09, 1062.35 examples/s]9829 examples [00:09, 1064.20 examples/s]9937 examples [00:09, 1068.51 examples/s]10044 examples [00:09, 982.25 examples/s]10155 examples [00:09, 1017.25 examples/s]10264 examples [00:09, 1036.93 examples/s]10376 examples [00:09, 1059.52 examples/s]10483 examples [00:09, 1042.07 examples/s]10588 examples [00:09, 1042.34 examples/s]10700 examples [00:09, 1064.34 examples/s]10807 examples [00:10, 1027.50 examples/s]10922 examples [00:10, 1060.22 examples/s]11032 examples [00:10, 1069.16 examples/s]11140 examples [00:10, 1044.71 examples/s]11245 examples [00:10, 1025.58 examples/s]11353 examples [00:10, 1040.41 examples/s]11469 examples [00:10, 1072.48 examples/s]11585 examples [00:10, 1094.92 examples/s]11695 examples [00:10, 1095.20 examples/s]11807 examples [00:10, 1101.22 examples/s]11918 examples [00:11, 1093.58 examples/s]12028 examples [00:11, 1089.63 examples/s]12140 examples [00:11, 1097.71 examples/s]12250 examples [00:11, 1080.04 examples/s]12360 examples [00:11, 1084.42 examples/s]12472 examples [00:11, 1092.25 examples/s]12588 examples [00:11, 1110.98 examples/s]12704 examples [00:11, 1124.90 examples/s]12817 examples [00:11, 1116.08 examples/s]12932 examples [00:12, 1122.83 examples/s]13045 examples [00:12, 1088.68 examples/s]13159 examples [00:12, 1102.76 examples/s]13276 examples [00:12, 1119.93 examples/s]13389 examples [00:12, 1103.88 examples/s]13507 examples [00:12, 1124.64 examples/s]13624 examples [00:12, 1136.19 examples/s]13738 examples [00:12, 1135.12 examples/s]13854 examples [00:12, 1140.37 examples/s]13969 examples [00:12, 1125.85 examples/s]14082 examples [00:13, 1118.48 examples/s]14195 examples [00:13, 1120.83 examples/s]14311 examples [00:13, 1130.50 examples/s]14425 examples [00:13, 1130.75 examples/s]14539 examples [00:13, 1102.83 examples/s]14650 examples [00:13, 1100.07 examples/s]14761 examples [00:13, 1091.50 examples/s]14874 examples [00:13, 1100.57 examples/s]14985 examples [00:13, 1098.33 examples/s]15095 examples [00:13, 1096.82 examples/s]15218 examples [00:14, 1130.73 examples/s]15332 examples [00:14, 1124.78 examples/s]15445 examples [00:14, 1125.40 examples/s]15558 examples [00:14, 1109.99 examples/s]15670 examples [00:14, 1061.10 examples/s]15777 examples [00:14, 1042.67 examples/s]15890 examples [00:14, 1066.10 examples/s]16002 examples [00:14, 1081.36 examples/s]16111 examples [00:14, 1079.45 examples/s]16222 examples [00:14, 1087.06 examples/s]16333 examples [00:15, 1091.98 examples/s]16443 examples [00:15, 1083.67 examples/s]16552 examples [00:15, 1070.76 examples/s]16661 examples [00:15, 1073.38 examples/s]16769 examples [00:15, 1063.49 examples/s]16883 examples [00:15, 1083.37 examples/s]17000 examples [00:15, 1105.98 examples/s]17115 examples [00:15, 1116.11 examples/s]17227 examples [00:15, 1114.43 examples/s]17339 examples [00:15, 1115.84 examples/s]17451 examples [00:16, 1110.70 examples/s]17563 examples [00:16, 1071.48 examples/s]17671 examples [00:16, 1058.02 examples/s]17778 examples [00:16, 1033.08 examples/s]17883 examples [00:16, 1037.65 examples/s]17998 examples [00:16, 1068.66 examples/s]18106 examples [00:16, 1053.56 examples/s]18212 examples [00:16, 1046.10 examples/s]18319 examples [00:16, 1050.73 examples/s]18433 examples [00:17, 1074.12 examples/s]18546 examples [00:17, 1089.38 examples/s]18660 examples [00:17, 1102.42 examples/s]18771 examples [00:17, 1030.99 examples/s]18876 examples [00:17, 1017.80 examples/s]18982 examples [00:17, 1029.36 examples/s]19102 examples [00:17, 1073.61 examples/s]19219 examples [00:17, 1099.35 examples/s]19333 examples [00:17, 1109.48 examples/s]19445 examples [00:17, 1075.54 examples/s]19554 examples [00:18, 1075.67 examples/s]19664 examples [00:18, 1080.28 examples/s]19774 examples [00:18, 1085.51 examples/s]19883 examples [00:18, 1059.30 examples/s]19990 examples [00:18, 1044.00 examples/s]20095 examples [00:18, 977.07 examples/s] 20199 examples [00:18, 994.47 examples/s]20310 examples [00:18, 1025.57 examples/s]20420 examples [00:18, 1045.24 examples/s]20534 examples [00:19, 1071.04 examples/s]20642 examples [00:19, 1067.13 examples/s]20750 examples [00:19, 1060.22 examples/s]20860 examples [00:19, 1071.08 examples/s]20968 examples [00:19, 1053.78 examples/s]21084 examples [00:19, 1081.71 examples/s]21193 examples [00:19, 1051.02 examples/s]21299 examples [00:19, 983.70 examples/s] 21399 examples [00:19, 983.28 examples/s]21504 examples [00:19, 1001.22 examples/s]21605 examples [00:20, 998.06 examples/s] 21706 examples [00:20, 999.53 examples/s]21817 examples [00:20, 1029.46 examples/s]21929 examples [00:20, 1053.16 examples/s]22037 examples [00:20, 1060.82 examples/s]22152 examples [00:20, 1083.78 examples/s]22261 examples [00:20, 1082.62 examples/s]22377 examples [00:20, 1103.58 examples/s]22493 examples [00:20, 1118.34 examples/s]22606 examples [00:20, 1097.62 examples/s]22721 examples [00:21, 1111.66 examples/s]22833 examples [00:21, 1105.00 examples/s]22944 examples [00:21, 1095.61 examples/s]23055 examples [00:21, 1099.82 examples/s]23166 examples [00:21, 1079.20 examples/s]23275 examples [00:21, 1062.24 examples/s]23382 examples [00:21, 1051.79 examples/s]23492 examples [00:21, 1063.85 examples/s]23605 examples [00:21, 1081.21 examples/s]23714 examples [00:22, 1068.68 examples/s]23822 examples [00:22, 1070.96 examples/s]23930 examples [00:22, 1073.48 examples/s]24041 examples [00:22, 1082.35 examples/s]24153 examples [00:22, 1093.21 examples/s]24263 examples [00:22, 1076.77 examples/s]24376 examples [00:22, 1090.32 examples/s]24486 examples [00:22, 1091.14 examples/s]24599 examples [00:22, 1101.38 examples/s]24712 examples [00:22, 1108.32 examples/s]24823 examples [00:23, 1086.63 examples/s]24936 examples [00:23, 1096.75 examples/s]25046 examples [00:23, 1080.86 examples/s]25156 examples [00:23, 1085.20 examples/s]25265 examples [00:23, 1078.62 examples/s]25373 examples [00:23, 1067.59 examples/s]25483 examples [00:23, 1076.24 examples/s]25593 examples [00:23, 1081.07 examples/s]25702 examples [00:23, 1047.85 examples/s]25808 examples [00:23, 1047.03 examples/s]25914 examples [00:24, 1049.12 examples/s]26025 examples [00:24, 1066.06 examples/s]26132 examples [00:24, 1066.60 examples/s]26244 examples [00:24, 1079.46 examples/s]26355 examples [00:24, 1087.19 examples/s]26464 examples [00:24, 1074.19 examples/s]26574 examples [00:24, 1079.49 examples/s]26683 examples [00:24, 1074.16 examples/s]26791 examples [00:24, 1059.76 examples/s]26903 examples [00:24, 1076.24 examples/s]27011 examples [00:25, 1056.38 examples/s]27124 examples [00:25, 1076.96 examples/s]27234 examples [00:25, 1082.91 examples/s]27349 examples [00:25, 1099.57 examples/s]27465 examples [00:25, 1114.15 examples/s]27577 examples [00:25, 1094.06 examples/s]27687 examples [00:25, 1063.82 examples/s]27801 examples [00:25, 1085.14 examples/s]27910 examples [00:25, 1079.66 examples/s]28023 examples [00:26, 1093.10 examples/s]28133 examples [00:26, 1085.52 examples/s]28245 examples [00:26, 1093.39 examples/s]28355 examples [00:26, 1083.17 examples/s]28464 examples [00:26, 1069.26 examples/s]28572 examples [00:26, 1060.34 examples/s]28679 examples [00:26, 1054.98 examples/s]28786 examples [00:26, 1059.42 examples/s]28898 examples [00:26, 1076.30 examples/s]29010 examples [00:26, 1087.90 examples/s]29121 examples [00:27, 1092.04 examples/s]29231 examples [00:27, 1075.63 examples/s]29343 examples [00:27, 1085.95 examples/s]29452 examples [00:27, 1086.49 examples/s]29562 examples [00:27, 1089.44 examples/s]29671 examples [00:27, 1088.09 examples/s]29780 examples [00:27, 1073.87 examples/s]29890 examples [00:27, 1078.85 examples/s]29998 examples [00:27, 1073.22 examples/s]30106 examples [00:27, 1018.46 examples/s]30218 examples [00:28, 1046.64 examples/s]30326 examples [00:28, 1055.06 examples/s]30437 examples [00:28, 1068.20 examples/s]30547 examples [00:28, 1075.81 examples/s]30662 examples [00:28, 1095.33 examples/s]30777 examples [00:28, 1110.99 examples/s]30889 examples [00:28, 1106.07 examples/s]31001 examples [00:28, 1108.93 examples/s]31113 examples [00:28, 1053.53 examples/s]31220 examples [00:29, 1005.51 examples/s]31329 examples [00:29, 1028.44 examples/s]31444 examples [00:29, 1060.55 examples/s]31558 examples [00:29, 1080.96 examples/s]31675 examples [00:29, 1104.85 examples/s]31789 examples [00:29, 1112.34 examples/s]31901 examples [00:29, 1086.74 examples/s]32015 examples [00:29, 1099.62 examples/s]32128 examples [00:29, 1106.40 examples/s]32239 examples [00:29, 1094.11 examples/s]32354 examples [00:30, 1110.25 examples/s]32467 examples [00:30, 1114.68 examples/s]32579 examples [00:30, 1111.19 examples/s]32691 examples [00:30, 1110.34 examples/s]32807 examples [00:30, 1123.64 examples/s]32920 examples [00:30, 1119.00 examples/s]33035 examples [00:30, 1125.57 examples/s]33148 examples [00:30, 1115.50 examples/s]33260 examples [00:30, 1105.74 examples/s]33371 examples [00:30, 1064.28 examples/s]33483 examples [00:31, 1078.45 examples/s]33592 examples [00:31, 1076.16 examples/s]33700 examples [00:31, 1054.46 examples/s]33806 examples [00:31, 1052.97 examples/s]33917 examples [00:31, 1068.14 examples/s]34031 examples [00:31, 1086.54 examples/s]34147 examples [00:31, 1105.48 examples/s]34258 examples [00:31, 1092.03 examples/s]34374 examples [00:31, 1109.88 examples/s]34488 examples [00:31, 1116.37 examples/s]34602 examples [00:32, 1120.60 examples/s]34716 examples [00:32, 1123.74 examples/s]34829 examples [00:32, 1108.83 examples/s]34940 examples [00:32, 1098.21 examples/s]35050 examples [00:32, 1098.11 examples/s]35161 examples [00:32, 1101.09 examples/s]35272 examples [00:32, 1102.47 examples/s]35383 examples [00:32, 1072.99 examples/s]35497 examples [00:32, 1090.18 examples/s]35610 examples [00:32, 1100.51 examples/s]35721 examples [00:33, 1103.20 examples/s]35837 examples [00:33, 1119.26 examples/s]35950 examples [00:33, 1103.16 examples/s]36061 examples [00:33, 1074.89 examples/s]36170 examples [00:33, 1079.21 examples/s]36279 examples [00:33, 1082.01 examples/s]36389 examples [00:33, 1085.15 examples/s]36498 examples [00:33, 1062.84 examples/s]36605 examples [00:33, 1064.85 examples/s]36715 examples [00:34, 1074.11 examples/s]36823 examples [00:34, 1062.01 examples/s]36930 examples [00:34, 1037.88 examples/s]37040 examples [00:34, 1055.38 examples/s]37146 examples [00:34, 1048.38 examples/s]37262 examples [00:34, 1078.01 examples/s]37371 examples [00:34, 1080.88 examples/s]37485 examples [00:34, 1097.46 examples/s]37595 examples [00:34, 1093.50 examples/s]37705 examples [00:34, 1080.53 examples/s]37820 examples [00:35, 1099.39 examples/s]37931 examples [00:35, 1102.15 examples/s]38044 examples [00:35, 1104.87 examples/s]38157 examples [00:35, 1109.55 examples/s]38269 examples [00:35, 1111.31 examples/s]38381 examples [00:35, 1095.74 examples/s]38496 examples [00:35, 1110.89 examples/s]38609 examples [00:35, 1110.77 examples/s]38721 examples [00:35, 1074.45 examples/s]38829 examples [00:35, 1048.79 examples/s]38935 examples [00:36, 1025.85 examples/s]39040 examples [00:36, 1030.16 examples/s]39144 examples [00:36, 1012.29 examples/s]39252 examples [00:36, 1030.13 examples/s]39363 examples [00:36, 1050.19 examples/s]39475 examples [00:36, 1067.35 examples/s]39583 examples [00:36, 1069.08 examples/s]39691 examples [00:36, 1064.88 examples/s]39800 examples [00:36, 1070.33 examples/s]39909 examples [00:36, 1074.53 examples/s]40017 examples [00:37, 1007.44 examples/s]40131 examples [00:37, 1041.80 examples/s]40239 examples [00:37, 1050.50 examples/s]40351 examples [00:37, 1068.19 examples/s]40462 examples [00:37, 1080.10 examples/s]40571 examples [00:37, 1071.21 examples/s]40683 examples [00:37, 1083.04 examples/s]40792 examples [00:37, 1072.15 examples/s]40904 examples [00:37, 1084.28 examples/s]41014 examples [00:38, 1088.89 examples/s]41124 examples [00:38, 1071.91 examples/s]41232 examples [00:38, 1062.08 examples/s]41339 examples [00:38, 1055.26 examples/s]41450 examples [00:38, 1069.76 examples/s]41564 examples [00:38, 1087.74 examples/s]41680 examples [00:38, 1105.91 examples/s]41795 examples [00:38, 1118.55 examples/s]41908 examples [00:38, 1102.77 examples/s]42021 examples [00:38, 1109.38 examples/s]42136 examples [00:39, 1120.55 examples/s]42249 examples [00:39, 1098.95 examples/s]42363 examples [00:39, 1110.26 examples/s]42475 examples [00:39, 1076.28 examples/s]42583 examples [00:39, 1068.35 examples/s]42697 examples [00:39, 1086.36 examples/s]42808 examples [00:39, 1091.56 examples/s]42920 examples [00:39, 1099.34 examples/s]43031 examples [00:39, 1081.60 examples/s]43140 examples [00:39, 1076.87 examples/s]43256 examples [00:40, 1099.09 examples/s]43367 examples [00:40, 1095.65 examples/s]43477 examples [00:40, 1088.23 examples/s]43586 examples [00:40, 1076.93 examples/s]43694 examples [00:40, 1075.21 examples/s]43802 examples [00:40, 1066.99 examples/s]43911 examples [00:40, 1071.60 examples/s]44019 examples [00:40, 1068.91 examples/s]44126 examples [00:40, 1063.82 examples/s]44237 examples [00:40, 1075.35 examples/s]44356 examples [00:41, 1105.59 examples/s]44469 examples [00:41, 1110.30 examples/s]44585 examples [00:41, 1122.83 examples/s]44698 examples [00:41, 1099.79 examples/s]44809 examples [00:41, 1058.73 examples/s]44921 examples [00:41, 1076.23 examples/s]45033 examples [00:41, 1086.44 examples/s]45143 examples [00:41, 1089.15 examples/s]45253 examples [00:41, 1066.22 examples/s]45363 examples [00:42, 1073.21 examples/s]45472 examples [00:42, 1075.71 examples/s]45580 examples [00:42, 1073.73 examples/s]45688 examples [00:42, 1059.26 examples/s]45795 examples [00:42, 1041.98 examples/s]45902 examples [00:42, 1049.34 examples/s]46013 examples [00:42, 1063.81 examples/s]46125 examples [00:42, 1078.05 examples/s]46237 examples [00:42, 1088.82 examples/s]46347 examples [00:42, 1083.93 examples/s]46459 examples [00:43, 1094.11 examples/s]46571 examples [00:43, 1100.86 examples/s]46682 examples [00:43, 1102.95 examples/s]46795 examples [00:43, 1108.70 examples/s]46906 examples [00:43, 1075.19 examples/s]47014 examples [00:43, 1053.49 examples/s]47130 examples [00:43, 1081.64 examples/s]47246 examples [00:43, 1101.61 examples/s]47357 examples [00:43, 1088.00 examples/s]47467 examples [00:43, 1072.35 examples/s]47578 examples [00:44, 1081.04 examples/s]47687 examples [00:44, 1065.04 examples/s]47796 examples [00:44, 1072.24 examples/s]47904 examples [00:44, 1071.88 examples/s]48012 examples [00:44, 1050.56 examples/s]48118 examples [00:44, 1045.34 examples/s]48238 examples [00:44, 1084.90 examples/s]48352 examples [00:44, 1098.26 examples/s]48463 examples [00:44, 1097.69 examples/s]48574 examples [00:45, 1081.31 examples/s]48688 examples [00:45, 1097.56 examples/s]48800 examples [00:45, 1101.56 examples/s]48916 examples [00:45, 1117.11 examples/s]49028 examples [00:45, 1093.87 examples/s]49138 examples [00:45, 1092.22 examples/s]49249 examples [00:45, 1096.03 examples/s]49365 examples [00:45, 1111.79 examples/s]49479 examples [00:45, 1118.87 examples/s]49591 examples [00:45, 1104.66 examples/s]49707 examples [00:46, 1117.66 examples/s]49822 examples [00:46, 1124.81 examples/s]49935 examples [00:46, 1097.70 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 9120/50000 [00:00<00:00, 91198.13 examples/s] 46%|████▌     | 22995/50000 [00:00<00:00, 101649.04 examples/s] 76%|███████▌  | 38046/50000 [00:00<00:00, 112614.80 examples/s]                                                                0 examples [00:00, ? examples/s]90 examples [00:00, 894.22 examples/s]207 examples [00:00, 961.50 examples/s]319 examples [00:00, 1002.59 examples/s]430 examples [00:00, 1030.62 examples/s]545 examples [00:00, 1061.83 examples/s]659 examples [00:00, 1084.11 examples/s]768 examples [00:00, 1085.38 examples/s]876 examples [00:00, 1082.34 examples/s]981 examples [00:00, 1070.98 examples/s]1095 examples [00:01, 1089.04 examples/s]1208 examples [00:01, 1098.79 examples/s]1321 examples [00:01, 1106.26 examples/s]1431 examples [00:01, 1088.37 examples/s]1544 examples [00:01, 1099.64 examples/s]1657 examples [00:01, 1105.79 examples/s]1768 examples [00:01, 1097.07 examples/s]1878 examples [00:01, 1049.85 examples/s]1984 examples [00:01, 1044.47 examples/s]2099 examples [00:01, 1071.45 examples/s]2215 examples [00:02, 1095.14 examples/s]2336 examples [00:02, 1125.36 examples/s]2452 examples [00:02, 1132.86 examples/s]2566 examples [00:02, 1115.70 examples/s]2679 examples [00:02, 1118.87 examples/s]2793 examples [00:02, 1124.64 examples/s]2908 examples [00:02, 1130.96 examples/s]3022 examples [00:02, 1132.01 examples/s]3136 examples [00:02, 1091.57 examples/s]3246 examples [00:02, 1086.98 examples/s]3355 examples [00:03, 1081.95 examples/s]3468 examples [00:03, 1093.72 examples/s]3578 examples [00:03, 1057.13 examples/s]3685 examples [00:03, 1032.65 examples/s]3789 examples [00:03, 1027.51 examples/s]3901 examples [00:03, 1053.53 examples/s]4009 examples [00:03, 1059.39 examples/s]4125 examples [00:03, 1085.93 examples/s]4234 examples [00:03, 1070.04 examples/s]4349 examples [00:03, 1091.33 examples/s]4464 examples [00:04, 1107.59 examples/s]4576 examples [00:04, 1110.57 examples/s]4691 examples [00:04, 1117.65 examples/s]4805 examples [00:04, 1123.53 examples/s]4919 examples [00:04, 1127.54 examples/s]5033 examples [00:04, 1130.27 examples/s]5147 examples [00:04, 1119.01 examples/s]5259 examples [00:04, 1103.04 examples/s]5370 examples [00:04, 1079.09 examples/s]5479 examples [00:05, 1078.46 examples/s]5592 examples [00:05, 1092.05 examples/s]5702 examples [00:05, 1090.04 examples/s]5812 examples [00:05, 1066.17 examples/s]5921 examples [00:05, 1071.63 examples/s]6029 examples [00:05, 1066.00 examples/s]6141 examples [00:05, 1081.46 examples/s]6250 examples [00:05, 1076.37 examples/s]6360 examples [00:05, 1081.00 examples/s]6469 examples [00:05, 1072.01 examples/s]6582 examples [00:06, 1086.61 examples/s]6694 examples [00:06, 1095.77 examples/s]6808 examples [00:06, 1107.26 examples/s]6919 examples [00:06, 1091.82 examples/s]7029 examples [00:06, 1071.41 examples/s]7137 examples [00:06, 1071.27 examples/s]7248 examples [00:06, 1081.08 examples/s]7357 examples [00:06, 1055.73 examples/s]7463 examples [00:06, 1042.30 examples/s]7568 examples [00:06, 1036.49 examples/s]7673 examples [00:07, 1038.46 examples/s]7784 examples [00:07, 1058.17 examples/s]7895 examples [00:07, 1073.08 examples/s]8003 examples [00:07, 1066.45 examples/s]8115 examples [00:07, 1081.25 examples/s]8227 examples [00:07, 1092.30 examples/s]8339 examples [00:07, 1099.07 examples/s]8450 examples [00:07, 1089.49 examples/s]8560 examples [00:07, 1088.53 examples/s]8676 examples [00:07, 1107.94 examples/s]8787 examples [00:08, 1106.22 examples/s]8898 examples [00:08, 1093.18 examples/s]9009 examples [00:08, 1095.79 examples/s]9119 examples [00:08, 1051.79 examples/s]9230 examples [00:08, 1067.45 examples/s]9338 examples [00:08, 1069.26 examples/s]9450 examples [00:08, 1080.74 examples/s]9559 examples [00:08, 1064.15 examples/s]9666 examples [00:08, 1045.39 examples/s]9777 examples [00:09, 1061.55 examples/s]9888 examples [00:09, 1073.08 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSW5GKO/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSW5GKO/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f27866a49d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f27866a49d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f27866a49d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f27f1b34f98>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f270c94d470>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f27866a49d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f27866a49d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f27866a49d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f270c94df28>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f27650d5668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f270809b2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f270809b2f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f270809b2f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f270809b400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f270809b400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f270809b400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=1d57de32d5c71311287bccd5904bf53fce33d2624c7dafcf9360ac809f5af7d5
  Stored in directory: /tmp/pip-ephem-wheel-cache-2qhrogo0/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<17:59:06, 13.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:49:13, 18.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:01:38, 26.5kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:19:41, 37.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:25:08, 54.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.13M/862M [00:01<3:04:29, 77.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:01<2:08:44, 110kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:01<1:29:39, 157kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:02:37, 224kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.1M/862M [00:01<43:38, 319kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.8M/862M [00:01<30:26, 454kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:01<21:15, 646kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.3M/862M [00:02<14:52, 917kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:02<10:24, 1.30MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<07:44, 1.74MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<07:18, 1.84MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<08:43, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:52, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:04<04:59, 2.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<08:44, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<07:43, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<05:48, 2.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<06:46, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<06:10, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:08<04:37, 2.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<06:10, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<06:58, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<05:32, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<06:00, 2.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<05:33, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:12<04:10, 3.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<05:59, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<06:49, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<05:26, 2.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<05:54, 2.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<05:26, 2.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<04:04, 3.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<05:55, 2.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<06:46, 1.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<05:17, 2.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:18<03:51, 3.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<10:39, 1.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<08:47, 1.47MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<06:25, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<07:30, 1.71MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<07:50, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<06:02, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.4M/862M [00:22<04:21, 2.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<14:39, 870kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<11:32, 1.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<08:23, 1.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<08:50, 1.43MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<08:45, 1.45MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:45, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<06:46, 1.86MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:01, 2.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:29, 2.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:04, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:47, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:23, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<05:46, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:20, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:02, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<05:42, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<05:16, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<04:00, 3.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:41, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:54, 3.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:38, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:09, 2.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:54, 3.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:37, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:24, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<05:00, 2.43MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:38, 3.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<11:32, 1.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<09:19, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:41<06:49, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<07:34, 1.59MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<07:43, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<06:00, 2.00MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<06:08, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:30, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<04:07, 2.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:41, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:17, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:00, 2.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<05:25, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:17, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<04:54, 2.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<03:35, 3.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<07:44, 1.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:37, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<04:55, 2.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:11, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:42, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:17, 2.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:34, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:06, 2.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<03:52, 3.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:23, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<04:56, 2.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<03:41, 3.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:19, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<04:41, 2.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:32, 3.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<02:37, 4.37MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<51:22, 223kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<38:16, 299kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<27:20, 418kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<20:54, 545kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<15:46, 721kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<11:18, 1.00MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<10:33, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<09:40, 1.17MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:20, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<05:14, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<1:24:53, 132kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<1:00:33, 186kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<42:34, 263kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<32:24, 345kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<23:41, 472kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<16:47, 664kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<14:19, 775kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<11:10, 994kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<08:05, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<08:14, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:52, 1.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<05:04, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:08, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:30, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:06, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:19, 2.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:56, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:38, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:21, 3.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<11:08, 973kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<08:53, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:28, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<07:02, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<07:06, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:30, 1.95MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<03:57, 2.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<10:23:29, 17.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<7:17:16, 24.4kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<5:05:37, 34.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<3:35:42, 49.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<2:33:05, 69.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<1:47:30, 98.7kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<1:15:04, 141kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<1:01:18, 172kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<43:58, 240kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<30:56, 340kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<24:02, 436kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<18:56, 554kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<13:46, 761kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<11:17, 923kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<08:57, 1.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:30, 1.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:58, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:56, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<04:22, 2.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:28, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:52, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:39, 2.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:58, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:33, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:23, 2.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:42, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:20, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:15, 2.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<03:05, 3.25MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:24:40, 119kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<1:00:15, 167kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<42:17, 238kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<31:51, 314kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<24:18, 412kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<17:25, 573kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<12:16, 811kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<16:25, 605kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<12:31, 794kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<08:59, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<08:35, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<08:00, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<06:05, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:50, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:04, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:44, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<03:25, 2.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<1:12:04, 135kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<51:23, 189kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<36:07, 269kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<27:28, 352kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<21:12, 456kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<15:16, 632kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<10:43, 895kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<37:18, 257kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<27:03, 354kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<19:08, 500kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<15:35, 612kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<11:51, 803kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<08:31, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:11, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:38, 1.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:49, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:34, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:51, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:37, 2.58MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:43, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:14, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:08, 2.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<02:59, 3.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<1:08:40, 135kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<48:58, 189kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<34:25, 268kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<26:08, 351kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<20:05, 457kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<14:30, 632kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<10:11, 894kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<8:55:44, 17.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<6:15:42, 24.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<4:22:28, 34.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<3:05:07, 48.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<2:11:23, 68.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<1:32:17, 97.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<1:05:41, 137kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<47:48, 188kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<33:53, 265kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<23:41, 376kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<48:47, 183kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<35:03, 254kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<24:39, 360kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<19:16, 459kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<15:15, 579kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<11:03, 798kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<07:49, 1.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<10:46, 815kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<08:26, 1.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:06, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<06:17, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:17, 1.65MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:54, 2.22MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:46, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:05, 1.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:59, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:10, 2.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:47, 3.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:55, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:38, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<02:43, 3.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:52, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:25, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:31, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:48, 2.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:32, 2.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:40, 3.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:48, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:21, 1.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:23, 2.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<02:28, 3.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:14, 1.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:12, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:48, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:34, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:51, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:44, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<02:42, 2.98MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:10, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:08, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:47, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:31, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:47, 1.67MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:45, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:54, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:32, 2.25MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:40, 2.96MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:42, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:24, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:34, 3.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:37, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:07, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:16, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<02:21, 3.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<7:28:59, 17.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<5:14:49, 24.6kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<3:39:51, 35.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<2:34:57, 49.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<1:49:58, 69.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<1:17:14, 99.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<54:56, 139kB/s]   .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<39:58, 190kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<28:16, 269kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<19:45, 382kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<20:41, 365kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<15:14, 494kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<10:49, 693kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<09:17, 804kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<08:01, 932kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:58, 1.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<05:21, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<04:29, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:19, 2.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:02, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:18, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:23, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:32, 2.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:00, 1.81MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:09, 2.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<02:17, 3.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<25:56, 277kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<18:47, 383kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<13:17, 539kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<10:54, 653kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:22, 851kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<06:00, 1.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:49, 1.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<05:30, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:12, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<04:03, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:32, 1.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:39, 2.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:28, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:49, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:01, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:13, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:57, 2.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:12, 3.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:07, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:33, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:50, 2.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:03, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:50, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:07, 3.15MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:02, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:30, 1.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:47, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:00, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:46, 2.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:06, 3.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:59, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:44, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:04, 3.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:58, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:20, 1.93MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:36, 2.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<01:54, 3.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:58, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:26, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:33, 2.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:15, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:36, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:48, 2.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<02:00, 3.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<09:36, 649kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<07:22, 846kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<05:17, 1.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:08, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:50, 1.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:38, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:37, 2.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:43, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:55, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:53, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:25, 1.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:36, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:46, 2.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:00, 2.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:39, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:52, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:49, 2.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:21, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:32, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:43, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<01:58, 2.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:27, 1.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:42, 1.57MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:44, 2.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:15, 1.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:26, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:38, 2.17MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<01:55, 2.97MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:58, 1.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:22, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:29, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:02, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:35, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:57, 2.86MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:40, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:26, 2.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<01:50, 3.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:35, 2.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:21, 2.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<01:46, 3.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:31, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:51, 1.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:13, 2.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:37, 3.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:53, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:16, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:24, 2.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:54, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<03:08, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:25, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:45, 2.97MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:06, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:24, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:28, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:56, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:34, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:53, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:32, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:48, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:12, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:20, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:41, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:08, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:32, 3.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<36:24, 136kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<25:57, 190kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<18:11, 270kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<13:47, 353kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<10:37, 458kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<07:38, 636kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<05:22, 897kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<06:19, 758kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<04:54, 976kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<03:31, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:33, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:26, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:38, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:35, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:16, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:41, 2.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:15, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:30, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:58, 2.31MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:06, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:56, 2.32MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:28, 3.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:04, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:21, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:52, 2.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:00, 2.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:50, 2.37MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:22, 3.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:58, 2.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:13, 1.94MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:46, 2.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:54, 2.22MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:14, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:47, 2.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<01:17, 3.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<30:47, 136kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<21:57, 190kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<15:21, 270kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<11:37, 354kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<08:57, 459kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<06:25, 637kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<04:30, 900kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<06:12, 651kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<04:41, 861kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<03:21, 1.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:16, 1.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:41, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:57, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:16, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:59, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:28, 2.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:56, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:07, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:40, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:46, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:36, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:13, 3.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:42, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:56, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:32, 2.39MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:39, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:28, 2.46MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:07, 3.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:36, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:50, 1.93MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:28, 2.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:03, 3.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<3:23:38, 17.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<2:22:38, 24.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<1:39:11, 34.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<1:09:30, 49.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<49:18, 69.5kB/s]  .vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<34:33, 98.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<24:21, 138kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<17:21, 193kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<12:08, 274kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<09:11, 358kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<06:44, 487kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<04:46, 684kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<04:03, 793kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<03:09, 1.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<02:16, 1.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:19, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:56, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:25, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:43, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:30, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:00<01:07, 2.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:29, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:20, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:00, 2.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:23, 2.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:16, 2.31MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<00:56, 3.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:20, 2.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:30, 1.90MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:12, 2.38MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:17, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:29, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:10, 2.37MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:08<00:50, 3.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<18:11, 151kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<12:59, 211kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<09:04, 299kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<06:53, 388kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<05:21, 498kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<03:51, 690kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<02:41, 974kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:24, 766kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:38, 985kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:52, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:53, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:50, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:23, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<00:59, 2.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<18:31, 133kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<13:11, 187kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<09:11, 265kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<06:54, 348kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<05:03, 473kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<03:34, 664kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<03:00, 775kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<02:34, 905kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:53, 1.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<01:20, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:59, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:37, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:10, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:19, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:22, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:03, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:04, 1.98MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:58, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:43, 2.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:58, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:05, 1.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:51, 2.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:36, 3.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:44, 1.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:25, 1.39MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:02, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:09, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:12, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:55, 2.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:56, 1.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:01, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:48, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:34, 3.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<13:09, 135kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<09:21, 190kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<06:30, 269kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<04:51, 353kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<03:44, 457kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:40, 636kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<01:50, 898kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<02:22, 692kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:49, 899kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:17, 1.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:15, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:11, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:54, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<00:51, 1.76MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:45, 1.99MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:33, 2.65MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:42, 2.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:47, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:37, 2.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:38, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:43, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:33, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:24, 3.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:50, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:41, 1.86MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:30, 2.49MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:38, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:34, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:25, 2.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:33, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:30, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:22, 3.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:30, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:27, 2.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:20, 3.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:28, 2.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:32, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:25, 2.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:26, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:30, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:23, 2.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:01<00:16, 3.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<06:34, 136kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<04:38, 191kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<03:10, 270kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<02:19, 355kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:47, 459kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<01:16, 635kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<00:50, 899kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<44:22, 17.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<30:52, 24.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<20:53, 34.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<14:02, 48.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<09:55, 68.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<06:50, 97.8kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<04:31, 137kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<03:11, 191kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<02:09, 272kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<01:32, 356kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<01:07, 484kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:45, 681kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:36, 789kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:28, 1.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:19, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:18, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:17, 1.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:06, 2.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 2.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:04, 3.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:10, 1.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:05, 1.94MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.97MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 3.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 134kB/s] .vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<33:57:38,  3.27it/s]  0%|          | 891/400000 [00:00<23:43:23,  4.67it/s]  0%|          | 1873/400000 [00:00<16:34:07,  6.67it/s]  1%|          | 2857/400000 [00:00<11:34:22,  9.53it/s]  1%|          | 3816/400000 [00:00<8:05:05, 13.61it/s]   1%|          | 4823/400000 [00:00<5:38:53, 19.43it/s]  1%|▏         | 5838/400000 [00:00<3:56:48, 27.74it/s]  2%|▏         | 6820/400000 [00:01<2:45:33, 39.58it/s]  2%|▏         | 7758/400000 [00:01<1:55:49, 56.44it/s]  2%|▏         | 8688/400000 [00:01<1:21:05, 80.42it/s]  2%|▏         | 9668/400000 [00:01<56:49, 114.49it/s]   3%|▎         | 10655/400000 [00:01<39:52, 162.74it/s]  3%|▎         | 11643/400000 [00:01<28:02, 230.86it/s]  3%|▎         | 12632/400000 [00:01<19:46, 326.53it/s]  3%|▎         | 13603/400000 [00:01<14:00, 459.73it/s]  4%|▎         | 14569/400000 [00:01<09:59, 643.37it/s]  4%|▍         | 15567/400000 [00:01<07:09, 894.36it/s]  4%|▍         | 16539/400000 [00:02<05:12, 1226.72it/s]  4%|▍         | 17496/400000 [00:02<03:51, 1655.32it/s]  5%|▍         | 18435/400000 [00:02<02:53, 2198.61it/s]  5%|▍         | 19417/400000 [00:02<02:12, 2865.87it/s]  5%|▌         | 20383/400000 [00:02<01:44, 3631.99it/s]  5%|▌         | 21357/400000 [00:02<01:24, 4473.43it/s]  6%|▌         | 22319/400000 [00:02<01:10, 5321.36it/s]  6%|▌         | 23283/400000 [00:02<01:01, 6147.25it/s]  6%|▌         | 24244/400000 [00:02<00:54, 6837.29it/s]  6%|▋         | 25195/400000 [00:02<00:50, 7356.63it/s]  7%|▋         | 26129/400000 [00:03<00:47, 7831.00it/s]  7%|▋         | 27059/400000 [00:03<00:45, 8199.16it/s]  7%|▋         | 28027/400000 [00:03<00:43, 8593.08it/s]  7%|▋         | 29030/400000 [00:03<00:41, 8978.20it/s]  7%|▋         | 29989/400000 [00:03<00:40, 9090.67it/s]  8%|▊         | 30990/400000 [00:03<00:39, 9346.59it/s]  8%|▊         | 31957/400000 [00:03<00:39, 9422.58it/s]  8%|▊         | 32922/400000 [00:03<00:38, 9447.64it/s]  8%|▊         | 33911/400000 [00:03<00:38, 9573.56it/s]  9%|▊         | 34926/400000 [00:03<00:37, 9738.59it/s]  9%|▉         | 35909/400000 [00:04<00:37, 9706.71it/s]  9%|▉         | 36886/400000 [00:04<00:37, 9642.48it/s]  9%|▉         | 37869/400000 [00:04<00:37, 9696.71it/s] 10%|▉         | 38883/400000 [00:04<00:36, 9824.29it/s] 10%|▉         | 39869/400000 [00:04<00:37, 9653.26it/s] 10%|█         | 40837/400000 [00:04<00:37, 9659.14it/s] 10%|█         | 41805/400000 [00:04<00:37, 9481.02it/s] 11%|█         | 42817/400000 [00:04<00:36, 9662.34it/s] 11%|█         | 43839/400000 [00:04<00:36, 9822.88it/s] 11%|█         | 44840/400000 [00:04<00:35, 9876.74it/s] 11%|█▏        | 45830/400000 [00:05<00:36, 9660.54it/s] 12%|█▏        | 46799/400000 [00:05<00:36, 9627.17it/s] 12%|█▏        | 47807/400000 [00:05<00:36, 9758.64it/s] 12%|█▏        | 48815/400000 [00:05<00:35, 9849.81it/s] 12%|█▏        | 49802/400000 [00:05<00:35, 9748.04it/s] 13%|█▎        | 50806/400000 [00:05<00:35, 9833.31it/s] 13%|█▎        | 51791/400000 [00:05<00:35, 9766.46it/s] 13%|█▎        | 52785/400000 [00:05<00:35, 9816.52it/s] 13%|█▎        | 53768/400000 [00:05<00:35, 9820.50it/s] 14%|█▎        | 54751/400000 [00:05<00:35, 9812.12it/s] 14%|█▍        | 55733/400000 [00:06<00:35, 9754.95it/s] 14%|█▍        | 56709/400000 [00:06<00:35, 9726.73it/s] 14%|█▍        | 57682/400000 [00:06<00:36, 9283.53it/s] 15%|█▍        | 58615/400000 [00:06<00:37, 9071.42it/s] 15%|█▍        | 59527/400000 [00:06<00:38, 8887.85it/s] 15%|█▌        | 60511/400000 [00:06<00:37, 9151.31it/s] 15%|█▌        | 61431/400000 [00:06<00:36, 9155.20it/s] 16%|█▌        | 62426/400000 [00:06<00:35, 9379.83it/s] 16%|█▌        | 63427/400000 [00:06<00:35, 9560.22it/s] 16%|█▌        | 64394/400000 [00:07<00:34, 9592.00it/s] 16%|█▋        | 65356/400000 [00:07<00:35, 9343.19it/s] 17%|█▋        | 66294/400000 [00:07<00:35, 9316.31it/s] 17%|█▋        | 67285/400000 [00:07<00:35, 9486.45it/s] 17%|█▋        | 68290/400000 [00:07<00:34, 9646.57it/s] 17%|█▋        | 69257/400000 [00:07<00:34, 9605.29it/s] 18%|█▊        | 70220/400000 [00:07<00:34, 9550.66it/s] 18%|█▊        | 71177/400000 [00:07<00:34, 9400.59it/s] 18%|█▊        | 72196/400000 [00:07<00:34, 9621.94it/s] 18%|█▊        | 73161/400000 [00:07<00:34, 9364.16it/s] 19%|█▊        | 74101/400000 [00:08<00:35, 9233.07it/s] 19%|█▉        | 75028/400000 [00:08<00:35, 9242.12it/s] 19%|█▉        | 75968/400000 [00:08<00:34, 9286.48it/s] 19%|█▉        | 76988/400000 [00:08<00:33, 9540.30it/s] 19%|█▉        | 77989/400000 [00:08<00:33, 9673.89it/s] 20%|█▉        | 78960/400000 [00:08<00:33, 9683.42it/s] 20%|█▉        | 79982/400000 [00:08<00:32, 9836.50it/s] 20%|██        | 80968/400000 [00:08<00:32, 9699.62it/s] 20%|██        | 81986/400000 [00:08<00:32, 9838.83it/s] 21%|██        | 82972/400000 [00:08<00:32, 9793.84it/s] 21%|██        | 83958/400000 [00:09<00:32, 9812.39it/s] 21%|██        | 84941/400000 [00:09<00:32, 9631.76it/s] 21%|██▏       | 85906/400000 [00:09<00:33, 9449.82it/s] 22%|██▏       | 86923/400000 [00:09<00:32, 9654.83it/s] 22%|██▏       | 87922/400000 [00:09<00:31, 9752.96it/s] 22%|██▏       | 88905/400000 [00:09<00:31, 9774.73it/s] 22%|██▏       | 89913/400000 [00:09<00:31, 9863.67it/s] 23%|██▎       | 90901/400000 [00:09<00:31, 9736.21it/s] 23%|██▎       | 91914/400000 [00:09<00:31, 9850.87it/s] 23%|██▎       | 92929/400000 [00:09<00:30, 9938.29it/s] 23%|██▎       | 93933/400000 [00:10<00:30, 9966.88it/s] 24%|██▎       | 94936/400000 [00:10<00:30, 9984.23it/s] 24%|██▍       | 95935/400000 [00:10<00:31, 9632.99it/s] 24%|██▍       | 96902/400000 [00:10<00:31, 9622.99it/s] 24%|██▍       | 97867/400000 [00:10<00:31, 9556.99it/s] 25%|██▍       | 98840/400000 [00:10<00:31, 9606.22it/s] 25%|██▍       | 99846/400000 [00:10<00:30, 9737.38it/s] 25%|██▌       | 100821/400000 [00:10<00:32, 9306.48it/s] 25%|██▌       | 101822/400000 [00:10<00:31, 9504.85it/s] 26%|██▌       | 102792/400000 [00:11<00:31, 9561.18it/s] 26%|██▌       | 103800/400000 [00:11<00:30, 9710.56it/s] 26%|██▌       | 104794/400000 [00:11<00:30, 9776.07it/s] 26%|██▋       | 105774/400000 [00:11<00:30, 9617.01it/s] 27%|██▋       | 106770/400000 [00:11<00:30, 9716.73it/s] 27%|██▋       | 107744/400000 [00:11<00:30, 9546.53it/s] 27%|██▋       | 108759/400000 [00:11<00:29, 9718.03it/s] 27%|██▋       | 109742/400000 [00:11<00:29, 9750.18it/s] 28%|██▊       | 110719/400000 [00:11<00:30, 9635.17it/s] 28%|██▊       | 111719/400000 [00:11<00:29, 9739.69it/s] 28%|██▊       | 112700/400000 [00:12<00:29, 9759.78it/s] 28%|██▊       | 113677/400000 [00:12<00:30, 9485.87it/s] 29%|██▊       | 114677/400000 [00:12<00:29, 9632.86it/s] 29%|██▉       | 115643/400000 [00:12<00:30, 9340.11it/s] 29%|██▉       | 116619/400000 [00:12<00:29, 9460.53it/s] 29%|██▉       | 117568/400000 [00:12<00:29, 9462.42it/s] 30%|██▉       | 118571/400000 [00:12<00:29, 9623.38it/s] 30%|██▉       | 119567/400000 [00:12<00:28, 9721.37it/s] 30%|███       | 120541/400000 [00:12<00:28, 9652.78it/s] 30%|███       | 121540/400000 [00:12<00:28, 9751.15it/s] 31%|███       | 122524/400000 [00:13<00:28, 9776.38it/s] 31%|███       | 123527/400000 [00:13<00:28, 9850.12it/s] 31%|███       | 124513/400000 [00:13<00:28, 9675.25it/s] 31%|███▏      | 125482/400000 [00:13<00:29, 9458.79it/s] 32%|███▏      | 126442/400000 [00:13<00:28, 9498.18it/s] 32%|███▏      | 127424/400000 [00:13<00:28, 9588.19it/s] 32%|███▏      | 128433/400000 [00:13<00:27, 9731.63it/s] 32%|███▏      | 129449/400000 [00:13<00:27, 9855.04it/s] 33%|███▎      | 130436/400000 [00:13<00:28, 9559.35it/s] 33%|███▎      | 131395/400000 [00:13<00:28, 9550.38it/s] 33%|███▎      | 132358/400000 [00:14<00:27, 9573.02it/s] 33%|███▎      | 133371/400000 [00:14<00:27, 9732.73it/s] 34%|███▎      | 134350/400000 [00:14<00:27, 9749.51it/s] 34%|███▍      | 135327/400000 [00:14<00:27, 9650.62it/s] 34%|███▍      | 136334/400000 [00:14<00:26, 9771.82it/s] 34%|███▍      | 137313/400000 [00:14<00:26, 9769.49it/s] 35%|███▍      | 138303/400000 [00:14<00:26, 9805.94it/s] 35%|███▍      | 139315/400000 [00:14<00:26, 9895.45it/s] 35%|███▌      | 140306/400000 [00:14<00:26, 9772.85it/s] 35%|███▌      | 141299/400000 [00:14<00:26, 9817.11it/s] 36%|███▌      | 142282/400000 [00:15<00:26, 9778.39it/s] 36%|███▌      | 143261/400000 [00:15<00:26, 9740.08it/s] 36%|███▌      | 144244/400000 [00:15<00:26, 9761.54it/s] 36%|███▋      | 145221/400000 [00:15<00:26, 9613.86it/s] 37%|███▋      | 146184/400000 [00:15<00:26, 9584.21it/s] 37%|███▋      | 147143/400000 [00:15<00:26, 9557.07it/s] 37%|███▋      | 148152/400000 [00:15<00:25, 9708.65it/s] 37%|███▋      | 149133/400000 [00:15<00:25, 9738.49it/s] 38%|███▊      | 150108/400000 [00:15<00:25, 9657.67it/s] 38%|███▊      | 151082/400000 [00:15<00:25, 9679.94it/s] 38%|███▊      | 152089/400000 [00:16<00:25, 9792.02it/s] 38%|███▊      | 153099/400000 [00:16<00:24, 9879.96it/s] 39%|███▊      | 154115/400000 [00:16<00:24, 9959.82it/s] 39%|███▉      | 155112/400000 [00:16<00:25, 9529.06it/s] 39%|███▉      | 156094/400000 [00:16<00:25, 9614.15it/s] 39%|███▉      | 157059/400000 [00:16<00:25, 9565.96it/s] 40%|███▉      | 158018/400000 [00:16<00:25, 9471.72it/s] 40%|███▉      | 158997/400000 [00:16<00:25, 9557.64it/s] 40%|███▉      | 159956/400000 [00:16<00:25, 9566.87it/s] 40%|████      | 160931/400000 [00:17<00:24, 9620.06it/s] 40%|████      | 161894/400000 [00:17<00:25, 9334.22it/s] 41%|████      | 162904/400000 [00:17<00:24, 9551.25it/s] 41%|████      | 163883/400000 [00:17<00:24, 9620.44it/s] 41%|████      | 164848/400000 [00:17<00:24, 9416.50it/s] 41%|████▏     | 165843/400000 [00:17<00:24, 9569.09it/s] 42%|████▏     | 166803/400000 [00:17<00:24, 9431.18it/s] 42%|████▏     | 167814/400000 [00:17<00:24, 9624.85it/s] 42%|████▏     | 168786/400000 [00:17<00:23, 9650.06it/s] 42%|████▏     | 169760/400000 [00:17<00:23, 9674.60it/s] 43%|████▎     | 170766/400000 [00:18<00:23, 9786.72it/s] 43%|████▎     | 171766/400000 [00:18<00:23, 9848.16it/s] 43%|████▎     | 172775/400000 [00:18<00:22, 9918.47it/s] 43%|████▎     | 173768/400000 [00:18<00:22, 9852.14it/s] 44%|████▎     | 174754/400000 [00:18<00:23, 9403.12it/s] 44%|████▍     | 175728/400000 [00:18<00:23, 9501.11it/s] 44%|████▍     | 176682/400000 [00:18<00:23, 9455.00it/s] 44%|████▍     | 177631/400000 [00:18<00:23, 9434.61it/s] 45%|████▍     | 178587/400000 [00:18<00:23, 9470.44it/s] 45%|████▍     | 179583/400000 [00:18<00:22, 9609.74it/s] 45%|████▌     | 180598/400000 [00:19<00:22, 9765.33it/s] 45%|████▌     | 181613/400000 [00:19<00:22, 9877.38it/s] 46%|████▌     | 182603/400000 [00:19<00:22, 9838.90it/s] 46%|████▌     | 183607/400000 [00:19<00:21, 9895.34it/s] 46%|████▌     | 184598/400000 [00:19<00:22, 9789.89it/s] 46%|████▋     | 185578/400000 [00:19<00:22, 9686.22it/s] 47%|████▋     | 186548/400000 [00:19<00:22, 9661.82it/s] 47%|████▋     | 187530/400000 [00:19<00:21, 9708.02it/s] 47%|████▋     | 188551/400000 [00:19<00:21, 9850.73it/s] 47%|████▋     | 189537/400000 [00:19<00:21, 9751.43it/s] 48%|████▊     | 190513/400000 [00:20<00:21, 9689.66it/s] 48%|████▊     | 191483/400000 [00:20<00:22, 9406.29it/s] 48%|████▊     | 192442/400000 [00:20<00:21, 9458.63it/s] 48%|████▊     | 193460/400000 [00:20<00:21, 9663.82it/s] 49%|████▊     | 194429/400000 [00:20<00:21, 9586.91it/s] 49%|████▉     | 195423/400000 [00:20<00:21, 9687.56it/s] 49%|████▉     | 196404/400000 [00:20<00:20, 9722.19it/s] 49%|████▉     | 197396/400000 [00:20<00:20, 9779.89it/s] 50%|████▉     | 198406/400000 [00:20<00:20, 9872.62it/s] 50%|████▉     | 199395/400000 [00:20<00:20, 9726.08it/s] 50%|█████     | 200393/400000 [00:21<00:20, 9799.00it/s] 50%|█████     | 201409/400000 [00:21<00:20, 9902.40it/s] 51%|█████     | 202421/400000 [00:21<00:19, 9964.33it/s] 51%|█████     | 203443/400000 [00:21<00:19, 10038.48it/s] 51%|█████     | 204448/400000 [00:21<00:20, 9770.90it/s]  51%|█████▏    | 205440/400000 [00:21<00:19, 9814.20it/s] 52%|█████▏    | 206423/400000 [00:21<00:19, 9685.73it/s] 52%|█████▏    | 207419/400000 [00:21<00:19, 9764.11it/s] 52%|█████▏    | 208397/400000 [00:21<00:20, 9533.65it/s] 52%|█████▏    | 209353/400000 [00:22<00:20, 9453.75it/s] 53%|█████▎    | 210300/400000 [00:22<00:20, 9251.81it/s] 53%|█████▎    | 211289/400000 [00:22<00:20, 9432.74it/s] 53%|█████▎    | 212298/400000 [00:22<00:19, 9620.20it/s] 53%|█████▎    | 213290/400000 [00:22<00:19, 9707.95it/s] 54%|█████▎    | 214263/400000 [00:22<00:19, 9630.61it/s] 54%|█████▍    | 215249/400000 [00:22<00:19, 9697.13it/s] 54%|█████▍    | 216254/400000 [00:22<00:18, 9799.30it/s] 54%|█████▍    | 217271/400000 [00:22<00:18, 9906.28it/s] 55%|█████▍    | 218289/400000 [00:22<00:18, 9984.83it/s] 55%|█████▍    | 219289/400000 [00:23<00:18, 9716.29it/s] 55%|█████▌    | 220293/400000 [00:23<00:18, 9808.80it/s] 55%|█████▌    | 221303/400000 [00:23<00:18, 9893.58it/s] 56%|█████▌    | 222323/400000 [00:23<00:17, 9982.58it/s] 56%|█████▌    | 223323/400000 [00:23<00:17, 9905.61it/s] 56%|█████▌    | 224315/400000 [00:23<00:18, 9369.93it/s] 56%|█████▋    | 225259/400000 [00:23<00:18, 9385.65it/s] 57%|█████▋    | 226253/400000 [00:23<00:18, 9545.30it/s] 57%|█████▋    | 227246/400000 [00:23<00:17, 9657.53it/s] 57%|█████▋    | 228233/400000 [00:23<00:17, 9715.83it/s] 57%|█████▋    | 229207/400000 [00:24<00:17, 9638.86it/s] 58%|█████▊    | 230209/400000 [00:24<00:17, 9747.58it/s] 58%|█████▊    | 231211/400000 [00:24<00:17, 9827.24it/s] 58%|█████▊    | 232231/400000 [00:24<00:16, 9934.82it/s] 58%|█████▊    | 233241/400000 [00:24<00:16, 9982.68it/s] 59%|█████▊    | 234241/400000 [00:24<00:17, 9690.56it/s] 59%|█████▉    | 235228/400000 [00:24<00:16, 9741.76it/s] 59%|█████▉    | 236215/400000 [00:24<00:16, 9777.53it/s] 59%|█████▉    | 237228/400000 [00:24<00:16, 9878.04it/s] 60%|█████▉    | 238222/400000 [00:24<00:16, 9895.09it/s] 60%|█████▉    | 239213/400000 [00:25<00:16, 9690.35it/s] 60%|██████    | 240224/400000 [00:25<00:16, 9809.98it/s] 60%|██████    | 241241/400000 [00:25<00:16, 9913.36it/s] 61%|██████    | 242247/400000 [00:25<00:15, 9954.24it/s] 61%|██████    | 243259/400000 [00:25<00:15, 10002.63it/s] 61%|██████    | 244260/400000 [00:25<00:16, 9689.02it/s]  61%|██████▏   | 245237/400000 [00:25<00:15, 9711.81it/s] 62%|██████▏   | 246247/400000 [00:25<00:15, 9821.14it/s] 62%|██████▏   | 247260/400000 [00:25<00:15, 9909.03it/s] 62%|██████▏   | 248253/400000 [00:25<00:15, 9792.19it/s] 62%|██████▏   | 249234/400000 [00:26<00:15, 9771.05it/s] 63%|██████▎   | 250247/400000 [00:26<00:15, 9873.32it/s] 63%|██████▎   | 251258/400000 [00:26<00:14, 9941.22it/s] 63%|██████▎   | 252276/400000 [00:26<00:14, 10008.84it/s] 63%|██████▎   | 253278/400000 [00:26<00:14, 9867.33it/s]  64%|██████▎   | 254266/400000 [00:26<00:15, 9595.69it/s] 64%|██████▍   | 255236/400000 [00:26<00:15, 9625.28it/s] 64%|██████▍   | 256249/400000 [00:26<00:14, 9770.49it/s] 64%|██████▍   | 257235/400000 [00:26<00:14, 9795.43it/s] 65%|██████▍   | 258216/400000 [00:27<00:14, 9585.55it/s] 65%|██████▍   | 259177/400000 [00:27<00:14, 9492.90it/s] 65%|██████▌   | 260128/400000 [00:27<00:14, 9495.87it/s] 65%|██████▌   | 261139/400000 [00:27<00:14, 9670.63it/s] 66%|██████▌   | 262153/400000 [00:27<00:14, 9804.46it/s] 66%|██████▌   | 263135/400000 [00:27<00:13, 9779.23it/s] 66%|██████▌   | 264114/400000 [00:27<00:14, 9526.32it/s] 66%|██████▋   | 265069/400000 [00:27<00:14, 9516.01it/s] 67%|██████▋   | 266083/400000 [00:27<00:13, 9693.93it/s] 67%|██████▋   | 267091/400000 [00:27<00:13, 9806.49it/s] 67%|██████▋   | 268079/400000 [00:28<00:13, 9825.52it/s] 67%|██████▋   | 269063/400000 [00:28<00:13, 9754.30it/s] 68%|██████▊   | 270040/400000 [00:28<00:13, 9722.01it/s] 68%|██████▊   | 271056/400000 [00:28<00:13, 9847.48it/s] 68%|██████▊   | 272055/400000 [00:28<00:12, 9887.67it/s] 68%|██████▊   | 273055/400000 [00:28<00:12, 9920.71it/s] 69%|██████▊   | 274048/400000 [00:28<00:12, 9724.86it/s] 69%|██████▉   | 275022/400000 [00:28<00:13, 9375.04it/s] 69%|██████▉   | 275974/400000 [00:28<00:13, 9415.58it/s] 69%|██████▉   | 276990/400000 [00:28<00:12, 9625.26it/s] 70%|██████▉   | 278006/400000 [00:29<00:12, 9776.66it/s] 70%|██████▉   | 278987/400000 [00:29<00:12, 9687.01it/s] 70%|███████   | 280000/400000 [00:29<00:12, 9814.01it/s] 70%|███████   | 281006/400000 [00:29<00:12, 9884.55it/s] 71%|███████   | 282009/400000 [00:29<00:11, 9927.65it/s] 71%|███████   | 283013/400000 [00:29<00:11, 9960.58it/s] 71%|███████   | 284010/400000 [00:29<00:11, 9866.95it/s] 71%|███████   | 284998/400000 [00:29<00:12, 9489.82it/s] 71%|███████▏  | 285951/400000 [00:29<00:12, 9470.70it/s] 72%|███████▏  | 286961/400000 [00:29<00:11, 9648.69it/s] 72%|███████▏  | 287947/400000 [00:30<00:11, 9706.36it/s] 72%|███████▏  | 288920/400000 [00:30<00:11, 9661.03it/s] 72%|███████▏  | 289932/400000 [00:30<00:11, 9793.10it/s] 73%|███████▎  | 290953/400000 [00:30<00:11, 9912.68it/s] 73%|███████▎  | 291951/400000 [00:30<00:10, 9930.06it/s] 73%|███████▎  | 292945/400000 [00:30<00:11, 9534.51it/s] 73%|███████▎  | 293920/400000 [00:30<00:11, 9596.70it/s] 74%|███████▎  | 294888/400000 [00:30<00:10, 9620.91it/s] 74%|███████▍  | 295853/400000 [00:30<00:10, 9570.04it/s] 74%|███████▍  | 296812/400000 [00:30<00:10, 9519.79it/s] 74%|███████▍  | 297792/400000 [00:31<00:10, 9592.22it/s] 75%|███████▍  | 298765/400000 [00:31<00:10, 9631.84it/s] 75%|███████▍  | 299777/400000 [00:31<00:10, 9772.62it/s] 75%|███████▌  | 300760/400000 [00:31<00:10, 9786.83it/s] 75%|███████▌  | 301740/400000 [00:31<00:10, 9734.20it/s] 76%|███████▌  | 302716/400000 [00:31<00:09, 9741.62it/s] 76%|███████▌  | 303695/400000 [00:31<00:09, 9755.50it/s] 76%|███████▌  | 304671/400000 [00:31<00:09, 9658.39it/s] 76%|███████▋  | 305682/400000 [00:31<00:09, 9786.90it/s] 77%|███████▋  | 306691/400000 [00:32<00:09, 9874.08it/s] 77%|███████▋  | 307680/400000 [00:32<00:09, 9787.75it/s] 77%|███████▋  | 308660/400000 [00:32<00:09, 9453.73it/s] 77%|███████▋  | 309674/400000 [00:32<00:09, 9648.26it/s] 78%|███████▊  | 310686/400000 [00:32<00:09, 9782.71it/s] 78%|███████▊  | 311667/400000 [00:32<00:09, 9751.01it/s] 78%|███████▊  | 312676/400000 [00:32<00:08, 9848.39it/s] 78%|███████▊  | 313663/400000 [00:32<00:08, 9720.77it/s] 79%|███████▊  | 314639/400000 [00:32<00:08, 9729.98it/s] 79%|███████▉  | 315651/400000 [00:32<00:08, 9843.66it/s] 79%|███████▉  | 316637/400000 [00:33<00:08, 9677.04it/s] 79%|███████▉  | 317607/400000 [00:33<00:08, 9673.29it/s] 80%|███████▉  | 318577/400000 [00:33<00:08, 9679.55it/s] 80%|███████▉  | 319575/400000 [00:33<00:08, 9765.74it/s] 80%|████████  | 320574/400000 [00:33<00:08, 9831.49it/s] 80%|████████  | 321574/400000 [00:33<00:07, 9880.82it/s] 81%|████████  | 322563/400000 [00:33<00:07, 9840.11it/s] 81%|████████  | 323548/400000 [00:33<00:07, 9789.16it/s] 81%|████████  | 324528/400000 [00:33<00:07, 9632.48it/s] 81%|████████▏ | 325493/400000 [00:33<00:07, 9596.50it/s] 82%|████████▏ | 326455/400000 [00:34<00:07, 9602.26it/s] 82%|████████▏ | 327416/400000 [00:34<00:07, 9597.09it/s] 82%|████████▏ | 328398/400000 [00:34<00:07, 9662.65it/s] 82%|████████▏ | 329413/400000 [00:34<00:07, 9801.52it/s] 83%|████████▎ | 330428/400000 [00:34<00:07, 9900.81it/s] 83%|████████▎ | 331424/400000 [00:34<00:06, 9917.90it/s] 83%|████████▎ | 332417/400000 [00:34<00:06, 9684.93it/s] 83%|████████▎ | 333388/400000 [00:34<00:06, 9681.34it/s] 84%|████████▎ | 334358/400000 [00:34<00:06, 9531.71it/s] 84%|████████▍ | 335377/400000 [00:34<00:06, 9718.12it/s] 84%|████████▍ | 336359/400000 [00:35<00:06, 9745.49it/s] 84%|████████▍ | 337335/400000 [00:35<00:06, 9749.35it/s] 85%|████████▍ | 338311/400000 [00:35<00:06, 9742.71it/s] 85%|████████▍ | 339330/400000 [00:35<00:06, 9871.51it/s] 85%|████████▌ | 340346/400000 [00:35<00:05, 9955.03it/s] 85%|████████▌ | 341349/400000 [00:35<00:05, 9976.82it/s] 86%|████████▌ | 342371/400000 [00:35<00:05, 10046.61it/s] 86%|████████▌ | 343377/400000 [00:35<00:05, 9771.02it/s]  86%|████████▌ | 344357/400000 [00:35<00:05, 9708.31it/s] 86%|████████▋ | 345370/400000 [00:35<00:05, 9829.44it/s] 87%|████████▋ | 346384/400000 [00:36<00:05, 9920.53it/s] 87%|████████▋ | 347378/400000 [00:36<00:05, 9601.69it/s] 87%|████████▋ | 348342/400000 [00:36<00:05, 9598.43it/s] 87%|████████▋ | 349304/400000 [00:36<00:05, 9362.89it/s] 88%|████████▊ | 350244/400000 [00:36<00:05, 8978.26it/s] 88%|████████▊ | 351148/400000 [00:36<00:05, 8715.24it/s] 88%|████████▊ | 352031/400000 [00:36<00:05, 8747.11it/s] 88%|████████▊ | 353001/400000 [00:36<00:05, 9012.39it/s] 88%|████████▊ | 353948/400000 [00:36<00:05, 9141.72it/s] 89%|████████▊ | 354950/400000 [00:37<00:04, 9386.69it/s] 89%|████████▉ | 355898/400000 [00:37<00:04, 9413.11it/s] 89%|████████▉ | 356843/400000 [00:37<00:04, 9289.60it/s] 89%|████████▉ | 357861/400000 [00:37<00:04, 9539.11it/s] 90%|████████▉ | 358867/400000 [00:37<00:04, 9688.65it/s] 90%|████████▉ | 359847/400000 [00:37<00:04, 9719.75it/s] 90%|█████████ | 360856/400000 [00:37<00:03, 9827.70it/s] 90%|█████████ | 361841/400000 [00:37<00:03, 9713.04it/s] 91%|█████████ | 362814/400000 [00:37<00:03, 9546.36it/s] 91%|█████████ | 363794/400000 [00:37<00:03, 9620.87it/s] 91%|█████████ | 364758/400000 [00:38<00:03, 9609.05it/s] 91%|█████████▏| 365760/400000 [00:38<00:03, 9727.86it/s] 92%|█████████▏| 366734/400000 [00:38<00:03, 9671.65it/s] 92%|█████████▏| 367735/400000 [00:38<00:03, 9768.67it/s] 92%|█████████▏| 368731/400000 [00:38<00:03, 9825.13it/s] 92%|█████████▏| 369715/400000 [00:38<00:03, 9791.51it/s] 93%|█████████▎| 370726/400000 [00:38<00:02, 9884.28it/s] 93%|█████████▎| 371715/400000 [00:38<00:02, 9761.38it/s] 93%|█████████▎| 372692/400000 [00:38<00:02, 9712.70it/s] 93%|█████████▎| 373712/400000 [00:38<00:02, 9851.99it/s] 94%|█████████▎| 374699/400000 [00:39<00:02, 9692.90it/s] 94%|█████████▍| 375674/400000 [00:39<00:02, 9709.02it/s] 94%|█████████▍| 376646/400000 [00:39<00:02, 9579.27it/s] 94%|█████████▍| 377605/400000 [00:39<00:02, 9572.80it/s] 95%|█████████▍| 378604/400000 [00:39<00:02, 9691.94it/s] 95%|█████████▍| 379607/400000 [00:39<00:02, 9788.09it/s] 95%|█████████▌| 380621/400000 [00:39<00:01, 9890.93it/s] 95%|█████████▌| 381611/400000 [00:39<00:01, 9828.86it/s] 96%|█████████▌| 382595/400000 [00:39<00:01, 9719.84it/s] 96%|█████████▌| 383606/400000 [00:39<00:01, 9833.06it/s] 96%|█████████▌| 384591/400000 [00:40<00:01, 9802.87it/s] 96%|█████████▋| 385605/400000 [00:40<00:01, 9899.01it/s] 97%|█████████▋| 386596/400000 [00:40<00:01, 9831.59it/s] 97%|█████████▋| 387583/400000 [00:40<00:01, 9843.07it/s] 97%|█████████▋| 388594/400000 [00:40<00:01, 9921.30it/s] 97%|█████████▋| 389596/400000 [00:40<00:01, 9949.95it/s] 98%|█████████▊| 390615/400000 [00:40<00:00, 10019.85it/s] 98%|█████████▊| 391618/400000 [00:40<00:00, 9898.53it/s]  98%|█████████▊| 392609/400000 [00:40<00:00, 9775.44it/s] 98%|█████████▊| 393593/400000 [00:40<00:00, 9792.06it/s] 99%|█████████▊| 394580/400000 [00:41<00:00, 9815.08it/s] 99%|█████████▉| 395576/400000 [00:41<00:00, 9855.50it/s] 99%|█████████▉| 396562/400000 [00:41<00:00, 9807.48it/s] 99%|█████████▉| 397549/400000 [00:41<00:00, 9823.80it/s]100%|█████████▉| 398532/400000 [00:41<00:00, 9718.36it/s]100%|█████████▉| 399537/400000 [00:41<00:00, 9815.39it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9610.47it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f27090aee48>, <torchtext.data.dataset.TabularDataset object at 0x7f27090aef98>, <torchtext.vocab.Vocab object at 0x7f27090aeeb8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.50 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.94 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.31 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.31 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.23 file/s]2020-07-26 00:16:48.588470: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-26 00:16:48.591596: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-26 00:16:48.591747: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5631f8a29d50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-26 00:16:48.591763: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 152711.65it/s] 76%|███████▌  | 7495680/9912422 [00:00<00:11, 217966.35it/s]9920512it [00:00, 43578292.53it/s]                           
0it [00:00, ?it/s]32768it [00:00, 563879.88it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 468439.68it/s]1654784it [00:00, 11908337.18it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202447.17it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
