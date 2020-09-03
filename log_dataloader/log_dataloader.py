
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '3fc653a7999aa2761b089e07ab0db303a5051c0b', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/3fc653a7999aa2761b089e07ab0db303a5051c0b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f0d3b888730> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f0d3b888730> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f0da65b1488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f0da65b1488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0dc57d0e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0dc57d0e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0d538e8510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0d538e8510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0d538e8510> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:22, 120678.62it/s] 80%|███████▉  | 7905280/9912422 [00:00<00:11, 172273.84it/s]9920512it [00:00, 38234177.07it/s]                           
0it [00:00, ?it/s]32768it [00:00, 569903.03it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 473188.82it/s]1654784it [00:00, 11527260.26it/s]                         
0it [00:00, ?it/s]8192it [00:00, 163773.78it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d3b7bef60>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d3ab04c88>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0d538e81e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0d538e81e0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f0d538e81e0> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:44,  1.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:44,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:43,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:42,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:42,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:41,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:40,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:10,  2.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:10,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:10,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:10,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:09,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:09,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:08,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:08,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:07,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:47,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:47,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:46,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:46,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:45,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:45,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:31,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:31,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:31,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:31,  4.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:30,  4.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:30,  4.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:30,  4.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:21,  6.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:21,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:21,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:21,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:21,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:21,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:20,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:20,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:14,  8.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:14,  8.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:14,  8.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:14,  8.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:14,  8.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:14,  8.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:14,  8.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:14,  8.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:13,  8.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 15.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 32.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 32.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 32.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 32.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 38.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 38.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 38.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 38.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 52.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 52.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 57.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 57.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 57.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 57.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 57.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 57.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 57.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 61.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.76s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.76s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.76s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.15 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.76s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 75.15 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.25 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ url]
0 examples [00:00, ? examples/s]2020-09-03 18:06:54.168296: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-03 18:06:54.178704: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-09-03 18:06:54.180060: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564818928080 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-03 18:06:54.180083: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
46 examples [00:00, 457.89 examples/s]143 examples [00:00, 543.50 examples/s]247 examples [00:00, 633.53 examples/s]348 examples [00:00, 713.25 examples/s]448 examples [00:00, 780.04 examples/s]545 examples [00:00, 827.23 examples/s]646 examples [00:00, 873.79 examples/s]753 examples [00:00, 921.62 examples/s]853 examples [00:00, 943.54 examples/s]949 examples [00:01, 939.06 examples/s]1045 examples [00:01, 931.75 examples/s]1139 examples [00:01, 930.37 examples/s]1234 examples [00:01, 934.45 examples/s]1332 examples [00:01, 947.13 examples/s]1430 examples [00:01, 954.47 examples/s]1527 examples [00:01, 958.09 examples/s]1625 examples [00:01, 963.98 examples/s]1722 examples [00:01, 953.85 examples/s]1818 examples [00:01, 939.57 examples/s]1915 examples [00:02, 948.07 examples/s]2011 examples [00:02, 950.50 examples/s]2107 examples [00:02, 948.45 examples/s]2204 examples [00:02, 952.80 examples/s]2303 examples [00:02, 961.83 examples/s]2400 examples [00:02, 960.48 examples/s]2500 examples [00:02, 970.44 examples/s]2600 examples [00:02, 976.97 examples/s]2699 examples [00:02, 978.64 examples/s]2797 examples [00:02, 976.56 examples/s]2895 examples [00:03, 948.43 examples/s]2991 examples [00:03, 948.02 examples/s]3089 examples [00:03, 955.12 examples/s]3188 examples [00:03, 964.97 examples/s]3285 examples [00:03, 943.19 examples/s]3383 examples [00:03, 953.06 examples/s]3479 examples [00:03, 941.45 examples/s]3578 examples [00:03, 953.99 examples/s]3676 examples [00:03, 959.00 examples/s]3773 examples [00:03, 954.93 examples/s]3870 examples [00:04, 958.03 examples/s]3966 examples [00:04, 956.29 examples/s]4063 examples [00:04, 959.83 examples/s]4162 examples [00:04, 968.08 examples/s]4260 examples [00:04, 969.85 examples/s]4358 examples [00:04, 965.98 examples/s]4457 examples [00:04, 971.63 examples/s]4555 examples [00:04, 973.44 examples/s]4653 examples [00:04, 948.29 examples/s]4748 examples [00:04, 939.48 examples/s]4843 examples [00:05, 921.00 examples/s]4938 examples [00:05, 927.51 examples/s]5031 examples [00:05, 916.83 examples/s]5126 examples [00:05, 926.45 examples/s]5219 examples [00:05, 923.27 examples/s]5312 examples [00:05, 905.62 examples/s]5408 examples [00:05, 920.45 examples/s]5504 examples [00:05, 929.82 examples/s]5598 examples [00:05, 897.43 examples/s]5700 examples [00:06, 930.07 examples/s]5799 examples [00:06, 947.02 examples/s]5898 examples [00:06, 959.41 examples/s]5995 examples [00:06, 960.57 examples/s]6092 examples [00:06, 959.05 examples/s]6189 examples [00:06, 959.95 examples/s]6286 examples [00:06, 956.98 examples/s]6385 examples [00:06, 964.29 examples/s]6482 examples [00:06, 946.64 examples/s]6579 examples [00:06, 952.29 examples/s]6678 examples [00:07, 960.63 examples/s]6775 examples [00:07, 955.40 examples/s]6871 examples [00:07, 938.64 examples/s]6967 examples [00:07, 944.57 examples/s]7065 examples [00:07, 952.81 examples/s]7163 examples [00:07, 959.06 examples/s]7263 examples [00:07, 969.72 examples/s]7367 examples [00:07, 989.67 examples/s]7470 examples [00:07, 997.70 examples/s]7570 examples [00:07, 982.86 examples/s]7669 examples [00:08, 979.89 examples/s]7768 examples [00:08, 977.48 examples/s]7866 examples [00:08, 972.12 examples/s]7968 examples [00:08, 985.05 examples/s]8069 examples [00:08, 990.04 examples/s]8169 examples [00:08, 987.92 examples/s]8270 examples [00:08, 993.66 examples/s]8372 examples [00:08, 999.31 examples/s]8472 examples [00:08, 975.35 examples/s]8570 examples [00:08, 966.89 examples/s]8667 examples [00:09, 953.64 examples/s]8763 examples [00:09, 954.27 examples/s]8861 examples [00:09, 960.98 examples/s]8959 examples [00:09, 965.64 examples/s]9056 examples [00:09, 966.32 examples/s]9154 examples [00:09, 967.31 examples/s]9251 examples [00:09, 961.10 examples/s]9351 examples [00:09, 971.88 examples/s]9451 examples [00:09, 978.86 examples/s]9551 examples [00:09, 983.98 examples/s]9652 examples [00:10, 989.66 examples/s]9752 examples [00:10, 988.28 examples/s]9852 examples [00:10, 990.02 examples/s]9955 examples [00:10, 999.54 examples/s]10055 examples [00:10, 932.37 examples/s]10150 examples [00:10, 935.99 examples/s]10245 examples [00:10, 913.04 examples/s]10341 examples [00:10, 925.33 examples/s]10443 examples [00:10, 949.66 examples/s]10541 examples [00:11, 956.75 examples/s]10641 examples [00:11, 968.44 examples/s]10741 examples [00:11, 976.05 examples/s]10841 examples [00:11, 980.09 examples/s]10940 examples [00:11, 968.69 examples/s]11038 examples [00:11, 965.78 examples/s]11135 examples [00:11, 965.06 examples/s]11232 examples [00:11, 962.23 examples/s]11329 examples [00:11, 954.82 examples/s]11425 examples [00:11, 939.43 examples/s]11523 examples [00:12, 949.58 examples/s]11621 examples [00:12, 958.29 examples/s]11722 examples [00:12, 970.15 examples/s]11823 examples [00:12, 981.57 examples/s]11922 examples [00:12, 976.92 examples/s]12020 examples [00:12, 977.84 examples/s]12118 examples [00:12, 974.87 examples/s]12217 examples [00:12, 977.21 examples/s]12317 examples [00:12, 982.51 examples/s]12418 examples [00:12, 987.94 examples/s]12517 examples [00:13, 979.93 examples/s]12616 examples [00:13, 975.72 examples/s]12714 examples [00:13, 932.49 examples/s]12817 examples [00:13, 959.53 examples/s]12920 examples [00:13, 976.98 examples/s]13022 examples [00:13, 987.49 examples/s]13122 examples [00:13, 987.13 examples/s]13222 examples [00:13, 990.45 examples/s]13322 examples [00:13, 981.97 examples/s]13421 examples [00:13, 934.49 examples/s]13516 examples [00:14, 937.91 examples/s]13611 examples [00:14, 940.47 examples/s]13712 examples [00:14, 958.70 examples/s]13812 examples [00:14, 969.76 examples/s]13911 examples [00:14, 973.41 examples/s]14009 examples [00:14, 960.87 examples/s]14106 examples [00:14, 958.12 examples/s]14206 examples [00:14, 969.97 examples/s]14304 examples [00:14, 971.66 examples/s]14405 examples [00:15, 981.72 examples/s]14506 examples [00:15, 989.33 examples/s]14606 examples [00:15, 980.09 examples/s]14708 examples [00:15, 989.33 examples/s]14808 examples [00:15, 959.86 examples/s]14908 examples [00:15, 970.85 examples/s]15010 examples [00:15, 984.14 examples/s]15109 examples [00:15, 974.41 examples/s]15210 examples [00:15, 983.37 examples/s]15312 examples [00:15, 991.70 examples/s]15412 examples [00:16, 984.68 examples/s]15511 examples [00:16, 961.40 examples/s]15610 examples [00:16, 969.03 examples/s]15709 examples [00:16, 973.17 examples/s]15807 examples [00:16, 971.66 examples/s]15905 examples [00:16, 965.79 examples/s]16002 examples [00:16, 954.03 examples/s]16100 examples [00:16, 961.23 examples/s]16200 examples [00:16, 970.43 examples/s]16299 examples [00:16, 974.25 examples/s]16397 examples [00:17, 968.94 examples/s]16499 examples [00:17, 980.77 examples/s]16598 examples [00:17, 980.29 examples/s]16698 examples [00:17, 984.49 examples/s]16797 examples [00:17, 981.76 examples/s]16897 examples [00:17, 985.63 examples/s]17003 examples [00:17, 1003.82 examples/s]17104 examples [00:17, 978.36 examples/s] 17203 examples [00:17, 977.42 examples/s]17301 examples [00:17, 975.44 examples/s]17399 examples [00:18, 974.91 examples/s]17498 examples [00:18, 977.66 examples/s]17601 examples [00:18, 991.25 examples/s]17701 examples [00:18, 988.33 examples/s]17800 examples [00:18, 983.91 examples/s]17899 examples [00:18, 981.09 examples/s]17998 examples [00:18, 976.11 examples/s]18099 examples [00:18, 983.45 examples/s]18198 examples [00:18, 978.46 examples/s]18296 examples [00:19, 904.43 examples/s]18388 examples [00:19, 902.83 examples/s]18483 examples [00:19, 916.46 examples/s]18578 examples [00:19, 925.81 examples/s]18673 examples [00:19, 932.10 examples/s]18769 examples [00:19, 938.55 examples/s]18865 examples [00:19, 944.23 examples/s]18960 examples [00:19, 942.88 examples/s]19058 examples [00:19, 951.50 examples/s]19154 examples [00:19, 948.80 examples/s]19249 examples [00:20, 929.36 examples/s]19348 examples [00:20, 945.52 examples/s]19443 examples [00:20, 923.72 examples/s]19541 examples [00:20, 938.99 examples/s]19636 examples [00:20, 918.80 examples/s]19729 examples [00:20, 916.95 examples/s]19821 examples [00:20, 914.94 examples/s]19918 examples [00:20, 928.98 examples/s]20012 examples [00:20, 884.89 examples/s]20102 examples [00:20, 887.38 examples/s]20194 examples [00:21, 895.72 examples/s]20291 examples [00:21, 914.56 examples/s]20393 examples [00:21, 943.72 examples/s]20488 examples [00:21, 937.17 examples/s]20583 examples [00:21, 939.29 examples/s]20678 examples [00:21, 936.61 examples/s]20772 examples [00:21, 936.75 examples/s]20866 examples [00:21, 937.19 examples/s]20963 examples [00:21, 946.30 examples/s]21059 examples [00:21, 949.61 examples/s]21155 examples [00:22, 928.92 examples/s]21255 examples [00:22, 949.09 examples/s]21355 examples [00:22, 960.76 examples/s]21452 examples [00:22, 962.38 examples/s]21551 examples [00:22, 967.81 examples/s]21649 examples [00:22, 971.35 examples/s]21747 examples [00:22, 934.40 examples/s]21847 examples [00:22, 951.70 examples/s]21946 examples [00:22, 960.52 examples/s]22043 examples [00:23, 951.63 examples/s]22142 examples [00:23, 961.77 examples/s]22240 examples [00:23, 965.99 examples/s]22340 examples [00:23, 975.51 examples/s]22443 examples [00:23, 989.87 examples/s]22543 examples [00:23, 982.69 examples/s]22642 examples [00:23, 969.01 examples/s]22740 examples [00:23, 965.87 examples/s]22837 examples [00:23, 961.16 examples/s]22936 examples [00:23, 965.27 examples/s]23035 examples [00:24, 971.59 examples/s]23133 examples [00:24, 945.63 examples/s]23236 examples [00:24, 966.57 examples/s]23333 examples [00:24, 967.07 examples/s]23431 examples [00:24, 968.40 examples/s]23531 examples [00:24, 974.97 examples/s]23629 examples [00:24, 970.42 examples/s]23729 examples [00:24, 977.00 examples/s]23828 examples [00:24, 979.18 examples/s]23927 examples [00:24, 980.96 examples/s]24026 examples [00:25, 978.63 examples/s]24124 examples [00:25, 967.18 examples/s]24223 examples [00:25, 971.15 examples/s]24323 examples [00:25, 976.92 examples/s]24424 examples [00:25, 985.73 examples/s]24526 examples [00:25, 988.09 examples/s]24625 examples [00:25, 984.92 examples/s]24724 examples [00:25, 978.25 examples/s]24823 examples [00:25, 980.87 examples/s]24922 examples [00:25, 978.01 examples/s]25023 examples [00:26, 986.07 examples/s]25122 examples [00:26, 970.39 examples/s]25220 examples [00:26, 949.46 examples/s]25320 examples [00:26, 962.04 examples/s]25419 examples [00:26, 968.32 examples/s]25517 examples [00:26, 969.53 examples/s]25615 examples [00:26, 953.09 examples/s]25713 examples [00:26, 960.21 examples/s]25810 examples [00:26, 952.41 examples/s]25906 examples [00:26, 945.12 examples/s]26001 examples [00:27, 939.41 examples/s]26095 examples [00:27, 938.88 examples/s]26192 examples [00:27, 945.24 examples/s]26288 examples [00:27, 947.22 examples/s]26383 examples [00:27, 914.92 examples/s]26478 examples [00:27, 923.98 examples/s]26580 examples [00:27, 949.88 examples/s]26677 examples [00:27, 954.27 examples/s]26773 examples [00:27, 949.85 examples/s]26871 examples [00:28, 956.58 examples/s]26970 examples [00:28, 964.97 examples/s]27067 examples [00:28, 961.33 examples/s]27165 examples [00:28, 965.60 examples/s]27263 examples [00:28, 969.46 examples/s]27370 examples [00:28, 997.38 examples/s]27473 examples [00:28, 1006.84 examples/s]27574 examples [00:28, 975.97 examples/s] 27672 examples [00:28, 957.92 examples/s]27769 examples [00:28, 954.74 examples/s]27865 examples [00:29, 956.05 examples/s]27961 examples [00:29, 947.97 examples/s]28056 examples [00:29, 935.05 examples/s]28153 examples [00:29, 942.74 examples/s]28250 examples [00:29, 948.33 examples/s]28345 examples [00:29, 934.06 examples/s]28441 examples [00:29, 940.03 examples/s]28540 examples [00:29, 952.80 examples/s]28636 examples [00:29, 933.89 examples/s]28730 examples [00:29, 934.51 examples/s]28827 examples [00:30, 942.82 examples/s]28922 examples [00:30, 934.39 examples/s]29016 examples [00:30, 934.96 examples/s]29119 examples [00:30, 958.92 examples/s]29216 examples [00:30, 944.92 examples/s]29315 examples [00:30, 955.73 examples/s]29413 examples [00:30, 960.70 examples/s]29515 examples [00:30, 976.00 examples/s]29614 examples [00:30, 978.55 examples/s]29712 examples [00:30, 972.23 examples/s]29810 examples [00:31, 969.81 examples/s]29908 examples [00:31, 947.01 examples/s]30003 examples [00:31, 878.44 examples/s]30099 examples [00:31, 900.17 examples/s]30194 examples [00:31, 913.36 examples/s]30287 examples [00:31, 917.97 examples/s]30382 examples [00:31, 926.01 examples/s]30475 examples [00:31, 925.06 examples/s]30574 examples [00:31, 942.98 examples/s]30673 examples [00:32, 954.10 examples/s]30769 examples [00:32, 952.15 examples/s]30865 examples [00:32, 940.98 examples/s]30960 examples [00:32, 918.57 examples/s]31064 examples [00:32, 949.79 examples/s]31168 examples [00:32, 975.14 examples/s]31269 examples [00:32, 985.20 examples/s]31368 examples [00:32, 983.11 examples/s]31468 examples [00:32, 985.15 examples/s]31567 examples [00:32, 985.10 examples/s]31666 examples [00:33, 946.77 examples/s]31762 examples [00:33, 943.00 examples/s]31857 examples [00:33, 940.44 examples/s]31955 examples [00:33, 950.69 examples/s]32051 examples [00:33, 951.42 examples/s]32150 examples [00:33, 960.93 examples/s]32247 examples [00:33, 954.16 examples/s]32346 examples [00:33, 963.15 examples/s]32445 examples [00:33, 968.88 examples/s]32542 examples [00:33, 954.51 examples/s]32638 examples [00:34, 942.29 examples/s]32733 examples [00:34, 926.53 examples/s]32832 examples [00:34, 943.58 examples/s]32934 examples [00:34, 964.76 examples/s]33031 examples [00:34, 964.53 examples/s]33130 examples [00:34, 970.78 examples/s]33228 examples [00:34, 938.48 examples/s]33324 examples [00:34, 944.10 examples/s]33423 examples [00:34, 953.45 examples/s]33519 examples [00:35, 932.84 examples/s]33617 examples [00:35, 945.06 examples/s]33712 examples [00:35, 934.15 examples/s]33808 examples [00:35, 939.53 examples/s]33905 examples [00:35, 945.96 examples/s]34004 examples [00:35, 956.90 examples/s]34102 examples [00:35, 961.27 examples/s]34201 examples [00:35, 967.48 examples/s]34300 examples [00:35, 971.93 examples/s]34398 examples [00:35, 966.88 examples/s]34495 examples [00:36, 966.59 examples/s]34592 examples [00:36, 959.04 examples/s]34691 examples [00:36, 967.95 examples/s]34790 examples [00:36, 971.53 examples/s]34888 examples [00:36, 969.30 examples/s]34985 examples [00:36, 962.03 examples/s]35082 examples [00:36, 955.93 examples/s]35181 examples [00:36, 964.73 examples/s]35278 examples [00:36, 934.67 examples/s]35372 examples [00:36, 934.05 examples/s]35470 examples [00:37, 946.90 examples/s]35567 examples [00:37, 950.95 examples/s]35663 examples [00:37, 952.66 examples/s]35762 examples [00:37, 961.75 examples/s]35864 examples [00:37, 975.99 examples/s]35962 examples [00:37, 963.39 examples/s]36062 examples [00:37, 971.48 examples/s]36162 examples [00:37, 977.39 examples/s]36260 examples [00:37, 973.37 examples/s]36359 examples [00:37, 977.44 examples/s]36459 examples [00:38, 981.88 examples/s]36558 examples [00:38, 973.41 examples/s]36656 examples [00:38, 966.54 examples/s]36754 examples [00:38, 969.97 examples/s]36857 examples [00:38, 984.59 examples/s]36956 examples [00:38, 983.96 examples/s]37055 examples [00:38, 974.17 examples/s]37156 examples [00:38, 981.65 examples/s]37255 examples [00:38, 971.57 examples/s]37353 examples [00:38, 951.93 examples/s]37449 examples [00:39, 943.34 examples/s]37544 examples [00:39, 941.11 examples/s]37642 examples [00:39, 950.83 examples/s]37741 examples [00:39, 960.52 examples/s]37838 examples [00:39, 928.65 examples/s]37932 examples [00:39, 931.01 examples/s]38031 examples [00:39, 945.98 examples/s]38129 examples [00:39, 953.71 examples/s]38225 examples [00:39, 951.00 examples/s]38326 examples [00:39, 966.43 examples/s]38424 examples [00:40, 970.19 examples/s]38526 examples [00:40, 983.74 examples/s]38630 examples [00:40, 999.30 examples/s]38731 examples [00:40, 998.87 examples/s]38831 examples [00:40, 996.38 examples/s]38937 examples [00:40, 1013.85 examples/s]39042 examples [00:40, 1022.53 examples/s]39145 examples [00:40, 1015.63 examples/s]39247 examples [00:40, 1003.93 examples/s]39348 examples [00:41, 1000.49 examples/s]39449 examples [00:41, 993.31 examples/s] 39549 examples [00:41, 968.91 examples/s]39650 examples [00:41, 979.56 examples/s]39749 examples [00:41, 977.28 examples/s]39852 examples [00:41, 991.24 examples/s]39952 examples [00:41, 991.89 examples/s]40052 examples [00:41, 927.72 examples/s]40146 examples [00:41, 905.26 examples/s]40239 examples [00:41, 911.51 examples/s]40335 examples [00:42, 914.99 examples/s]40427 examples [00:42, 914.94 examples/s]40528 examples [00:42, 941.02 examples/s]40630 examples [00:42, 963.38 examples/s]40727 examples [00:42, 964.55 examples/s]40829 examples [00:42, 980.45 examples/s]40928 examples [00:42, 978.55 examples/s]41027 examples [00:42, 979.27 examples/s]41126 examples [00:42, 976.69 examples/s]41224 examples [00:42, 970.28 examples/s]41322 examples [00:43, 965.02 examples/s]41419 examples [00:43, 962.97 examples/s]41516 examples [00:43, 951.35 examples/s]41616 examples [00:43, 964.44 examples/s]41714 examples [00:43, 966.40 examples/s]41811 examples [00:43, 966.16 examples/s]41908 examples [00:43, 965.96 examples/s]42007 examples [00:43, 970.53 examples/s]42105 examples [00:43, 965.85 examples/s]42202 examples [00:43, 949.48 examples/s]42300 examples [00:44, 955.64 examples/s]42396 examples [00:44, 947.60 examples/s]42491 examples [00:44, 914.27 examples/s]42588 examples [00:44, 930.28 examples/s]42688 examples [00:44, 948.42 examples/s]42786 examples [00:44, 955.61 examples/s]42882 examples [00:44, 950.78 examples/s]42978 examples [00:44, 940.12 examples/s]43074 examples [00:44, 945.57 examples/s]43169 examples [00:45, 946.61 examples/s]43268 examples [00:45, 958.52 examples/s]43371 examples [00:45, 976.65 examples/s]43469 examples [00:45, 976.07 examples/s]43568 examples [00:45, 979.08 examples/s]43669 examples [00:45, 987.21 examples/s]43768 examples [00:45, 978.74 examples/s]43867 examples [00:45, 981.94 examples/s]43969 examples [00:45, 991.11 examples/s]44070 examples [00:45, 995.81 examples/s]44170 examples [00:46, 981.73 examples/s]44269 examples [00:46, 965.11 examples/s]44370 examples [00:46, 977.18 examples/s]44471 examples [00:46, 985.22 examples/s]44570 examples [00:46, 981.98 examples/s]44669 examples [00:46, 969.72 examples/s]44767 examples [00:46, 936.71 examples/s]44865 examples [00:46, 948.34 examples/s]44964 examples [00:46, 957.85 examples/s]45060 examples [00:46, 954.55 examples/s]45156 examples [00:47, 945.21 examples/s]45256 examples [00:47, 958.80 examples/s]45356 examples [00:47, 968.87 examples/s]45456 examples [00:47, 975.81 examples/s]45556 examples [00:47, 982.64 examples/s]45658 examples [00:47, 991.23 examples/s]45758 examples [00:47, 983.76 examples/s]45857 examples [00:47, 976.71 examples/s]45955 examples [00:47, 963.44 examples/s]46052 examples [00:47, 956.84 examples/s]46148 examples [00:48, 936.86 examples/s]46246 examples [00:48, 947.08 examples/s]46348 examples [00:48, 966.05 examples/s]46447 examples [00:48, 971.36 examples/s]46546 examples [00:48, 974.81 examples/s]46647 examples [00:48, 983.12 examples/s]46746 examples [00:48, 968.08 examples/s]46846 examples [00:48, 976.46 examples/s]46944 examples [00:48, 962.46 examples/s]47041 examples [00:49, 939.83 examples/s]47136 examples [00:49, 931.80 examples/s]47237 examples [00:49, 950.67 examples/s]47340 examples [00:49, 971.32 examples/s]47438 examples [00:49, 970.49 examples/s]47538 examples [00:49, 977.45 examples/s]47636 examples [00:49, 964.85 examples/s]47733 examples [00:49, 961.97 examples/s]47834 examples [00:49, 975.49 examples/s]47933 examples [00:49, 978.15 examples/s]48031 examples [00:50, 977.51 examples/s]48132 examples [00:50, 985.34 examples/s]48234 examples [00:50, 993.88 examples/s]48336 examples [00:50, 1000.55 examples/s]48437 examples [00:50, 949.83 examples/s] 48540 examples [00:50, 972.01 examples/s]48638 examples [00:50, 972.05 examples/s]48736 examples [00:50, 970.58 examples/s]48836 examples [00:50, 979.07 examples/s]48935 examples [00:50, 978.86 examples/s]49039 examples [00:51, 993.83 examples/s]49139 examples [00:51, 982.87 examples/s]49238 examples [00:51, 973.00 examples/s]49336 examples [00:51, 950.84 examples/s]49432 examples [00:51, 926.20 examples/s]49527 examples [00:51, 931.48 examples/s]49624 examples [00:51, 939.57 examples/s]49719 examples [00:51, 940.06 examples/s]49815 examples [00:51, 943.82 examples/s]49917 examples [00:51, 963.15 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7176/50000 [00:00<00:00, 71757.33 examples/s] 40%|████      | 20181/50000 [00:00<00:00, 82905.58 examples/s] 65%|██████▌   | 32670/50000 [00:00<00:00, 92203.96 examples/s] 91%|█████████ | 45317/50000 [00:00<00:00, 100360.73 examples/s]                                                                0 examples [00:00, ? examples/s]74 examples [00:00, 733.83 examples/s]177 examples [00:00, 801.68 examples/s]275 examples [00:00, 846.80 examples/s]378 examples [00:00, 893.29 examples/s]482 examples [00:00, 932.43 examples/s]579 examples [00:00, 942.50 examples/s]681 examples [00:00, 961.75 examples/s]784 examples [00:00, 978.52 examples/s]888 examples [00:00, 994.06 examples/s]986 examples [00:01, 984.67 examples/s]1087 examples [00:01, 990.17 examples/s]1185 examples [00:01, 980.34 examples/s]1284 examples [00:01, 981.20 examples/s]1382 examples [00:01, 971.58 examples/s]1483 examples [00:01, 982.08 examples/s]1581 examples [00:01, 975.74 examples/s]1679 examples [00:01, 976.72 examples/s]1777 examples [00:01, 976.35 examples/s]1878 examples [00:01, 983.07 examples/s]1977 examples [00:02, 980.39 examples/s]2076 examples [00:02, 971.67 examples/s]2177 examples [00:02, 980.53 examples/s]2276 examples [00:02, 978.67 examples/s]2374 examples [00:02, 972.44 examples/s]2476 examples [00:02, 984.13 examples/s]2575 examples [00:02, 963.37 examples/s]2672 examples [00:02, 953.82 examples/s]2774 examples [00:02, 969.69 examples/s]2877 examples [00:02, 986.62 examples/s]2978 examples [00:03, 993.05 examples/s]3078 examples [00:03, 981.46 examples/s]3177 examples [00:03, 943.71 examples/s]3275 examples [00:03, 954.28 examples/s]3373 examples [00:03, 961.59 examples/s]3470 examples [00:03, 963.29 examples/s]3568 examples [00:03, 966.57 examples/s]3667 examples [00:03, 971.11 examples/s]3769 examples [00:03, 982.82 examples/s]3875 examples [00:03, 1003.53 examples/s]3976 examples [00:04, 997.17 examples/s] 4077 examples [00:04, 1000.40 examples/s]4178 examples [00:04, 1001.19 examples/s]4279 examples [00:04, 994.94 examples/s] 4379 examples [00:04, 986.75 examples/s]4479 examples [00:04, 988.83 examples/s]4578 examples [00:04, 950.03 examples/s]4674 examples [00:04, 945.43 examples/s]4769 examples [00:04, 783.35 examples/s]4867 examples [00:05, 824.72 examples/s]4962 examples [00:05, 857.42 examples/s]5060 examples [00:05, 889.93 examples/s]5155 examples [00:05, 905.53 examples/s]5248 examples [00:05, 905.90 examples/s]5351 examples [00:05, 938.01 examples/s]5446 examples [00:05, 930.62 examples/s]5540 examples [00:05, 928.67 examples/s]5637 examples [00:05, 938.99 examples/s]5732 examples [00:05, 940.66 examples/s]5827 examples [00:06, 934.17 examples/s]5921 examples [00:06, 932.69 examples/s]6019 examples [00:06, 944.14 examples/s]6114 examples [00:06, 941.83 examples/s]6209 examples [00:06, 930.57 examples/s]6303 examples [00:06, 932.71 examples/s]6400 examples [00:06, 940.68 examples/s]6498 examples [00:06, 951.26 examples/s]6595 examples [00:06, 954.07 examples/s]6691 examples [00:06, 954.64 examples/s]6788 examples [00:07, 956.88 examples/s]6884 examples [00:07, 955.81 examples/s]6982 examples [00:07, 962.92 examples/s]7081 examples [00:07, 968.43 examples/s]7183 examples [00:07, 980.74 examples/s]7282 examples [00:07, 975.28 examples/s]7380 examples [00:07, 969.67 examples/s]7481 examples [00:07, 979.99 examples/s]7581 examples [00:07, 985.40 examples/s]7685 examples [00:08, 999.32 examples/s]7786 examples [00:08, 997.60 examples/s]7886 examples [00:08, 979.49 examples/s]7985 examples [00:08, 978.12 examples/s]8083 examples [00:08, 971.01 examples/s]8182 examples [00:08, 975.84 examples/s]8281 examples [00:08, 978.72 examples/s]8385 examples [00:08, 995.82 examples/s]8485 examples [00:08, 988.23 examples/s]8586 examples [00:08, 993.88 examples/s]8686 examples [00:09, 992.23 examples/s]8786 examples [00:09, 991.86 examples/s]8886 examples [00:09, 992.80 examples/s]8986 examples [00:09, 983.74 examples/s]9085 examples [00:09, 983.03 examples/s]9184 examples [00:09, 980.16 examples/s]9288 examples [00:09, 994.99 examples/s]9393 examples [00:09, 1008.24 examples/s]9494 examples [00:09, 1007.75 examples/s]9595 examples [00:09, 1004.64 examples/s]9696 examples [00:10, 1005.73 examples/s]9798 examples [00:10, 1009.36 examples/s]9901 examples [00:10, 1015.32 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4OFG39/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4OFG39/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0d538e8510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0d538e8510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0d538e8510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d5194c278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0cddb72320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0d538e8510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0d538e8510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0d538e8510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0cddb721d0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d3ab27208>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.80 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.63 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.63 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.63 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.53 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.53 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.06 file/s]2020-09-03 18:08:04.392850: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-03 18:08:04.396589: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-09-03 18:08:04.396786: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56501d402320 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-03 18:08:04.396801: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'mnist2', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy', 'train', 'test'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 158090.65it/s] 66%|██████▋   | 6586368/9912422 [00:00<00:14, 225611.12it/s]9920512it [00:00, 42461939.21it/s]                           
0it [00:00, ?it/s]32768it [00:00, 473401.79it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 155410.64it/s]1654784it [00:00, 10972502.06it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186718.43it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
