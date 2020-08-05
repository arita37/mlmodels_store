
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6ca6da91408244e26c157e9e6467cc18ede43e71', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff68b530b70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff68b530b70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff6f619f510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff6f619f510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff7153cdea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff7153cdea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff6a34cea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff6a34cea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff6a34cea60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 19%|█▉        | 1925120/9912422 [00:00<00:00, 19070664.73it/s] 91%|█████████ | 9043968/9912422 [00:00<00:00, 24418290.87it/s]9920512it [00:00, 27848613.05it/s]                             
0it [00:00, ?it/s]32768it [00:00, 490713.20it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 469551.42it/s]1654784it [00:00, 11477280.23it/s]                         
0it [00:00, ?it/s]8192it [00:00, 208725.33it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff6a16a96a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff6a169a940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff6a34ce6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff6a34ce6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff6a34ce6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:00,  1.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:00,  1.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:59,  1.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:58,  1.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:58,  1.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:57,  1.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:56,  1.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:21,  1.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:21,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:21,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:20,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:20,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:19,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:19,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:18,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:18,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:54,  2.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:54,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:54,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:54,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:53,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:53,  2.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:53,  2.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:52,  2.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:52,  2.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:01<00:36,  3.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:36,  3.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:36,  3.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:36,  3.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:36,  3.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:35,  3.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:35,  3.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:35,  3.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:35,  3.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:24,  5.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:24,  5.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:24,  5.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:24,  5.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:24,  5.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:24,  5.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:23,  5.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:23,  5.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:23,  5.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:16,  7.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:16,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:16,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:16,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:16,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:16,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:16,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11, 10.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11, 10.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11, 10.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11, 10.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11, 10.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 18.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 18.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 18.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 18.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 18.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 29.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 29.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 29.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 42.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 48.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 48.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 48.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 48.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 48.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 48.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 48.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 48.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 48.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 54.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 73.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 73.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 73.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.88s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.88s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.88s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.47 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.32s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.88s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 73.47 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.32s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.32s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.44 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.32s/ url]
0 examples [00:00, ? examples/s]2020-08-05 12:10:04.752488: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-05 12:10:04.776247: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-05 12:10:04.777240: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b64170ae70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-05 12:10:04.777266: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
12 examples [00:00, 119.39 examples/s]97 examples [00:00, 160.77 examples/s]179 examples [00:00, 211.84 examples/s]257 examples [00:00, 270.84 examples/s]339 examples [00:00, 338.25 examples/s]424 examples [00:00, 412.68 examples/s]512 examples [00:00, 490.71 examples/s]595 examples [00:00, 558.09 examples/s]677 examples [00:00, 615.55 examples/s]758 examples [00:01, 661.92 examples/s]843 examples [00:01, 707.54 examples/s]927 examples [00:01, 741.99 examples/s]1013 examples [00:01, 773.43 examples/s]1096 examples [00:01, 785.63 examples/s]1179 examples [00:01, 798.23 examples/s]1262 examples [00:01, 798.56 examples/s]1347 examples [00:01, 810.51 examples/s]1430 examples [00:01, 806.83 examples/s]1514 examples [00:01, 815.35 examples/s]1597 examples [00:02, 811.75 examples/s]1681 examples [00:02, 819.90 examples/s]1764 examples [00:02, 817.26 examples/s]1846 examples [00:02, 807.03 examples/s]1931 examples [00:02, 818.48 examples/s]2015 examples [00:02, 824.23 examples/s]2098 examples [00:02, 820.27 examples/s]2181 examples [00:02, 814.91 examples/s]2263 examples [00:02, 811.51 examples/s]2345 examples [00:02, 799.70 examples/s]2426 examples [00:03, 794.30 examples/s]2506 examples [00:03, 780.51 examples/s]2590 examples [00:03, 797.10 examples/s]2678 examples [00:03, 819.36 examples/s]2764 examples [00:03, 829.62 examples/s]2848 examples [00:03, 805.96 examples/s]2929 examples [00:03, 797.23 examples/s]3016 examples [00:03, 816.45 examples/s]3100 examples [00:03, 821.20 examples/s]3185 examples [00:03, 828.29 examples/s]3273 examples [00:04, 841.39 examples/s]3358 examples [00:04, 819.16 examples/s]3441 examples [00:04, 797.37 examples/s]3528 examples [00:04, 815.83 examples/s]3610 examples [00:04, 807.83 examples/s]3693 examples [00:04, 812.61 examples/s]3775 examples [00:04, 812.50 examples/s]3862 examples [00:04, 827.70 examples/s]3945 examples [00:04, 823.04 examples/s]4028 examples [00:05, 778.41 examples/s]4112 examples [00:05, 789.24 examples/s]4192 examples [00:05, 783.24 examples/s]4280 examples [00:05, 807.80 examples/s]4366 examples [00:05, 821.82 examples/s]4449 examples [00:05, 820.38 examples/s]4532 examples [00:05, 806.31 examples/s]4617 examples [00:05, 800.13 examples/s]4702 examples [00:05, 812.13 examples/s]4785 examples [00:05, 816.93 examples/s]4869 examples [00:06, 821.10 examples/s]4953 examples [00:06, 824.52 examples/s]5036 examples [00:06, 821.61 examples/s]5123 examples [00:06, 833.63 examples/s]5211 examples [00:06, 846.30 examples/s]5296 examples [00:06, 841.50 examples/s]5381 examples [00:06, 829.28 examples/s]5465 examples [00:06, 831.78 examples/s]5549 examples [00:06, 832.56 examples/s]5636 examples [00:06, 841.62 examples/s]5721 examples [00:07, 811.44 examples/s]5804 examples [00:07, 815.65 examples/s]5886 examples [00:07, 809.79 examples/s]5968 examples [00:07, 776.02 examples/s]6054 examples [00:07, 797.12 examples/s]6140 examples [00:07, 812.71 examples/s]6230 examples [00:07, 834.95 examples/s]6319 examples [00:07, 849.82 examples/s]6407 examples [00:07, 858.15 examples/s]6494 examples [00:08, 854.58 examples/s]6580 examples [00:08, 831.25 examples/s]6664 examples [00:08, 825.96 examples/s]6747 examples [00:08, 825.82 examples/s]6830 examples [00:08, 819.94 examples/s]6913 examples [00:08, 804.63 examples/s]6999 examples [00:08, 819.52 examples/s]7084 examples [00:08, 827.66 examples/s]7167 examples [00:08, 811.23 examples/s]7249 examples [00:08, 797.72 examples/s]7329 examples [00:09, 764.93 examples/s]7406 examples [00:09, 760.88 examples/s]7489 examples [00:09, 779.50 examples/s]7573 examples [00:09, 794.09 examples/s]7654 examples [00:09, 798.70 examples/s]7736 examples [00:09, 800.31 examples/s]7817 examples [00:09, 779.93 examples/s]7896 examples [00:09, 780.91 examples/s]7980 examples [00:09, 795.26 examples/s]8065 examples [00:09, 808.56 examples/s]8147 examples [00:10, 786.35 examples/s]8226 examples [00:10, 777.03 examples/s]8312 examples [00:10, 799.20 examples/s]8393 examples [00:10, 796.83 examples/s]8473 examples [00:10, 795.28 examples/s]8553 examples [00:10, 795.99 examples/s]8641 examples [00:10, 817.36 examples/s]8730 examples [00:10, 835.06 examples/s]8814 examples [00:10, 834.09 examples/s]8902 examples [00:11, 845.99 examples/s]8992 examples [00:11, 859.98 examples/s]9080 examples [00:11, 865.70 examples/s]9169 examples [00:11, 870.13 examples/s]9257 examples [00:11, 863.36 examples/s]9344 examples [00:11, 859.50 examples/s]9431 examples [00:11, 849.31 examples/s]9516 examples [00:11, 832.02 examples/s]9600 examples [00:11, 833.26 examples/s]9685 examples [00:11, 836.57 examples/s]9769 examples [00:12, 834.73 examples/s]9856 examples [00:12, 843.20 examples/s]9941 examples [00:12, 837.51 examples/s]10025 examples [00:12, 798.83 examples/s]10114 examples [00:12, 822.69 examples/s]10204 examples [00:12, 842.87 examples/s]10291 examples [00:12, 848.53 examples/s]10377 examples [00:12, 847.24 examples/s]10462 examples [00:12, 845.40 examples/s]10547 examples [00:12, 837.86 examples/s]10637 examples [00:13, 854.73 examples/s]10727 examples [00:13, 865.31 examples/s]10814 examples [00:13, 864.43 examples/s]10901 examples [00:13, 854.79 examples/s]10987 examples [00:13, 850.56 examples/s]11073 examples [00:13, 810.52 examples/s]11157 examples [00:13, 818.50 examples/s]11240 examples [00:13, 794.84 examples/s]11320 examples [00:13, 785.32 examples/s]11399 examples [00:14, 767.98 examples/s]11481 examples [00:14, 782.16 examples/s]11567 examples [00:14, 803.58 examples/s]11649 examples [00:14, 804.76 examples/s]11730 examples [00:14, 777.85 examples/s]11809 examples [00:14, 766.00 examples/s]11893 examples [00:14, 786.16 examples/s]11975 examples [00:14, 795.83 examples/s]12061 examples [00:14, 812.94 examples/s]12145 examples [00:14, 819.65 examples/s]12228 examples [00:15, 819.87 examples/s]12311 examples [00:15, 817.57 examples/s]12393 examples [00:15, 816.01 examples/s]12475 examples [00:15, 789.46 examples/s]12560 examples [00:15, 804.09 examples/s]12641 examples [00:15, 795.91 examples/s]12723 examples [00:15, 802.94 examples/s]12810 examples [00:15, 820.63 examples/s]12893 examples [00:15, 820.79 examples/s]12976 examples [00:15, 821.20 examples/s]13061 examples [00:16, 826.92 examples/s]13150 examples [00:16, 843.02 examples/s]13235 examples [00:16, 834.00 examples/s]13322 examples [00:16, 841.74 examples/s]13407 examples [00:16, 839.89 examples/s]13492 examples [00:16, 821.85 examples/s]13575 examples [00:16, 819.25 examples/s]13659 examples [00:16, 822.72 examples/s]13743 examples [00:16, 826.73 examples/s]13826 examples [00:16, 827.13 examples/s]13912 examples [00:17, 834.84 examples/s]13998 examples [00:17, 841.48 examples/s]14086 examples [00:17, 851.02 examples/s]14172 examples [00:17, 852.52 examples/s]14258 examples [00:17, 850.15 examples/s]14346 examples [00:17, 856.91 examples/s]14438 examples [00:17, 874.51 examples/s]14526 examples [00:17, 875.22 examples/s]14614 examples [00:17, 871.90 examples/s]14702 examples [00:17, 864.52 examples/s]14792 examples [00:18, 873.86 examples/s]14880 examples [00:18, 869.28 examples/s]14967 examples [00:18, 863.80 examples/s]15054 examples [00:18, 864.62 examples/s]15141 examples [00:18, 854.25 examples/s]15227 examples [00:18, 851.29 examples/s]15314 examples [00:18, 855.54 examples/s]15400 examples [00:18, 843.19 examples/s]15485 examples [00:18, 830.25 examples/s]15569 examples [00:19, 817.20 examples/s]15651 examples [00:19, 799.95 examples/s]15732 examples [00:19, 802.09 examples/s]15819 examples [00:19, 820.65 examples/s]15906 examples [00:19, 834.59 examples/s]15990 examples [00:19, 835.61 examples/s]16079 examples [00:19, 847.88 examples/s]16170 examples [00:19, 863.70 examples/s]16257 examples [00:19, 864.07 examples/s]16344 examples [00:19, 853.76 examples/s]16430 examples [00:20, 840.89 examples/s]16515 examples [00:20, 820.15 examples/s]16601 examples [00:20, 831.19 examples/s]16690 examples [00:20, 846.75 examples/s]16775 examples [00:20, 824.32 examples/s]16858 examples [00:20, 798.42 examples/s]16941 examples [00:20, 804.73 examples/s]17026 examples [00:20, 816.53 examples/s]17109 examples [00:20, 820.31 examples/s]17192 examples [00:20, 812.86 examples/s]17281 examples [00:21, 832.79 examples/s]17365 examples [00:21, 825.99 examples/s]17454 examples [00:21, 842.03 examples/s]17539 examples [00:21, 833.65 examples/s]17623 examples [00:21, 827.19 examples/s]17707 examples [00:21, 829.56 examples/s]17798 examples [00:21, 849.90 examples/s]17887 examples [00:21, 859.94 examples/s]17977 examples [00:21, 870.51 examples/s]18069 examples [00:21, 883.80 examples/s]18158 examples [00:22, 885.33 examples/s]18248 examples [00:22, 887.47 examples/s]18341 examples [00:22, 897.15 examples/s]18431 examples [00:22, 886.37 examples/s]18520 examples [00:22, 885.12 examples/s]18609 examples [00:22, 876.63 examples/s]18700 examples [00:22, 884.45 examples/s]18793 examples [00:22, 897.49 examples/s]18883 examples [00:22, 895.13 examples/s]18973 examples [00:23, 892.56 examples/s]19064 examples [00:23, 896.23 examples/s]19156 examples [00:23, 901.63 examples/s]19247 examples [00:23, 899.77 examples/s]19338 examples [00:23, 890.98 examples/s]19428 examples [00:23, 869.62 examples/s]19516 examples [00:23, 856.11 examples/s]19605 examples [00:23, 864.70 examples/s]19692 examples [00:23, 847.67 examples/s]19777 examples [00:23, 833.93 examples/s]19865 examples [00:24, 847.20 examples/s]19953 examples [00:24, 855.60 examples/s]20039 examples [00:24, 787.43 examples/s]20128 examples [00:24, 813.85 examples/s]20211 examples [00:24, 747.40 examples/s]20297 examples [00:24, 776.18 examples/s]20382 examples [00:24, 794.76 examples/s]20467 examples [00:24, 809.99 examples/s]20551 examples [00:24, 816.55 examples/s]20638 examples [00:25, 829.49 examples/s]20726 examples [00:25, 843.42 examples/s]20812 examples [00:25, 848.16 examples/s]20898 examples [00:25, 842.47 examples/s]20983 examples [00:25, 840.78 examples/s]21068 examples [00:25, 840.48 examples/s]21153 examples [00:25, 843.29 examples/s]21243 examples [00:25, 857.42 examples/s]21332 examples [00:25, 866.91 examples/s]21419 examples [00:25, 862.70 examples/s]21507 examples [00:26, 866.53 examples/s]21594 examples [00:26, 856.94 examples/s]21680 examples [00:26, 832.33 examples/s]21767 examples [00:26, 841.31 examples/s]21852 examples [00:26, 790.98 examples/s]21939 examples [00:26, 811.51 examples/s]22023 examples [00:26, 816.94 examples/s]22112 examples [00:26, 837.28 examples/s]22200 examples [00:26, 848.60 examples/s]22286 examples [00:26, 846.92 examples/s]22371 examples [00:27, 842.46 examples/s]22461 examples [00:27, 856.76 examples/s]22551 examples [00:27, 867.44 examples/s]22642 examples [00:27, 878.31 examples/s]22730 examples [00:27, 874.96 examples/s]22818 examples [00:27, 867.80 examples/s]22905 examples [00:27, 833.12 examples/s]22989 examples [00:27, 809.40 examples/s]23071 examples [00:27, 786.48 examples/s]23152 examples [00:28, 792.89 examples/s]23241 examples [00:28, 817.91 examples/s]23325 examples [00:28, 822.99 examples/s]23409 examples [00:28, 825.35 examples/s]23496 examples [00:28, 835.94 examples/s]23582 examples [00:28, 840.81 examples/s]23668 examples [00:28, 843.87 examples/s]23753 examples [00:28, 836.35 examples/s]23837 examples [00:28, 835.45 examples/s]23921 examples [00:28, 834.17 examples/s]24009 examples [00:29, 845.64 examples/s]24098 examples [00:29, 857.71 examples/s]24185 examples [00:29, 859.20 examples/s]24271 examples [00:29, 856.71 examples/s]24359 examples [00:29, 861.35 examples/s]24446 examples [00:29, 835.32 examples/s]24530 examples [00:29, 829.33 examples/s]24615 examples [00:29, 831.17 examples/s]24699 examples [00:29, 830.97 examples/s]24788 examples [00:29, 847.51 examples/s]24876 examples [00:30, 856.74 examples/s]24966 examples [00:30, 868.74 examples/s]25053 examples [00:30, 864.53 examples/s]25145 examples [00:30, 877.98 examples/s]25233 examples [00:30, 871.34 examples/s]25321 examples [00:30, 867.38 examples/s]25408 examples [00:30, 843.90 examples/s]25493 examples [00:30, 843.86 examples/s]25580 examples [00:30, 848.95 examples/s]25665 examples [00:30, 835.66 examples/s]25749 examples [00:31, 816.26 examples/s]25831 examples [00:31, 815.53 examples/s]25915 examples [00:31, 820.25 examples/s]26000 examples [00:31, 828.86 examples/s]26086 examples [00:31, 835.90 examples/s]26170 examples [00:31, 825.05 examples/s]26255 examples [00:31, 832.14 examples/s]26342 examples [00:31, 843.10 examples/s]26430 examples [00:31, 853.79 examples/s]26519 examples [00:31, 862.17 examples/s]26606 examples [00:32, 847.73 examples/s]26691 examples [00:32, 845.72 examples/s]26783 examples [00:32, 866.05 examples/s]26870 examples [00:32, 854.45 examples/s]26956 examples [00:32, 841.83 examples/s]27042 examples [00:32, 846.29 examples/s]27128 examples [00:32, 848.51 examples/s]27219 examples [00:32, 864.52 examples/s]27308 examples [00:32, 868.99 examples/s]27395 examples [00:33, 867.44 examples/s]27482 examples [00:33, 857.62 examples/s]27568 examples [00:33, 855.40 examples/s]27654 examples [00:33, 816.09 examples/s]27741 examples [00:33, 830.09 examples/s]27828 examples [00:33, 839.19 examples/s]27917 examples [00:33, 851.63 examples/s]28008 examples [00:33, 868.14 examples/s]28097 examples [00:33, 874.36 examples/s]28185 examples [00:33, 871.12 examples/s]28273 examples [00:34, 856.29 examples/s]28359 examples [00:34, 829.41 examples/s]28444 examples [00:34, 815.35 examples/s]28531 examples [00:34, 828.52 examples/s]28617 examples [00:34, 835.07 examples/s]28701 examples [00:34, 833.24 examples/s]28785 examples [00:34, 817.34 examples/s]28873 examples [00:34, 834.62 examples/s]28958 examples [00:34, 835.97 examples/s]29042 examples [00:34, 827.97 examples/s]29125 examples [00:35, 795.88 examples/s]29205 examples [00:35, 782.38 examples/s]29289 examples [00:35, 797.52 examples/s]29381 examples [00:35, 829.65 examples/s]29466 examples [00:35, 834.31 examples/s]29552 examples [00:35, 841.63 examples/s]29640 examples [00:35, 850.90 examples/s]29730 examples [00:35, 862.92 examples/s]29817 examples [00:35, 853.30 examples/s]29904 examples [00:35, 855.48 examples/s]29990 examples [00:36, 846.69 examples/s]30075 examples [00:36, 787.41 examples/s]30163 examples [00:36, 811.80 examples/s]30249 examples [00:36, 823.31 examples/s]30337 examples [00:36, 836.51 examples/s]30422 examples [00:36, 837.78 examples/s]30510 examples [00:36, 848.83 examples/s]30598 examples [00:36, 856.31 examples/s]30689 examples [00:36, 870.52 examples/s]30777 examples [00:37, 870.09 examples/s]30865 examples [00:37, 854.45 examples/s]30951 examples [00:37, 854.85 examples/s]31039 examples [00:37, 859.55 examples/s]31126 examples [00:37, 840.37 examples/s]31211 examples [00:37, 822.95 examples/s]31294 examples [00:37, 815.49 examples/s]31376 examples [00:37, 797.07 examples/s]31463 examples [00:37, 815.94 examples/s]31552 examples [00:37, 835.39 examples/s]31639 examples [00:38, 845.14 examples/s]31726 examples [00:38, 851.62 examples/s]31815 examples [00:38, 861.89 examples/s]31902 examples [00:38, 864.27 examples/s]31989 examples [00:38, 850.09 examples/s]32075 examples [00:38, 813.46 examples/s]32157 examples [00:38, 815.16 examples/s]32244 examples [00:38, 830.39 examples/s]32335 examples [00:38, 851.42 examples/s]32424 examples [00:39, 862.36 examples/s]32516 examples [00:39, 877.80 examples/s]32605 examples [00:39, 853.66 examples/s]32693 examples [00:39, 860.83 examples/s]32780 examples [00:39, 853.54 examples/s]32870 examples [00:39, 865.05 examples/s]32957 examples [00:39, 866.33 examples/s]33044 examples [00:39, 834.78 examples/s]33135 examples [00:39, 854.17 examples/s]33226 examples [00:39, 868.15 examples/s]33314 examples [00:40, 840.28 examples/s]33399 examples [00:40, 835.63 examples/s]33484 examples [00:40, 839.65 examples/s]33569 examples [00:40, 827.89 examples/s]33652 examples [00:40, 808.13 examples/s]33734 examples [00:40, 800.79 examples/s]33823 examples [00:40, 825.41 examples/s]33906 examples [00:40, 815.00 examples/s]33990 examples [00:40, 821.87 examples/s]34073 examples [00:40, 823.46 examples/s]34159 examples [00:41, 831.30 examples/s]34246 examples [00:41, 841.74 examples/s]34331 examples [00:41, 843.64 examples/s]34419 examples [00:41, 853.65 examples/s]34507 examples [00:41, 860.00 examples/s]34598 examples [00:41, 873.46 examples/s]34686 examples [00:41, 864.08 examples/s]34773 examples [00:41, 863.45 examples/s]34862 examples [00:41, 870.25 examples/s]34950 examples [00:41, 861.78 examples/s]35040 examples [00:42, 870.01 examples/s]35128 examples [00:42, 868.88 examples/s]35215 examples [00:42, 850.03 examples/s]35303 examples [00:42, 857.22 examples/s]35391 examples [00:42, 861.89 examples/s]35478 examples [00:42, 862.38 examples/s]35568 examples [00:42, 872.26 examples/s]35656 examples [00:42, 867.81 examples/s]35746 examples [00:42, 875.93 examples/s]35834 examples [00:43, 867.85 examples/s]35921 examples [00:43, 858.10 examples/s]36011 examples [00:43, 870.21 examples/s]36099 examples [00:43, 856.51 examples/s]36185 examples [00:43, 834.39 examples/s]36272 examples [00:43, 843.36 examples/s]36361 examples [00:43, 856.80 examples/s]36452 examples [00:43, 870.15 examples/s]36540 examples [00:43, 862.58 examples/s]36627 examples [00:43, 862.84 examples/s]36714 examples [00:44, 839.40 examples/s]36799 examples [00:44, 808.59 examples/s]36886 examples [00:44, 824.33 examples/s]36970 examples [00:44, 827.14 examples/s]37059 examples [00:44, 842.90 examples/s]37149 examples [00:44, 859.05 examples/s]37240 examples [00:44, 870.34 examples/s]37328 examples [00:44, 859.46 examples/s]37415 examples [00:44, 861.38 examples/s]37502 examples [00:44, 857.12 examples/s]37596 examples [00:45, 877.92 examples/s]37684 examples [00:45, 876.59 examples/s]37774 examples [00:45, 880.51 examples/s]37864 examples [00:45, 885.01 examples/s]37953 examples [00:45, 884.75 examples/s]38042 examples [00:45, 860.55 examples/s]38129 examples [00:45, 828.92 examples/s]38213 examples [00:45, 809.67 examples/s]38301 examples [00:45, 829.49 examples/s]38388 examples [00:46, 839.60 examples/s]38478 examples [00:46, 854.93 examples/s]38564 examples [00:46, 852.18 examples/s]38650 examples [00:46, 853.53 examples/s]38739 examples [00:46, 862.06 examples/s]38831 examples [00:46, 877.89 examples/s]38925 examples [00:46, 893.76 examples/s]39019 examples [00:46, 904.61 examples/s]39110 examples [00:46, 893.98 examples/s]39200 examples [00:46, 891.98 examples/s]39291 examples [00:47, 895.71 examples/s]39381 examples [00:47, 895.98 examples/s]39471 examples [00:47, 883.40 examples/s]39560 examples [00:47, 883.11 examples/s]39650 examples [00:47, 887.49 examples/s]39739 examples [00:47, 870.61 examples/s]39827 examples [00:47, 871.33 examples/s]39915 examples [00:47, 865.82 examples/s]40002 examples [00:47, 810.39 examples/s]40092 examples [00:47, 834.79 examples/s]40177 examples [00:48, 837.50 examples/s]40264 examples [00:48, 840.32 examples/s]40350 examples [00:48, 843.75 examples/s]40435 examples [00:48, 833.89 examples/s]40519 examples [00:48, 834.95 examples/s]40605 examples [00:48, 841.42 examples/s]40694 examples [00:48, 854.15 examples/s]40780 examples [00:48, 852.57 examples/s]40866 examples [00:48, 853.64 examples/s]40955 examples [00:48, 863.26 examples/s]41042 examples [00:49, 844.24 examples/s]41127 examples [00:49, 832.47 examples/s]41211 examples [00:49, 810.16 examples/s]41293 examples [00:49, 795.39 examples/s]41380 examples [00:49, 816.06 examples/s]41467 examples [00:49, 830.92 examples/s]41551 examples [00:49, 818.60 examples/s]41640 examples [00:49, 837.36 examples/s]41725 examples [00:49, 839.11 examples/s]41811 examples [00:50, 843.11 examples/s]41899 examples [00:50, 852.11 examples/s]41986 examples [00:50, 855.58 examples/s]42074 examples [00:50, 859.96 examples/s]42161 examples [00:50, 858.46 examples/s]42250 examples [00:50, 866.88 examples/s]42340 examples [00:50, 875.42 examples/s]42428 examples [00:50, 870.29 examples/s]42516 examples [00:50, 845.57 examples/s]42601 examples [00:50, 810.10 examples/s]42683 examples [00:51, 808.89 examples/s]42765 examples [00:51, 811.46 examples/s]42851 examples [00:51, 824.21 examples/s]42936 examples [00:51, 830.89 examples/s]43020 examples [00:51, 795.95 examples/s]43102 examples [00:51, 801.41 examples/s]43186 examples [00:51, 810.46 examples/s]43268 examples [00:51, 795.44 examples/s]43357 examples [00:51, 819.58 examples/s]43446 examples [00:51, 838.13 examples/s]43533 examples [00:52, 847.21 examples/s]43622 examples [00:52, 857.64 examples/s]43712 examples [00:52, 868.51 examples/s]43800 examples [00:52, 869.30 examples/s]43888 examples [00:52, 864.40 examples/s]43976 examples [00:52, 867.36 examples/s]44063 examples [00:52, 839.92 examples/s]44149 examples [00:52, 844.11 examples/s]44234 examples [00:52, 842.40 examples/s]44319 examples [00:52, 834.60 examples/s]44406 examples [00:53, 842.50 examples/s]44494 examples [00:53, 853.32 examples/s]44580 examples [00:53, 852.23 examples/s]44666 examples [00:53, 830.86 examples/s]44750 examples [00:53, 832.20 examples/s]44834 examples [00:53, 822.30 examples/s]44917 examples [00:53, 821.20 examples/s]45007 examples [00:53, 843.29 examples/s]45092 examples [00:53, 842.34 examples/s]45180 examples [00:54, 853.25 examples/s]45266 examples [00:54, 845.16 examples/s]45353 examples [00:54, 851.96 examples/s]45442 examples [00:54, 861.95 examples/s]45529 examples [00:54, 851.43 examples/s]45620 examples [00:54, 867.62 examples/s]45707 examples [00:54, 859.17 examples/s]45795 examples [00:54, 864.38 examples/s]45882 examples [00:54, 862.75 examples/s]45969 examples [00:54, 849.18 examples/s]46058 examples [00:55, 860.31 examples/s]46146 examples [00:55, 863.74 examples/s]46233 examples [00:55, 847.86 examples/s]46319 examples [00:55, 848.42 examples/s]46407 examples [00:55, 857.61 examples/s]46493 examples [00:55, 847.72 examples/s]46580 examples [00:55, 851.65 examples/s]46666 examples [00:55, 835.40 examples/s]46754 examples [00:55, 846.40 examples/s]46839 examples [00:55, 841.40 examples/s]46926 examples [00:56, 848.01 examples/s]47015 examples [00:56, 857.55 examples/s]47103 examples [00:56, 863.23 examples/s]47190 examples [00:56, 861.94 examples/s]47277 examples [00:56, 857.41 examples/s]47363 examples [00:56, 855.49 examples/s]47449 examples [00:56, 838.62 examples/s]47536 examples [00:56, 845.32 examples/s]47621 examples [00:56, 840.40 examples/s]47710 examples [00:56, 853.93 examples/s]47796 examples [00:57, 852.27 examples/s]47882 examples [00:57, 840.66 examples/s]47974 examples [00:57, 861.66 examples/s]48061 examples [00:57, 858.23 examples/s]48147 examples [00:57, 839.76 examples/s]48232 examples [00:57, 835.16 examples/s]48316 examples [00:57, 825.07 examples/s]48405 examples [00:57, 842.66 examples/s]48492 examples [00:57, 844.38 examples/s]48577 examples [00:58, 836.61 examples/s]48661 examples [00:58, 834.39 examples/s]48746 examples [00:58, 836.77 examples/s]48830 examples [00:58, 833.66 examples/s]48914 examples [00:58, 817.42 examples/s]48996 examples [00:58, 816.71 examples/s]49078 examples [00:58, 811.18 examples/s]49166 examples [00:58, 829.51 examples/s]49250 examples [00:58, 827.77 examples/s]49335 examples [00:58, 832.18 examples/s]49419 examples [00:59, 821.11 examples/s]49504 examples [00:59, 828.27 examples/s]49589 examples [00:59, 834.31 examples/s]49673 examples [00:59, 818.00 examples/s]49761 examples [00:59, 835.28 examples/s]49845 examples [00:59, 827.92 examples/s]49928 examples [00:59, 819.94 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5406/50000 [00:00<00:00, 54057.60 examples/s] 33%|███▎      | 16253/50000 [00:00<00:00, 63633.68 examples/s] 53%|█████▎    | 26356/50000 [00:00<00:00, 71581.91 examples/s] 75%|███████▌  | 37749/50000 [00:00<00:00, 80565.20 examples/s] 98%|█████████▊| 48993/50000 [00:00<00:00, 88051.76 examples/s]                                                               0 examples [00:00, ? examples/s]69 examples [00:00, 689.06 examples/s]154 examples [00:00, 729.28 examples/s]237 examples [00:00, 755.26 examples/s]324 examples [00:00, 786.21 examples/s]413 examples [00:00, 813.14 examples/s]500 examples [00:00, 826.49 examples/s]590 examples [00:00, 846.97 examples/s]671 examples [00:00, 834.23 examples/s]760 examples [00:00, 849.72 examples/s]851 examples [00:01, 866.55 examples/s]943 examples [00:01, 881.71 examples/s]1033 examples [00:01, 885.94 examples/s]1121 examples [00:01, 865.64 examples/s]1211 examples [00:01, 874.49 examples/s]1304 examples [00:01, 887.98 examples/s]1393 examples [00:01, 874.19 examples/s]1485 examples [00:01, 885.86 examples/s]1574 examples [00:01, 878.35 examples/s]1666 examples [00:01, 889.28 examples/s]1759 examples [00:02, 893.16 examples/s]1851 examples [00:02, 899.53 examples/s]1944 examples [00:02, 906.68 examples/s]2035 examples [00:02, 890.77 examples/s]2125 examples [00:02, 868.60 examples/s]2213 examples [00:02, 855.84 examples/s]2299 examples [00:02, 842.04 examples/s]2384 examples [00:02, 843.25 examples/s]2471 examples [00:02, 848.93 examples/s]2556 examples [00:02, 848.18 examples/s]2648 examples [00:03, 866.41 examples/s]2738 examples [00:03, 873.68 examples/s]2829 examples [00:03, 881.67 examples/s]2918 examples [00:03, 883.29 examples/s]3007 examples [00:03, 859.11 examples/s]3094 examples [00:03, 856.89 examples/s]3188 examples [00:03, 878.16 examples/s]3283 examples [00:03, 896.58 examples/s]3373 examples [00:03, 861.16 examples/s]3460 examples [00:04, 810.13 examples/s]3551 examples [00:04, 835.87 examples/s]3645 examples [00:04, 864.35 examples/s]3733 examples [00:04, 858.25 examples/s]3825 examples [00:04, 873.39 examples/s]3913 examples [00:04, 866.06 examples/s]4003 examples [00:04, 875.75 examples/s]4092 examples [00:04, 878.44 examples/s]4181 examples [00:04, 860.43 examples/s]4270 examples [00:04, 867.36 examples/s]4360 examples [00:05, 876.55 examples/s]4448 examples [00:05, 863.95 examples/s]4543 examples [00:05, 885.38 examples/s]4632 examples [00:05, 879.36 examples/s]4721 examples [00:05, 879.64 examples/s]4810 examples [00:05, 846.89 examples/s]4897 examples [00:05, 852.59 examples/s]4984 examples [00:05, 856.57 examples/s]5072 examples [00:05, 862.87 examples/s]5160 examples [00:05, 867.58 examples/s]5247 examples [00:06, 867.15 examples/s]5335 examples [00:06, 869.83 examples/s]5423 examples [00:06, 871.84 examples/s]5511 examples [00:06, 789.75 examples/s]5599 examples [00:06, 813.93 examples/s]5690 examples [00:06, 840.50 examples/s]5780 examples [00:06, 855.47 examples/s]5867 examples [00:06, 855.91 examples/s]5955 examples [00:06, 862.80 examples/s]6043 examples [00:06, 866.60 examples/s]6136 examples [00:07, 883.12 examples/s]6231 examples [00:07, 901.24 examples/s]6324 examples [00:07, 907.92 examples/s]6416 examples [00:07, 910.55 examples/s]6509 examples [00:07, 914.37 examples/s]6601 examples [00:07, 915.76 examples/s]6693 examples [00:07, 909.93 examples/s]6785 examples [00:07, 912.48 examples/s]6879 examples [00:07, 918.26 examples/s]6971 examples [00:08, 909.79 examples/s]7063 examples [00:08, 900.28 examples/s]7154 examples [00:08, 890.09 examples/s]7244 examples [00:08, 890.29 examples/s]7334 examples [00:08, 888.78 examples/s]7428 examples [00:08, 901.28 examples/s]7519 examples [00:08, 891.95 examples/s]7609 examples [00:08, 893.04 examples/s]7699 examples [00:08, 891.84 examples/s]7789 examples [00:08, 883.16 examples/s]7878 examples [00:09, 875.72 examples/s]7966 examples [00:09, 873.84 examples/s]8055 examples [00:09, 877.46 examples/s]8146 examples [00:09, 885.10 examples/s]8235 examples [00:09, 884.52 examples/s]8324 examples [00:09, 884.44 examples/s]8418 examples [00:09, 898.03 examples/s]8510 examples [00:09, 902.87 examples/s]8601 examples [00:09, 860.76 examples/s]8688 examples [00:09, 860.11 examples/s]8775 examples [00:10, 860.44 examples/s]8862 examples [00:10, 860.93 examples/s]8952 examples [00:10, 870.38 examples/s]9041 examples [00:10, 875.35 examples/s]9129 examples [00:10, 870.69 examples/s]9217 examples [00:10, 861.55 examples/s]9304 examples [00:10, 852.91 examples/s]9392 examples [00:10, 860.54 examples/s]9479 examples [00:10, 846.15 examples/s]9569 examples [00:10, 857.74 examples/s]9658 examples [00:11, 865.65 examples/s]9747 examples [00:11, 872.29 examples/s]9835 examples [00:11, 872.28 examples/s]9923 examples [00:11, 868.48 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM2YN3G/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM2YN3G/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff6a34cea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff6a34cea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff6a34cea60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff6eeeb50b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff629837438>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff6a34cea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff6a34cea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff6a34cea60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff6a169a080>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff6a169a048>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.48 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.24 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.30 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.23 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.23 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.73 file/s]2020-08-05 12:11:24.971916: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-05 12:11:24.976188: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-05 12:11:24.976401: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5565edab4720 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-05 12:11:24.976420: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 156543.53it/s] 72%|███████▏  | 7143424/9912422 [00:00<00:12, 223423.23it/s]9920512it [00:00, 42885751.59it/s]                           
0it [00:00, ?it/s]32768it [00:00, 421192.65it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161299.32it/s]1654784it [00:00, 11786876.98it/s]                         
0it [00:00, ?it/s]8192it [00:00, 242734.09it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
