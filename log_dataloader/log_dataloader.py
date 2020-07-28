
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fdd972c6b70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fdd972c6b70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fde01f36510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fde01f36510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fde21163ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fde21163ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fddaf2dea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fddaf2dea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fddaf2dea60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 158660.05it/s] 51%|█████▏    | 5087232/9912422 [00:00<00:21, 226353.67it/s]9920512it [00:00, 34802893.29it/s]                           
0it [00:00, ?it/s]32768it [00:00, 632095.04it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 142040.79it/s]1654784it [00:00, 10193267.44it/s]                         
0it [00:00, ?it/s]8192it [00:00, 233684.08it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd971836d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd97190940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fddaf2de6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fddaf2de6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fddaf2de6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:30,  1.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:30,  1.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:29,  1.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:29,  1.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:28,  1.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:28,  1.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:27,  1.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:26,  1.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:26,  1.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:00,  2.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:00,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:00,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:59,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:59,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:59,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:58,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:58,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:57,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:57,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:57,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:56,  2.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:39,  3.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:39,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:39,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:39,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:38,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:38,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:38,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:38,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:37,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:37,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:37,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:26,  5.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:26,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:26,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:25,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:25,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:25,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:25,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:25,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:24,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:24,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:24,  5.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:00<00:17,  7.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:17,  7.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:17,  7.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:17,  7.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:16,  7.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:16,  7.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:16,  7.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:16,  7.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:16,  7.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:11,  9.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11,  9.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11,  9.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10,  9.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 12.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 12.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 12.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 16.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 16.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 16.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 21.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 21.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 21.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 26.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 36.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 36.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 36.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 36.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 36.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 36.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 36.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 36.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 41.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 41.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 41.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 44.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 44.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 44.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 44.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 44.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 44.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 44.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 44.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 48.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 48.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 48.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 48.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 48.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 48.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 48.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 51.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 51.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 51.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 51.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 51.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 51.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 51.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 54.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 56.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 56.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 56.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 56.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 56.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 56.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 56.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 56.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 56.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 56.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 56.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 56.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 56.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 56.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 56.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 56.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 56.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 56.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 56.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 56.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 56.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 56.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 58.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 58.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 58.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 58.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 58.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 58.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 58.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.02s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 58.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 58.58 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.92s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  3.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 58.58 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.92s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.92s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.93 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.92s/ url]
0 examples [00:00, ? examples/s]2020-07-28 06:08:35.181173: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-28 06:08:35.431835: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-28 06:08:35.433066: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b31900ed90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-28 06:08:35.433104: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.52 examples/s]116 examples [00:00,  2.16 examples/s]234 examples [00:00,  3.09 examples/s]346 examples [00:00,  4.41 examples/s]466 examples [00:01,  6.29 examples/s]586 examples [00:01,  8.96 examples/s]712 examples [00:01, 12.76 examples/s]829 examples [00:01, 18.15 examples/s]948 examples [00:01, 25.75 examples/s]1066 examples [00:01, 36.45 examples/s]1180 examples [00:01, 51.36 examples/s]1296 examples [00:01, 72.00 examples/s]1415 examples [00:01, 100.24 examples/s]1532 examples [00:01, 138.11 examples/s]1655 examples [00:02, 188.22 examples/s]1773 examples [00:02, 250.84 examples/s]1890 examples [00:02, 328.10 examples/s]2014 examples [00:02, 420.66 examples/s]2133 examples [00:02, 517.75 examples/s]2255 examples [00:02, 625.69 examples/s]2376 examples [00:02, 731.57 examples/s]2495 examples [00:02, 825.04 examples/s]2614 examples [00:02, 897.01 examples/s]2732 examples [00:02, 965.78 examples/s]2853 examples [00:03, 1027.64 examples/s]2972 examples [00:03, 1065.64 examples/s]3090 examples [00:03, 1050.00 examples/s]3209 examples [00:03, 1085.80 examples/s]3324 examples [00:03, 1099.73 examples/s]3443 examples [00:03, 1122.96 examples/s]3561 examples [00:03, 1139.00 examples/s]3681 examples [00:03, 1153.92 examples/s]3799 examples [00:03, 1159.99 examples/s]3917 examples [00:04, 1152.01 examples/s]4038 examples [00:04, 1168.34 examples/s]4160 examples [00:04, 1181.51 examples/s]4279 examples [00:04, 1157.66 examples/s]4397 examples [00:04, 1163.92 examples/s]4514 examples [00:04, 1158.65 examples/s]4631 examples [00:04, 1073.31 examples/s]4745 examples [00:04, 1088.75 examples/s]4859 examples [00:04, 1101.78 examples/s]4970 examples [00:04, 1091.92 examples/s]5086 examples [00:05, 1110.46 examples/s]5206 examples [00:05, 1134.79 examples/s]5325 examples [00:05, 1148.71 examples/s]5446 examples [00:05, 1164.02 examples/s]5563 examples [00:05, 1154.16 examples/s]5679 examples [00:05, 1126.73 examples/s]5800 examples [00:05, 1148.82 examples/s]5916 examples [00:05, 1111.02 examples/s]6029 examples [00:05, 1114.45 examples/s]6148 examples [00:05, 1134.01 examples/s]6270 examples [00:06, 1157.28 examples/s]6387 examples [00:06, 1157.70 examples/s]6504 examples [00:06, 1136.97 examples/s]6621 examples [00:06, 1146.31 examples/s]6738 examples [00:06, 1151.96 examples/s]6854 examples [00:06, 1131.24 examples/s]6971 examples [00:06, 1141.91 examples/s]7088 examples [00:06, 1149.86 examples/s]7204 examples [00:06, 1137.01 examples/s]7318 examples [00:07, 1125.50 examples/s]7431 examples [00:07, 1116.81 examples/s]7543 examples [00:07, 1107.33 examples/s]7657 examples [00:07, 1115.60 examples/s]7781 examples [00:07, 1147.98 examples/s]7900 examples [00:07, 1159.52 examples/s]8018 examples [00:07, 1165.52 examples/s]8135 examples [00:07, 1147.72 examples/s]8250 examples [00:07, 1129.89 examples/s]8364 examples [00:07, 1093.76 examples/s]8474 examples [00:08, 1088.37 examples/s]8587 examples [00:08, 1092.73 examples/s]8701 examples [00:08, 1089.22 examples/s]8815 examples [00:08, 1103.22 examples/s]8930 examples [00:08, 1113.73 examples/s]9042 examples [00:08, 1078.47 examples/s]9153 examples [00:08, 1085.91 examples/s]9262 examples [00:08, 1077.15 examples/s]9370 examples [00:08, 1071.98 examples/s]9480 examples [00:08, 1078.02 examples/s]9593 examples [00:09, 1091.58 examples/s]9703 examples [00:09, 1078.34 examples/s]9816 examples [00:09, 1091.90 examples/s]9932 examples [00:09, 1111.19 examples/s]10044 examples [00:09, 1043.33 examples/s]10151 examples [00:09, 1039.95 examples/s]10262 examples [00:09, 1058.68 examples/s]10379 examples [00:09, 1088.47 examples/s]10502 examples [00:09, 1125.33 examples/s]10625 examples [00:10, 1152.99 examples/s]10741 examples [00:10, 1144.97 examples/s]10857 examples [00:10, 1147.99 examples/s]10976 examples [00:10, 1160.21 examples/s]11093 examples [00:10, 1154.57 examples/s]11209 examples [00:10, 1151.43 examples/s]11325 examples [00:10, 1140.14 examples/s]11440 examples [00:10, 1132.97 examples/s]11554 examples [00:10, 1130.33 examples/s]11673 examples [00:10, 1146.78 examples/s]11792 examples [00:11, 1158.40 examples/s]11908 examples [00:11, 1135.65 examples/s]12022 examples [00:11, 1129.23 examples/s]12137 examples [00:11, 1134.62 examples/s]12251 examples [00:11, 1117.41 examples/s]12367 examples [00:11, 1128.60 examples/s]12484 examples [00:11, 1137.99 examples/s]12605 examples [00:11, 1157.15 examples/s]12721 examples [00:11, 1150.37 examples/s]12837 examples [00:11, 1148.63 examples/s]12952 examples [00:12, 1148.24 examples/s]13071 examples [00:12, 1159.24 examples/s]13187 examples [00:12, 1157.11 examples/s]13303 examples [00:12, 1154.75 examples/s]13425 examples [00:12, 1172.24 examples/s]13546 examples [00:12, 1180.88 examples/s]13665 examples [00:12, 1182.51 examples/s]13786 examples [00:12, 1188.41 examples/s]13905 examples [00:12, 1187.77 examples/s]14027 examples [00:12, 1195.76 examples/s]14147 examples [00:13, 1192.44 examples/s]14267 examples [00:13, 1174.34 examples/s]14385 examples [00:13, 1165.56 examples/s]14502 examples [00:13, 1162.77 examples/s]14621 examples [00:13, 1170.57 examples/s]14739 examples [00:13, 1171.17 examples/s]14859 examples [00:13, 1178.36 examples/s]14980 examples [00:13, 1186.91 examples/s]15102 examples [00:13, 1196.06 examples/s]15223 examples [00:13, 1198.80 examples/s]15345 examples [00:14, 1202.48 examples/s]15466 examples [00:14, 1202.81 examples/s]15587 examples [00:14, 1186.82 examples/s]15708 examples [00:14, 1193.53 examples/s]15833 examples [00:14, 1209.34 examples/s]15955 examples [00:14, 1186.70 examples/s]16074 examples [00:14, 1170.18 examples/s]16192 examples [00:14, 1146.97 examples/s]16314 examples [00:14, 1166.35 examples/s]16431 examples [00:14, 1166.65 examples/s]16550 examples [00:15, 1173.05 examples/s]16668 examples [00:15, 1163.90 examples/s]16785 examples [00:15, 1138.39 examples/s]16903 examples [00:15, 1150.25 examples/s]17019 examples [00:15, 1142.85 examples/s]17142 examples [00:15, 1166.69 examples/s]17259 examples [00:15, 1159.30 examples/s]17376 examples [00:15, 1154.06 examples/s]17492 examples [00:15, 1149.18 examples/s]17610 examples [00:16, 1157.57 examples/s]17726 examples [00:16, 1121.16 examples/s]17840 examples [00:16, 1125.50 examples/s]17956 examples [00:16, 1134.17 examples/s]18070 examples [00:16, 1126.72 examples/s]18183 examples [00:16, 1113.27 examples/s]18295 examples [00:16, 1076.52 examples/s]18404 examples [00:16, 1075.20 examples/s]18512 examples [00:16, 1070.37 examples/s]18625 examples [00:16, 1085.04 examples/s]18738 examples [00:17, 1097.96 examples/s]18851 examples [00:17, 1104.75 examples/s]18966 examples [00:17, 1116.33 examples/s]19083 examples [00:17, 1131.44 examples/s]19204 examples [00:17, 1153.66 examples/s]19320 examples [00:17, 1154.95 examples/s]19436 examples [00:17, 1152.32 examples/s]19552 examples [00:17, 1133.38 examples/s]19667 examples [00:17, 1136.59 examples/s]19782 examples [00:17, 1138.88 examples/s]19899 examples [00:18, 1147.32 examples/s]20014 examples [00:18, 1106.34 examples/s]20133 examples [00:18, 1127.41 examples/s]20253 examples [00:18, 1147.60 examples/s]20372 examples [00:18, 1157.68 examples/s]20489 examples [00:18, 1157.65 examples/s]20608 examples [00:18, 1165.99 examples/s]20725 examples [00:18, 1150.42 examples/s]20841 examples [00:18, 1139.14 examples/s]20956 examples [00:18, 1131.52 examples/s]21070 examples [00:19, 1129.21 examples/s]21183 examples [00:19, 1027.07 examples/s]21296 examples [00:19, 1055.71 examples/s]21410 examples [00:19, 1077.99 examples/s]21525 examples [00:19, 1096.36 examples/s]21644 examples [00:19, 1122.63 examples/s]21758 examples [00:19, 1125.91 examples/s]21873 examples [00:19, 1130.55 examples/s]21987 examples [00:19, 1124.89 examples/s]22101 examples [00:20, 1128.77 examples/s]22215 examples [00:20, 1122.56 examples/s]22328 examples [00:20, 1109.66 examples/s]22440 examples [00:20, 1101.79 examples/s]22553 examples [00:20, 1109.25 examples/s]22665 examples [00:20, 1109.81 examples/s]22777 examples [00:20, 1103.85 examples/s]22888 examples [00:20, 1061.73 examples/s]22995 examples [00:20, 1044.52 examples/s]23110 examples [00:20, 1073.41 examples/s]23218 examples [00:21, 1010.12 examples/s]23321 examples [00:21, 998.07 examples/s] 23422 examples [00:21, 983.73 examples/s]23532 examples [00:21, 1014.75 examples/s]23639 examples [00:21, 1030.71 examples/s]23743 examples [00:21, 1027.50 examples/s]23849 examples [00:21, 1034.40 examples/s]23961 examples [00:21, 1057.66 examples/s]24073 examples [00:21, 1075.24 examples/s]24183 examples [00:21, 1080.81 examples/s]24300 examples [00:22, 1105.94 examples/s]24413 examples [00:22, 1112.72 examples/s]24525 examples [00:22, 1105.24 examples/s]24636 examples [00:22, 1093.35 examples/s]24750 examples [00:22, 1105.30 examples/s]24861 examples [00:22, 1106.17 examples/s]24972 examples [00:22, 1074.89 examples/s]25080 examples [00:22, 1060.12 examples/s]25187 examples [00:22, 1060.72 examples/s]25294 examples [00:23, 1054.13 examples/s]25416 examples [00:23, 1096.85 examples/s]25531 examples [00:23, 1111.84 examples/s]25643 examples [00:23, 1111.84 examples/s]25760 examples [00:23, 1127.53 examples/s]25877 examples [00:23, 1138.59 examples/s]25992 examples [00:23, 1140.31 examples/s]26110 examples [00:23, 1151.36 examples/s]26226 examples [00:23, 1152.66 examples/s]26351 examples [00:23, 1178.95 examples/s]26471 examples [00:24, 1184.88 examples/s]26594 examples [00:24, 1195.41 examples/s]26718 examples [00:24, 1205.33 examples/s]26839 examples [00:24, 1177.36 examples/s]26958 examples [00:24, 1180.98 examples/s]27079 examples [00:24, 1189.49 examples/s]27199 examples [00:24, 1165.79 examples/s]27316 examples [00:24, 1155.60 examples/s]27432 examples [00:24, 1130.96 examples/s]27546 examples [00:24, 1129.65 examples/s]27660 examples [00:25, 1072.78 examples/s]27770 examples [00:25, 1080.79 examples/s]27882 examples [00:25, 1090.97 examples/s]27995 examples [00:25, 1099.74 examples/s]28108 examples [00:25, 1106.55 examples/s]28230 examples [00:25, 1136.80 examples/s]28352 examples [00:25, 1158.81 examples/s]28469 examples [00:25, 1144.75 examples/s]28584 examples [00:25, 1120.99 examples/s]28697 examples [00:25, 1107.21 examples/s]28808 examples [00:26, 1103.44 examples/s]28919 examples [00:26, 1079.01 examples/s]29028 examples [00:26, 1047.19 examples/s]29134 examples [00:26, 1038.98 examples/s]29253 examples [00:26, 1078.81 examples/s]29371 examples [00:26, 1106.91 examples/s]29483 examples [00:26, 1110.61 examples/s]29600 examples [00:26, 1127.54 examples/s]29714 examples [00:26, 1123.31 examples/s]29828 examples [00:27, 1128.17 examples/s]29941 examples [00:27, 1087.70 examples/s]30051 examples [00:27, 1043.49 examples/s]30165 examples [00:27, 1069.69 examples/s]30282 examples [00:27, 1095.02 examples/s]30393 examples [00:27, 1015.34 examples/s]30497 examples [00:27, 973.54 examples/s] 30601 examples [00:27, 991.89 examples/s]30702 examples [00:27, 961.34 examples/s]30800 examples [00:28, 900.26 examples/s]30898 examples [00:28, 920.60 examples/s]30992 examples [00:28, 911.07 examples/s]31085 examples [00:28, 915.68 examples/s]31180 examples [00:28, 924.59 examples/s]31273 examples [00:28, 909.70 examples/s]31395 examples [00:28, 983.50 examples/s]31516 examples [00:28, 1040.17 examples/s]31630 examples [00:28, 1067.09 examples/s]31739 examples [00:28, 1068.12 examples/s]31848 examples [00:29, 1072.89 examples/s]31960 examples [00:29, 1084.90 examples/s]32070 examples [00:29, 1086.76 examples/s]32184 examples [00:29, 1098.97 examples/s]32306 examples [00:29, 1131.38 examples/s]32427 examples [00:29, 1153.14 examples/s]32544 examples [00:29, 1156.36 examples/s]32660 examples [00:29, 1155.54 examples/s]32776 examples [00:29, 1149.09 examples/s]32892 examples [00:29, 1126.25 examples/s]33005 examples [00:30, 1086.81 examples/s]33127 examples [00:30, 1123.47 examples/s]33240 examples [00:30, 1108.04 examples/s]33352 examples [00:30, 1106.25 examples/s]33469 examples [00:30, 1122.98 examples/s]33583 examples [00:30, 1126.57 examples/s]33697 examples [00:30, 1130.22 examples/s]33811 examples [00:30, 1132.00 examples/s]33925 examples [00:30, 1123.45 examples/s]34038 examples [00:30, 1124.78 examples/s]34156 examples [00:31, 1139.73 examples/s]34271 examples [00:31, 1139.41 examples/s]34386 examples [00:31, 1137.70 examples/s]34500 examples [00:31, 1105.46 examples/s]34611 examples [00:31, 1071.85 examples/s]34725 examples [00:31, 1091.17 examples/s]34840 examples [00:31, 1105.30 examples/s]34951 examples [00:31, 1095.96 examples/s]35061 examples [00:31, 1080.19 examples/s]35170 examples [00:32, 1081.12 examples/s]35281 examples [00:32, 1088.65 examples/s]35395 examples [00:32, 1101.84 examples/s]35515 examples [00:32, 1128.13 examples/s]35631 examples [00:32, 1134.72 examples/s]35745 examples [00:32, 1118.23 examples/s]35858 examples [00:32, 1103.38 examples/s]35969 examples [00:32, 1038.29 examples/s]36074 examples [00:32, 1038.49 examples/s]36181 examples [00:32, 1045.30 examples/s]36289 examples [00:33, 1054.93 examples/s]36398 examples [00:33, 1062.80 examples/s]36505 examples [00:33, 1064.49 examples/s]36620 examples [00:33, 1088.27 examples/s]36730 examples [00:33, 1089.50 examples/s]36840 examples [00:33, 1076.58 examples/s]36948 examples [00:33, 1063.99 examples/s]37058 examples [00:33, 1072.10 examples/s]37170 examples [00:33, 1083.16 examples/s]37279 examples [00:33, 1074.07 examples/s]37387 examples [00:34, 1075.07 examples/s]37499 examples [00:34, 1085.22 examples/s]37608 examples [00:34, 1069.73 examples/s]37716 examples [00:34, 1068.18 examples/s]37823 examples [00:34, 1065.28 examples/s]37931 examples [00:34, 1067.99 examples/s]38043 examples [00:34, 1080.46 examples/s]38152 examples [00:34, 1077.75 examples/s]38260 examples [00:34, 1075.92 examples/s]38368 examples [00:34, 1066.22 examples/s]38475 examples [00:35, 1055.35 examples/s]38581 examples [00:35, 1042.49 examples/s]38686 examples [00:35, 1037.61 examples/s]38792 examples [00:35, 1041.14 examples/s]38897 examples [00:35, 1029.59 examples/s]39005 examples [00:35, 1041.44 examples/s]39110 examples [00:35, 1038.50 examples/s]39214 examples [00:35, 1022.36 examples/s]39325 examples [00:35, 1046.36 examples/s]39436 examples [00:36, 1062.76 examples/s]39546 examples [00:36, 1070.90 examples/s]39654 examples [00:36, 1058.68 examples/s]39762 examples [00:36, 1064.85 examples/s]39869 examples [00:36, 1044.75 examples/s]39974 examples [00:36, 1031.33 examples/s]40078 examples [00:36, 895.89 examples/s] 40180 examples [00:36, 928.44 examples/s]40288 examples [00:36, 969.01 examples/s]40396 examples [00:36, 986.84 examples/s]40497 examples [00:37, 972.72 examples/s]40607 examples [00:37, 1005.86 examples/s]40709 examples [00:37, 1005.29 examples/s]40813 examples [00:37, 1015.21 examples/s]40920 examples [00:37, 1022.74 examples/s]41023 examples [00:37, 1023.39 examples/s]41129 examples [00:37, 1032.42 examples/s]41233 examples [00:37, 1027.31 examples/s]41338 examples [00:37, 1031.57 examples/s]41446 examples [00:38, 1043.12 examples/s]41551 examples [00:38, 1022.32 examples/s]41657 examples [00:38, 1031.43 examples/s]41761 examples [00:38, 1015.62 examples/s]41866 examples [00:38, 1023.87 examples/s]41972 examples [00:38, 1031.40 examples/s]42076 examples [00:38, 1028.13 examples/s]42179 examples [00:38, 1015.39 examples/s]42281 examples [00:38, 1006.02 examples/s]42392 examples [00:38, 1033.54 examples/s]42500 examples [00:39, 1046.78 examples/s]42605 examples [00:39, 1039.34 examples/s]42714 examples [00:39, 1053.15 examples/s]42820 examples [00:39, 1035.39 examples/s]42924 examples [00:39, 1036.59 examples/s]43032 examples [00:39, 1048.25 examples/s]43137 examples [00:39, 1031.51 examples/s]43244 examples [00:39, 1040.92 examples/s]43350 examples [00:39, 1044.05 examples/s]43464 examples [00:39, 1069.38 examples/s]43579 examples [00:40, 1090.35 examples/s]43689 examples [00:40, 1049.61 examples/s]43795 examples [00:40, 1039.35 examples/s]43900 examples [00:40, 1041.87 examples/s]44010 examples [00:40, 1056.41 examples/s]44118 examples [00:40, 1062.66 examples/s]44225 examples [00:40, 1053.10 examples/s]44334 examples [00:40, 1062.29 examples/s]44441 examples [00:40, 1050.95 examples/s]44547 examples [00:40, 1006.33 examples/s]44649 examples [00:41, 969.83 examples/s] 44752 examples [00:41, 985.26 examples/s]44852 examples [00:41, 983.69 examples/s]44951 examples [00:41, 971.64 examples/s]45057 examples [00:41, 994.65 examples/s]45158 examples [00:41, 998.98 examples/s]45265 examples [00:41, 1016.59 examples/s]45367 examples [00:41, 950.66 examples/s] 45464 examples [00:41, 940.36 examples/s]45562 examples [00:42, 951.64 examples/s]45673 examples [00:42, 993.01 examples/s]45774 examples [00:42, 989.95 examples/s]45874 examples [00:42, 989.01 examples/s]45981 examples [00:42, 1009.97 examples/s]46090 examples [00:42, 1031.79 examples/s]46194 examples [00:42, 1029.41 examples/s]46298 examples [00:42, 987.64 examples/s] 46398 examples [00:42, 980.77 examples/s]46506 examples [00:42, 1007.90 examples/s]46615 examples [00:43, 1029.44 examples/s]46719 examples [00:43, 1019.73 examples/s]46825 examples [00:43, 1029.80 examples/s]46929 examples [00:43, 1030.58 examples/s]47033 examples [00:43, 1007.81 examples/s]47135 examples [00:43, 979.58 examples/s] 47235 examples [00:43, 983.16 examples/s]47334 examples [00:43, 964.12 examples/s]47441 examples [00:43, 992.07 examples/s]47549 examples [00:44, 1014.43 examples/s]47655 examples [00:44, 1027.06 examples/s]47759 examples [00:44, 1025.25 examples/s]47862 examples [00:44, 994.78 examples/s] 47969 examples [00:44, 1014.40 examples/s]48076 examples [00:44, 1029.90 examples/s]48181 examples [00:44, 1035.27 examples/s]48286 examples [00:44, 1039.59 examples/s]48391 examples [00:44, 1030.01 examples/s]48496 examples [00:44, 1035.66 examples/s]48601 examples [00:45, 1037.22 examples/s]48711 examples [00:45, 1053.51 examples/s]48817 examples [00:45, 1043.85 examples/s]48929 examples [00:45, 1062.68 examples/s]49037 examples [00:45, 1067.24 examples/s]49146 examples [00:45, 1072.82 examples/s]49258 examples [00:45, 1083.81 examples/s]49367 examples [00:45, 1054.92 examples/s]49473 examples [00:45, 1019.14 examples/s]49578 examples [00:45, 1025.45 examples/s]49689 examples [00:46, 1048.75 examples/s]49796 examples [00:46, 1054.10 examples/s]49907 examples [00:46, 1070.00 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7853/50000 [00:00<00:00, 78527.83 examples/s] 44%|████▎     | 21793/50000 [00:00<00:00, 90365.23 examples/s] 69%|██████▉   | 34548/50000 [00:00<00:00, 99025.50 examples/s] 97%|█████████▋| 48475/50000 [00:00<00:00, 108423.45 examples/s]                                                                0 examples [00:00, ? examples/s]82 examples [00:00, 819.02 examples/s]183 examples [00:00, 868.05 examples/s]291 examples [00:00, 920.78 examples/s]400 examples [00:00, 965.69 examples/s]509 examples [00:00, 999.23 examples/s]613 examples [00:00, 1008.46 examples/s]720 examples [00:00, 1025.95 examples/s]827 examples [00:00, 1036.89 examples/s]936 examples [00:00, 1050.53 examples/s]1041 examples [00:01, 1048.18 examples/s]1150 examples [00:01, 1058.48 examples/s]1255 examples [00:01, 1039.96 examples/s]1368 examples [00:01, 1064.37 examples/s]1481 examples [00:01, 1083.17 examples/s]1590 examples [00:01, 1066.26 examples/s]1706 examples [00:01, 1091.53 examples/s]1820 examples [00:01, 1105.38 examples/s]1934 examples [00:01, 1113.39 examples/s]2046 examples [00:01, 1103.68 examples/s]2157 examples [00:02, 1100.88 examples/s]2271 examples [00:02, 1111.77 examples/s]2383 examples [00:02, 1109.00 examples/s]2494 examples [00:02, 1084.75 examples/s]2604 examples [00:02, 1086.81 examples/s]2713 examples [00:02, 1083.98 examples/s]2824 examples [00:02, 1089.61 examples/s]2934 examples [00:02, 1082.29 examples/s]3043 examples [00:02, 1084.14 examples/s]3152 examples [00:02, 1080.50 examples/s]3261 examples [00:03, 1076.86 examples/s]3374 examples [00:03, 1090.64 examples/s]3487 examples [00:03, 1102.11 examples/s]3599 examples [00:03, 1107.06 examples/s]3710 examples [00:03, 1096.55 examples/s]3820 examples [00:03, 1069.19 examples/s]3928 examples [00:03, 1051.15 examples/s]4035 examples [00:03, 1053.91 examples/s]4142 examples [00:03, 1058.08 examples/s]4248 examples [00:03, 1047.06 examples/s]4353 examples [00:04, 1029.97 examples/s]4459 examples [00:04, 1037.56 examples/s]4563 examples [00:04, 1035.72 examples/s]4671 examples [00:04, 1047.80 examples/s]4781 examples [00:04, 1060.79 examples/s]4888 examples [00:04, 1034.37 examples/s]4992 examples [00:04, 1022.34 examples/s]5101 examples [00:04, 1040.01 examples/s]5207 examples [00:04, 1044.81 examples/s]5312 examples [00:04, 1024.74 examples/s]5419 examples [00:05, 1037.13 examples/s]5530 examples [00:05, 1057.45 examples/s]5636 examples [00:05, 1043.27 examples/s]5748 examples [00:05, 1062.57 examples/s]5855 examples [00:05, 1050.73 examples/s]5961 examples [00:05, 1023.84 examples/s]6074 examples [00:05, 1052.18 examples/s]6184 examples [00:05, 1063.73 examples/s]6298 examples [00:05, 1085.04 examples/s]6411 examples [00:06, 1096.48 examples/s]6522 examples [00:06, 1098.41 examples/s]6633 examples [00:06, 1097.42 examples/s]6743 examples [00:06, 1097.09 examples/s]6853 examples [00:06, 1073.14 examples/s]6961 examples [00:06, 1057.05 examples/s]7068 examples [00:06, 1060.25 examples/s]7175 examples [00:06, 1059.92 examples/s]7282 examples [00:06, 1059.42 examples/s]7392 examples [00:06, 1070.36 examples/s]7500 examples [00:07, 1060.55 examples/s]7607 examples [00:07, 1053.13 examples/s]7713 examples [00:07, 1045.10 examples/s]7823 examples [00:07, 1059.01 examples/s]7940 examples [00:07, 1088.78 examples/s]8050 examples [00:07, 1067.79 examples/s]8159 examples [00:07, 1073.19 examples/s]8267 examples [00:07, 1068.30 examples/s]8378 examples [00:07, 1078.60 examples/s]8488 examples [00:07, 1084.15 examples/s]8597 examples [00:08, 1085.90 examples/s]8712 examples [00:08, 1102.63 examples/s]8829 examples [00:08, 1121.80 examples/s]8943 examples [00:08, 1126.83 examples/s]9056 examples [00:08, 1116.23 examples/s]9168 examples [00:08, 1088.35 examples/s]9282 examples [00:08, 1102.61 examples/s]9395 examples [00:08, 1109.75 examples/s]9507 examples [00:08, 1109.07 examples/s]9622 examples [00:08, 1118.32 examples/s]9734 examples [00:09, 1110.89 examples/s]9846 examples [00:09, 1087.72 examples/s]9965 examples [00:09, 1115.92 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePKHM2Q/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePKHM2Q/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fddaf2dea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fddaf2dea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fddaf2dea60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fddfac4bcc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd3558c400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fddaf2dea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fddaf2dea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fddaf2dea60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd971905f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd971901d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fdd27bff378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fdd27bff378> 

  function with postional parmater data_info <function split_train_valid at 0x7fdd27bff378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fdd27bff488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fdd27bff488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fdd27bff488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=fe095a8557833d51cc75b7ef53901212fbc1e855056c8ffd224016c19391e44d
  Stored in directory: /tmp/pip-ephem-wheel-cache-wu90iex_/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<16:50:13, 14.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:01:10, 19.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:28:04, 28.3kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:00<5:56:12, 40.3kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:08:45, 57.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.27M/862M [00:01<2:53:05, 82.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.9M/862M [00:01<2:00:27, 117kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.5M/862M [00:01<1:23:50, 167kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.4M/862M [00:01<58:38, 238kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.1M/862M [00:01<40:52, 340kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:01<28:37, 483kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.1M/862M [00:01<19:58, 688kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.8M/862M [00:02<14:01, 975kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<09:51, 1.38MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.6M/862M [00:02<06:58, 1.94MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<06:00, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<06:06, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<07:54, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<06:19, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:04<04:37, 2.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<07:44, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<06:54, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:06<05:08, 2.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:31, 2.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<07:15, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<05:45, 2.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<06:09, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<05:41, 2.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<04:16, 3.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<06:03, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<05:34, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<04:13, 3.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<06:03, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<06:47, 1.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:14<05:18, 2.46MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.7M/862M [00:14<03:54, 3.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<07:41, 1.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:43, 1.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:16<05:01, 2.58MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<06:33, 1.97MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<07:13, 1.79MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<05:36, 2.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:18<04:06, 3.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<08:19, 1.55MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<07:07, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:20<05:18, 2.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<06:43, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<06:00, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:22<04:31, 2.83MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<06:10, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<06:55, 1.84MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<05:23, 2.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<03:56, 3.22MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:26, 1.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:55, 1.60MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:51, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:03, 1.79MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:29, 1.68MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:47, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:14, 2.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:36, 1.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:36, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:56, 2.53MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:22, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:43, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:16, 2.91MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:54, 2.10MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:39, 1.86MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:16, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<03:49, 3.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<11:55:57, 17.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<8:22:11, 24.5kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<5:51:03, 35.0kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<4:07:54, 49.5kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<2:56:00, 69.7kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<2:03:41, 99.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<1:28:11, 138kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<1:02:56, 194kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<44:14, 275kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<33:44, 359kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<26:07, 464kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<18:50, 643kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<13:17, 908kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<11:41:05, 17.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<8:11:45, 24.5kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<5:43:47, 35.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<4:02:44, 49.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<2:52:18, 69.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<2:01:00, 99.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<1:24:37, 141kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<1:04:50, 184kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<46:34, 256kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<32:50, 362kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<25:42, 461kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<20:23, 581kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<14:46, 801kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<10:30, 1.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<11:31, 1.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<09:15, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:44, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<07:27, 1.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<07:36, 1.54MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:54, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:00, 1.94MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:25, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<04:04, 2.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:34, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:04, 2.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<03:50, 3.01MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:24, 2.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:07, 1.88MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [00:59<04:51, 2.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [00:59<03:31, 3.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<1:37:02, 118kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<1:09:03, 166kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<48:31, 235kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<36:31, 311kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<26:42, 425kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<18:54, 600kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<15:53, 711kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<12:15, 921kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<08:50, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:50, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:19, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:21, 2.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:24, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:27, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<04:05, 2.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:30, 2.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:59, 2.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:43, 2.97MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:13, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:45, 2.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:33, 3.09MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:06, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:41, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:32, 3.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:03, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:45, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:29, 2.42MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:17, 3.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:32, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:23, 1.69MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:44, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:49, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:12, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:51, 2.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:12, 2.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:49, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:31, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:18, 3.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:38, 1.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:25, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:43, 2.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:47, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:07, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:50, 2.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:09, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:44, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:27, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:16, 3.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<06:21, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:31, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:04, 2.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:17, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:45, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:34, 2.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<04:56, 2.08MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:18, 2.38MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:15, 3.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<02:25, 4.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<45:07, 226kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<32:36, 313kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<22:58, 443kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<18:27, 549kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<14:58, 677kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<10:59, 922kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<09:17, 1.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<07:31, 1.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:30, 1.82MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:11, 1.62MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:20, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:58, 2.50MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:07, 1.94MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<04:36, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:28, 2.85MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:45, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:20, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:09, 2.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:02, 3.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:59, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:54, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<04:27, 2.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<03:21, 2.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<07:30, 1.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<10:01, 971kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<08:13, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<06:04, 1.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<04:29, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:31, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<09:59, 966kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:10, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<05:59, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:27, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:23, 1.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<09:52, 971kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<08:05, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<05:58, 1.60MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:24, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<07:19, 1.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<09:49, 969kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<08:02, 1.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<05:53, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<04:19, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:20, 2.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<32:00, 295kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<27:00, 350kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<20:02, 471kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<14:16, 660kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<10:08, 926kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<07:25, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<33:19, 282kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<28:25, 330kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<21:01, 446kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<14:57, 625kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<10:40, 874kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<11:42, 795kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<12:46, 729kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<10:04, 923kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<07:21, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:02<05:20, 1.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:50, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<10:00, 923kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<08:08, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<05:56, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<04:26, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<03:16, 2.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<49:12, 186kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<38:55, 236kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<28:18, 324kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<20:02, 456kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<14:10, 644kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<14:03, 648kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<13:52, 656kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<10:48, 842kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<07:48, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<05:39, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<08:52, 1.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<10:38, 849kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<08:30, 1.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<06:13, 1.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<04:33, 1.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:26, 1.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<09:52, 908kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:58, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<05:52, 1.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<04:16, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<08:19, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<09:46, 911kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:46, 1.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<05:43, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<04:09, 2.12MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<09:37, 918kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<10:19, 855kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<08:09, 1.08MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<05:58, 1.47MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<04:19, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<11:30, 761kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<11:38, 752kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<09:02, 969kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<06:31, 1.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<04:43, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<20:32, 423kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<17:37, 493kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<13:02, 666kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<09:22, 925kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<06:45, 1.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<08:01, 1.07MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<08:52, 971kB/s] .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<06:59, 1.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<05:03, 1.70MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<19:58, 428kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<16:58, 504kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<12:30, 683kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<08:57, 952kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<08:15, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<08:43, 972kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:51, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<04:59, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:28, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:34, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:10, 1.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:48, 2.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:47, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:53, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:38, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:24, 2.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:48, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<05:36, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:29, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<03:16, 2.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:57, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:41, 1.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:27, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:15, 2.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<02:26, 3.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<16:30, 493kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<13:38, 596kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<10:02, 809kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<07:08, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<08:02, 1.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:35, 1.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:46, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<04:09, 1.92MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<06:39, 1.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<06:26, 1.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:56, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:34, 2.22MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:51, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<06:28, 1.22MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:56, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<03:34, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<08:19, 944kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<07:25, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:31, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:58, 1.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<05:53, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:38, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:15, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:05, 2.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<05:15, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:39, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:26, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:11, 2.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<06:20, 1.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:12, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:42, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:23, 2.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:16, 1.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:58, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:44, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:41, 2.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<13:19, 564kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<11:00, 683kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<08:02, 933kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<05:43, 1.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<06:35, 1.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<05:44, 1.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:17, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:19, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<04:31, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<03:32, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:38, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:02, 1.81MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:08, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<02:17, 3.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:46, 1.52MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:49, 1.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:44, 1.93MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:45, 1.91MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:05, 1.75MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:13, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:18, 3.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<2:19:52, 50.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<1:39:18, 71.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<1:09:41, 102kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<48:41, 145kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<36:33, 192kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<27:00, 260kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<19:13, 365kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<13:26, 519kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<2:23:52, 48.4kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<1:42:04, 68.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<1:11:40, 97.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<50:54, 135kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<37:01, 186kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<26:13, 262kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<18:17, 373kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<1:51:32, 61.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<1:19:25, 85.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<55:47, 122kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<38:55, 174kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<30:55, 219kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<22:33, 300kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<15:56, 423kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:15<11:09, 600kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<52:17, 128kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<37:47, 177kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<26:42, 250kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<19:39, 337kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<14:57, 443kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<10:42, 617kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<07:33, 869kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<08:04, 812kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<06:52, 953kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<05:05, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:37, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:24, 1.47MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:22, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:25, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:33, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:46, 2.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:59, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:14, 1.95MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:33, 2.48MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:50, 2.22MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:07, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:27, 2.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:45, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:05, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:24, 2.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<01:45, 3.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:37, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:23, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:18, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<02:22, 2.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<06:00, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<05:20, 1.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:00, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:47, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:46, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:52, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:04, 2.86MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<05:13, 1.13MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:44, 1.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:34, 1.65MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:28, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:30, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:42, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:51, 2.03MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:02, 1.90MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:23, 2.42MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:37, 2.18MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:52, 2.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:16, 2.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:31, 2.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:47, 2.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:10, 2.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<01:35, 3.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<04:09, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:55, 1.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:59, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:00, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:06, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:25, 2.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:35, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:47, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:09, 2.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:34, 3.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<05:08, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<04:34, 1.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:26, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:16, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:17, 1.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:32, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:38, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:50, 1.85MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:11, 2.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<01:34, 3.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<06:47, 763kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<05:43, 905kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<04:13, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:47, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<03:37, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:45, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:45, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:50, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:10, 2.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<01:34, 3.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:03, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:44, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:50, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:47, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:50, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:12, 2.22MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:20, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:30, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:58, 2.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:10, 2.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:22, 2.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:50, 2.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<01:20, 3.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:06, 1.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:43, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:46, 1.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<01:59, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<05:25, 854kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:37, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:26, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:09, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:01, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:19, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:21, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:27, 1.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:53, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:22, 3.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<05:11, 853kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:27, 992kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<03:18, 1.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:01, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:55, 1.48MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:14, 1.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:16, 1.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:24, 1.79MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:52, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:00, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:11, 1.92MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:41, 2.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:13, 3.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:01, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:51, 1.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:09, 1.92MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [04:30<01:32, 2.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:42, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:20, 1.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:29, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<01:46, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<04:35, 874kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<03:56, 1.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:55, 1.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:41, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:35, 1.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:57, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:24, 2.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:41, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:16, 1.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:27, 1.57MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<02:20, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:19, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:45, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:16, 2.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<03:02, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:48, 1.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:07, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:05, 1.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:07, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:38, 2.22MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:44, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:52, 1.92MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:28, 2.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:36, 2.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:47, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:22, 2.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:00, 3.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:28, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<02:22, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:49, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:49, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:54, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<01:27, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:03, 3.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<03:26, 967kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<03:01, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<02:14, 1.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:35, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<03:12, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:50, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<02:07, 1.52MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:00, 1.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:58, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:29, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:04, 2.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:30, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:18, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:45, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:43, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:45, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:20, 2.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<00:58, 3.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:17, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:43, 1.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:08, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:32, 1.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:45, 1.65MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:46, 1.63MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:21, 2.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<00:58, 2.94MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:41, 1.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:24, 1.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:48, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:42, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:43, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:19, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:21, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:27, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:07, 2.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<00:48, 3.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:55, 900kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:31, 1.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:52, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:43, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:40, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:16, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:17, 1.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:21, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:02, 2.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<00:45, 3.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:44, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:39, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:14, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:53, 2.67MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:00, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:49, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:22, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:20, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:20, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:01, 2.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<00:43, 3.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:25<02:30, 887kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:09, 1.03MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:35, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<01:07, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<02:18, 937kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:59, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:29, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:22, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:20, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:00, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:43, 2.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:44, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:35, 1.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:11, 1.69MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:50, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:53, 1.03MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:41, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:15, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:10, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:10, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:53, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:37, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:33, 1.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:25, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:04, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:01, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:01, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:52, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:39, 2.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:47, 2.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:51, 1.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:40, 2.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:43, 2.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:47, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:37, 2.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:41, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:45, 2.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:35, 2.56MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:38, 2.27MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:43, 2.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:33, 2.57MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:37, 2.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:41, 2.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:32, 2.57MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:35, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:39, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:30, 2.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:33, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:49, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:40, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:29, 2.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:37, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:39, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:30, 2.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:32, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:34, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:26, 2.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:28, 2.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:43, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:36, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:26, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:32, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:33, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:25, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:26, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:28, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:22, 2.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:23, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:25, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:20, 2.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:21, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:23, 1.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:18, 2.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:19, 2.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:21, 2.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:16, 2.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:17, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:19, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:14, 2.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<00:10, 3.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:24, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:23, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:17, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:12<00:11, 2.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:26, 1.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:23, 1.28MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:17, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:15, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:15, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:11, 2.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:07, 3.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:16, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:15, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:11, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:04, 3.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:02, 3.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 1.76MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 1.76MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.73MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.22MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<66:08:20,  1.68it/s]  0%|          | 908/400000 [00:00<46:11:45,  2.40it/s]  0%|          | 1859/400000 [00:00<32:15:49,  3.43it/s]  1%|          | 2763/400000 [00:00<22:32:12,  4.90it/s]  1%|          | 3679/400000 [00:00<15:44:35,  6.99it/s]  1%|          | 4563/400000 [00:01<10:59:57,  9.99it/s]  1%|▏         | 5508/400000 [00:01<7:41:04, 14.26it/s]   2%|▏         | 6397/400000 [00:01<5:22:14, 20.36it/s]  2%|▏         | 7278/400000 [00:01<3:45:17, 29.05it/s]  2%|▏         | 8134/400000 [00:01<2:37:35, 41.44it/s]  2%|▏         | 9034/400000 [00:01<1:50:16, 59.09it/s]  2%|▏         | 9956/400000 [00:01<1:17:13, 84.18it/s]  3%|▎         | 10969/400000 [00:01<54:06, 119.83it/s]  3%|▎         | 11890/400000 [00:01<38:00, 170.18it/s]  3%|▎         | 12796/400000 [00:02<26:46, 241.00it/s]  3%|▎         | 13679/400000 [00:02<18:55, 340.20it/s]  4%|▎         | 14589/400000 [00:02<13:25, 478.33it/s]  4%|▍         | 15596/400000 [00:02<09:34, 669.69it/s]  4%|▍         | 16558/400000 [00:02<06:52, 928.98it/s]  4%|▍         | 17493/400000 [00:02<05:02, 1264.16it/s]  5%|▍         | 18387/400000 [00:02<03:45, 1692.80it/s]  5%|▍         | 19311/400000 [00:02<02:49, 2242.19it/s]  5%|▌         | 20208/400000 [00:02<02:11, 2893.09it/s]  5%|▌         | 21125/400000 [00:02<01:44, 3640.63it/s]  6%|▌         | 22072/400000 [00:03<01:24, 4464.75it/s]  6%|▌         | 23001/400000 [00:03<01:11, 5288.10it/s]  6%|▌         | 23970/400000 [00:03<01:01, 6122.25it/s]  6%|▌         | 24961/400000 [00:03<00:54, 6914.06it/s]  6%|▋         | 25911/400000 [00:03<00:51, 7318.69it/s]  7%|▋         | 26846/400000 [00:03<00:47, 7827.55it/s]  7%|▋         | 27783/400000 [00:03<00:45, 8232.86it/s]  7%|▋         | 28711/400000 [00:03<00:44, 8331.33it/s]  7%|▋         | 29686/400000 [00:03<00:42, 8711.37it/s]  8%|▊         | 30613/400000 [00:03<00:42, 8775.84it/s]  8%|▊         | 31530/400000 [00:04<00:42, 8690.23it/s]  8%|▊         | 32431/400000 [00:04<00:41, 8781.27it/s]  8%|▊         | 33329/400000 [00:04<00:41, 8764.05it/s]  9%|▊         | 34268/400000 [00:04<00:40, 8942.52it/s]  9%|▉         | 35175/400000 [00:04<00:40, 8980.24it/s]  9%|▉         | 36104/400000 [00:04<00:40, 9068.75it/s]  9%|▉         | 37017/400000 [00:04<00:40, 9044.67it/s]  9%|▉         | 37941/400000 [00:04<00:39, 9102.15it/s] 10%|▉         | 38854/400000 [00:04<00:40, 8855.47it/s] 10%|▉         | 39743/400000 [00:05<00:42, 8494.06it/s] 10%|█         | 40625/400000 [00:05<00:41, 8588.43it/s] 10%|█         | 41536/400000 [00:05<00:41, 8736.12it/s] 11%|█         | 42466/400000 [00:05<00:40, 8895.46it/s] 11%|█         | 43429/400000 [00:05<00:39, 9103.66it/s] 11%|█         | 44343/400000 [00:05<00:39, 9000.75it/s] 11%|█▏        | 45308/400000 [00:05<00:38, 9185.24it/s] 12%|█▏        | 46265/400000 [00:05<00:38, 9293.26it/s] 12%|█▏        | 47198/400000 [00:05<00:37, 9302.93it/s] 12%|█▏        | 48195/400000 [00:05<00:37, 9493.41it/s] 12%|█▏        | 49147/400000 [00:06<00:37, 9420.98it/s] 13%|█▎        | 50091/400000 [00:06<00:37, 9421.36it/s] 13%|█▎        | 51087/400000 [00:06<00:36, 9575.21it/s] 13%|█▎        | 52053/400000 [00:06<00:36, 9598.71it/s] 13%|█▎        | 53074/400000 [00:06<00:35, 9772.52it/s] 14%|█▎        | 54053/400000 [00:06<00:36, 9505.06it/s] 14%|█▍        | 55007/400000 [00:06<00:36, 9476.70it/s] 14%|█▍        | 55957/400000 [00:06<00:37, 9290.98it/s] 14%|█▍        | 56889/400000 [00:06<00:36, 9284.99it/s] 14%|█▍        | 57819/400000 [00:06<00:37, 9024.23it/s] 15%|█▍        | 58725/400000 [00:07<00:37, 9002.13it/s] 15%|█▍        | 59628/400000 [00:07<00:37, 8999.60it/s] 15%|█▌        | 60580/400000 [00:07<00:37, 9147.63it/s] 15%|█▌        | 61497/400000 [00:07<00:37, 9077.91it/s] 16%|█▌        | 62418/400000 [00:07<00:37, 9115.15it/s] 16%|█▌        | 63331/400000 [00:07<00:37, 9060.80it/s] 16%|█▌        | 64271/400000 [00:07<00:36, 9159.66it/s] 16%|█▋        | 65242/400000 [00:07<00:35, 9316.61it/s] 17%|█▋        | 66222/400000 [00:07<00:35, 9453.74it/s] 17%|█▋        | 67170/400000 [00:07<00:35, 9459.28it/s] 17%|█▋        | 68117/400000 [00:08<00:36, 9206.23it/s] 17%|█▋        | 69040/400000 [00:08<00:36, 9055.31it/s] 17%|█▋        | 69977/400000 [00:08<00:36, 9146.09it/s] 18%|█▊        | 70941/400000 [00:08<00:35, 9287.94it/s] 18%|█▊        | 71892/400000 [00:08<00:35, 9352.02it/s] 18%|█▊        | 72829/400000 [00:08<00:35, 9210.67it/s] 18%|█▊        | 73805/400000 [00:08<00:34, 9366.92it/s] 19%|█▊        | 74744/400000 [00:08<00:35, 9263.73it/s] 19%|█▉        | 75685/400000 [00:08<00:34, 9306.13it/s] 19%|█▉        | 76617/400000 [00:08<00:34, 9267.22it/s] 19%|█▉        | 77545/400000 [00:09<00:35, 9095.35it/s] 20%|█▉        | 78456/400000 [00:09<00:35, 8986.28it/s] 20%|█▉        | 79356/400000 [00:09<00:36, 8808.89it/s] 20%|██        | 80295/400000 [00:09<00:35, 8973.33it/s] 20%|██        | 81224/400000 [00:09<00:35, 9064.82it/s] 21%|██        | 82133/400000 [00:09<00:35, 8913.72it/s] 21%|██        | 83030/400000 [00:09<00:35, 8927.97it/s] 21%|██        | 83924/400000 [00:09<00:35, 8829.86it/s] 21%|██        | 84811/400000 [00:09<00:35, 8841.32it/s] 21%|██▏       | 85699/400000 [00:09<00:35, 8851.28it/s] 22%|██▏       | 86585/400000 [00:10<00:35, 8716.95it/s] 22%|██▏       | 87536/400000 [00:10<00:34, 8940.44it/s] 22%|██▏       | 88433/400000 [00:10<00:35, 8744.01it/s] 22%|██▏       | 89334/400000 [00:10<00:35, 8820.09it/s] 23%|██▎       | 90284/400000 [00:10<00:34, 9009.89it/s] 23%|██▎       | 91188/400000 [00:10<00:34, 8967.97it/s] 23%|██▎       | 92129/400000 [00:10<00:33, 9095.52it/s] 23%|██▎       | 93054/400000 [00:10<00:33, 9141.18it/s] 23%|██▎       | 93970/400000 [00:10<00:33, 9140.90it/s] 24%|██▎       | 94908/400000 [00:11<00:33, 9209.87it/s] 24%|██▍       | 95830/400000 [00:11<00:33, 9140.14it/s] 24%|██▍       | 96745/400000 [00:11<00:33, 9113.36it/s] 24%|██▍       | 97657/400000 [00:11<00:33, 8993.32it/s] 25%|██▍       | 98608/400000 [00:11<00:32, 9141.40it/s] 25%|██▍       | 99571/400000 [00:11<00:32, 9280.21it/s] 25%|██▌       | 100501/400000 [00:11<00:33, 9048.25it/s] 25%|██▌       | 101421/400000 [00:11<00:32, 9091.63it/s] 26%|██▌       | 102433/400000 [00:11<00:31, 9375.83it/s] 26%|██▌       | 103374/400000 [00:11<00:31, 9296.88it/s] 26%|██▌       | 104308/400000 [00:12<00:31, 9307.87it/s] 26%|██▋       | 105241/400000 [00:12<00:31, 9230.72it/s] 27%|██▋       | 106210/400000 [00:12<00:31, 9361.53it/s] 27%|██▋       | 107148/400000 [00:12<00:31, 9186.33it/s] 27%|██▋       | 108069/400000 [00:12<00:31, 9162.13it/s] 27%|██▋       | 108987/400000 [00:12<00:31, 9151.06it/s] 27%|██▋       | 109911/400000 [00:12<00:31, 9177.36it/s] 28%|██▊       | 110873/400000 [00:12<00:31, 9298.39it/s] 28%|██▊       | 111804/400000 [00:12<00:31, 9021.15it/s] 28%|██▊       | 112727/400000 [00:12<00:31, 9081.94it/s] 28%|██▊       | 113637/400000 [00:13<00:31, 9050.37it/s] 29%|██▊       | 114551/400000 [00:13<00:31, 9075.08it/s] 29%|██▉       | 115460/400000 [00:13<00:32, 8882.72it/s] 29%|██▉       | 116352/400000 [00:13<00:31, 8893.45it/s] 29%|██▉       | 117296/400000 [00:13<00:31, 9049.79it/s] 30%|██▉       | 118203/400000 [00:13<00:31, 8855.24it/s] 30%|██▉       | 119096/400000 [00:13<00:31, 8875.46it/s] 30%|███       | 120044/400000 [00:13<00:30, 9047.15it/s] 30%|███       | 120987/400000 [00:13<00:30, 9156.81it/s] 30%|███       | 121959/400000 [00:13<00:29, 9316.98it/s] 31%|███       | 122893/400000 [00:14<00:30, 9210.73it/s] 31%|███       | 123820/400000 [00:14<00:29, 9228.42it/s] 31%|███       | 124750/400000 [00:14<00:29, 9249.76it/s] 31%|███▏      | 125676/400000 [00:14<00:31, 8747.17it/s] 32%|███▏      | 126571/400000 [00:14<00:31, 8804.74it/s] 32%|███▏      | 127456/400000 [00:14<00:30, 8807.29it/s] 32%|███▏      | 128340/400000 [00:14<00:30, 8782.15it/s] 32%|███▏      | 129288/400000 [00:14<00:30, 8980.24it/s] 33%|███▎      | 130236/400000 [00:14<00:29, 9123.86it/s] 33%|███▎      | 131195/400000 [00:14<00:29, 9258.09it/s] 33%|███▎      | 132123/400000 [00:15<00:29, 9174.42it/s] 33%|███▎      | 133077/400000 [00:15<00:28, 9280.53it/s] 34%|███▎      | 134070/400000 [00:15<00:28, 9465.06it/s] 34%|███▍      | 135041/400000 [00:15<00:27, 9533.95it/s] 34%|███▍      | 136002/400000 [00:15<00:27, 9555.68it/s] 34%|███▍      | 136959/400000 [00:15<00:27, 9404.08it/s] 34%|███▍      | 137936/400000 [00:15<00:27, 9508.11it/s] 35%|███▍      | 138925/400000 [00:15<00:27, 9618.33it/s] 35%|███▍      | 139888/400000 [00:15<00:27, 9606.30it/s] 35%|███▌      | 140850/400000 [00:16<00:27, 9546.32it/s] 35%|███▌      | 141806/400000 [00:16<00:27, 9380.03it/s] 36%|███▌      | 142746/400000 [00:16<00:27, 9224.85it/s] 36%|███▌      | 143670/400000 [00:16<00:27, 9165.40it/s] 36%|███▌      | 144588/400000 [00:16<00:28, 9006.22it/s] 36%|███▋      | 145531/400000 [00:16<00:27, 9126.99it/s] 37%|███▋      | 146446/400000 [00:16<00:28, 9013.21it/s] 37%|███▋      | 147429/400000 [00:16<00:27, 9242.74it/s] 37%|███▋      | 148373/400000 [00:16<00:27, 9300.79it/s] 37%|███▋      | 149305/400000 [00:16<00:27, 9284.70it/s] 38%|███▊      | 150284/400000 [00:17<00:26, 9429.36it/s] 38%|███▊      | 151229/400000 [00:17<00:26, 9346.11it/s] 38%|███▊      | 152223/400000 [00:17<00:26, 9516.30it/s] 38%|███▊      | 153177/400000 [00:17<00:25, 9508.38it/s] 39%|███▊      | 154129/400000 [00:17<00:26, 9313.14it/s] 39%|███▉      | 155062/400000 [00:17<00:26, 9184.46it/s] 39%|███▉      | 155983/400000 [00:17<00:27, 8813.31it/s] 39%|███▉      | 156869/400000 [00:17<00:27, 8773.69it/s] 39%|███▉      | 157750/400000 [00:17<00:27, 8716.17it/s] 40%|███▉      | 158734/400000 [00:17<00:26, 9023.53it/s] 40%|███▉      | 159686/400000 [00:18<00:26, 9166.85it/s] 40%|████      | 160607/400000 [00:18<00:26, 9104.08it/s] 40%|████      | 161543/400000 [00:18<00:25, 9178.55it/s] 41%|████      | 162544/400000 [00:18<00:25, 9412.95it/s] 41%|████      | 163557/400000 [00:18<00:24, 9616.79it/s] 41%|████      | 164522/400000 [00:18<00:24, 9544.57it/s] 41%|████▏     | 165479/400000 [00:18<00:25, 9201.03it/s] 42%|████▏     | 166404/400000 [00:18<00:25, 9213.99it/s] 42%|████▏     | 167369/400000 [00:18<00:24, 9339.24it/s] 42%|████▏     | 168306/400000 [00:18<00:24, 9330.54it/s] 42%|████▏     | 169250/400000 [00:19<00:24, 9362.16it/s] 43%|████▎     | 170188/400000 [00:19<00:24, 9279.09it/s] 43%|████▎     | 171117/400000 [00:19<00:24, 9234.45it/s] 43%|████▎     | 172042/400000 [00:19<00:24, 9225.44it/s] 43%|████▎     | 172966/400000 [00:19<00:24, 9094.11it/s] 43%|████▎     | 173877/400000 [00:19<00:25, 8852.56it/s] 44%|████▎     | 174765/400000 [00:19<00:26, 8594.70it/s] 44%|████▍     | 175628/400000 [00:19<00:26, 8599.10it/s] 44%|████▍     | 176534/400000 [00:19<00:25, 8731.24it/s] 44%|████▍     | 177454/400000 [00:20<00:25, 8865.80it/s] 45%|████▍     | 178343/400000 [00:20<00:25, 8735.81it/s] 45%|████▍     | 179248/400000 [00:20<00:25, 8825.62it/s] 45%|████▌     | 180140/400000 [00:20<00:24, 8851.89it/s] 45%|████▌     | 181036/400000 [00:20<00:24, 8883.44it/s] 45%|████▌     | 181926/400000 [00:20<00:25, 8622.84it/s] 46%|████▌     | 182791/400000 [00:20<00:25, 8515.67it/s] 46%|████▌     | 183645/400000 [00:20<00:25, 8487.54it/s] 46%|████▌     | 184506/400000 [00:20<00:25, 8521.57it/s] 46%|████▋     | 185388/400000 [00:20<00:24, 8606.60it/s] 47%|████▋     | 186284/400000 [00:21<00:24, 8708.35it/s] 47%|████▋     | 187193/400000 [00:21<00:24, 8817.96it/s] 47%|████▋     | 188090/400000 [00:21<00:23, 8861.78it/s] 47%|████▋     | 188994/400000 [00:21<00:23, 8913.55it/s] 47%|████▋     | 189939/400000 [00:21<00:23, 9066.07it/s] 48%|████▊     | 190887/400000 [00:21<00:22, 9183.98it/s] 48%|████▊     | 191807/400000 [00:21<00:22, 9102.03it/s] 48%|████▊     | 192719/400000 [00:21<00:23, 8881.36it/s] 48%|████▊     | 193609/400000 [00:21<00:23, 8861.42it/s] 49%|████▊     | 194497/400000 [00:21<00:23, 8791.82it/s] 49%|████▉     | 195409/400000 [00:22<00:23, 8887.12it/s] 49%|████▉     | 196309/400000 [00:22<00:22, 8918.05it/s] 49%|████▉     | 197202/400000 [00:22<00:23, 8622.49it/s] 50%|████▉     | 198067/400000 [00:22<00:23, 8585.02it/s] 50%|████▉     | 199030/400000 [00:22<00:22, 8873.60it/s] 50%|████▉     | 199984/400000 [00:22<00:22, 9062.81it/s] 50%|█████     | 200908/400000 [00:22<00:21, 9114.44it/s] 50%|█████     | 201823/400000 [00:22<00:21, 9112.71it/s] 51%|█████     | 202834/400000 [00:22<00:21, 9387.74it/s] 51%|█████     | 203777/400000 [00:22<00:20, 9381.92it/s] 51%|█████     | 204727/400000 [00:23<00:20, 9415.93it/s] 51%|█████▏    | 205680/400000 [00:23<00:20, 9448.06it/s] 52%|█████▏    | 206627/400000 [00:23<00:20, 9355.79it/s] 52%|█████▏    | 207564/400000 [00:23<00:20, 9249.24it/s] 52%|█████▏    | 208495/400000 [00:23<00:20, 9238.63it/s] 52%|█████▏    | 209420/400000 [00:23<00:21, 8971.15it/s] 53%|█████▎    | 210320/400000 [00:23<00:21, 8901.10it/s] 53%|█████▎    | 211212/400000 [00:23<00:21, 8669.10it/s] 53%|█████▎    | 212082/400000 [00:23<00:21, 8642.94it/s] 53%|█████▎    | 213037/400000 [00:24<00:21, 8894.16it/s] 53%|█████▎    | 213998/400000 [00:24<00:20, 9094.47it/s] 54%|█████▎    | 214940/400000 [00:24<00:20, 9188.72it/s] 54%|█████▍    | 215872/400000 [00:24<00:19, 9227.30it/s] 54%|█████▍    | 216801/400000 [00:24<00:19, 9243.59it/s] 54%|█████▍    | 217727/400000 [00:24<00:20, 9056.25it/s] 55%|█████▍    | 218653/400000 [00:24<00:19, 9116.33it/s] 55%|█████▍    | 219566/400000 [00:24<00:19, 9097.84it/s] 55%|█████▌    | 220477/400000 [00:24<00:20, 8930.41it/s] 55%|█████▌    | 221372/400000 [00:24<00:20, 8849.46it/s] 56%|█████▌    | 222259/400000 [00:25<00:20, 8796.10it/s] 56%|█████▌    | 223164/400000 [00:25<00:19, 8870.00it/s] 56%|█████▌    | 224064/400000 [00:25<00:19, 8906.78it/s] 56%|█████▌    | 224956/400000 [00:25<00:19, 8874.06it/s] 56%|█████▋    | 225896/400000 [00:25<00:19, 9023.81it/s] 57%|█████▋    | 226828/400000 [00:25<00:19, 9107.85it/s] 57%|█████▋    | 227749/400000 [00:25<00:18, 9135.29it/s] 57%|█████▋    | 228664/400000 [00:25<00:18, 9133.30it/s] 57%|█████▋    | 229593/400000 [00:25<00:18, 9177.46it/s] 58%|█████▊    | 230512/400000 [00:25<00:18, 9057.75it/s] 58%|█████▊    | 231419/400000 [00:26<00:18, 8935.00it/s] 58%|█████▊    | 232314/400000 [00:26<00:18, 8858.09it/s] 58%|█████▊    | 233246/400000 [00:26<00:18, 8990.95it/s] 59%|█████▊    | 234205/400000 [00:26<00:18, 9159.17it/s] 59%|█████▉    | 235123/400000 [00:26<00:18, 8974.81it/s] 59%|█████▉    | 236077/400000 [00:26<00:17, 9135.27it/s] 59%|█████▉    | 237010/400000 [00:26<00:17, 9192.13it/s] 59%|█████▉    | 237958/400000 [00:26<00:17, 9273.98it/s] 60%|█████▉    | 238971/400000 [00:26<00:16, 9514.28it/s] 60%|█████▉    | 239925/400000 [00:26<00:17, 9308.58it/s] 60%|██████    | 240859/400000 [00:27<00:17, 9311.78it/s] 60%|██████    | 241807/400000 [00:27<00:16, 9360.58it/s] 61%|██████    | 242770/400000 [00:27<00:16, 9437.82it/s] 61%|██████    | 243721/400000 [00:27<00:16, 9457.82it/s] 61%|██████    | 244671/400000 [00:27<00:16, 9468.60it/s] 61%|██████▏   | 245619/400000 [00:27<00:16, 9338.63it/s] 62%|██████▏   | 246554/400000 [00:27<00:16, 9229.20it/s] 62%|██████▏   | 247506/400000 [00:27<00:16, 9314.20it/s] 62%|██████▏   | 248478/400000 [00:27<00:16, 9429.65it/s] 62%|██████▏   | 249429/400000 [00:27<00:15, 9452.63it/s] 63%|██████▎   | 250396/400000 [00:28<00:15, 9514.66it/s] 63%|██████▎   | 251364/400000 [00:28<00:15, 9561.32it/s] 63%|██████▎   | 252361/400000 [00:28<00:15, 9677.71it/s] 63%|██████▎   | 253330/400000 [00:28<00:15, 9551.03it/s] 64%|██████▎   | 254286/400000 [00:28<00:15, 9501.81it/s] 64%|██████▍   | 255275/400000 [00:28<00:15, 9613.06it/s] 64%|██████▍   | 256238/400000 [00:28<00:14, 9592.46it/s] 64%|██████▍   | 257230/400000 [00:28<00:14, 9688.34it/s] 65%|██████▍   | 258200/400000 [00:28<00:15, 9391.29it/s] 65%|██████▍   | 259173/400000 [00:28<00:14, 9489.47it/s] 65%|██████▌   | 260124/400000 [00:29<00:14, 9332.13it/s] 65%|██████▌   | 261060/400000 [00:29<00:15, 9184.21it/s] 66%|██████▌   | 262032/400000 [00:29<00:14, 9336.58it/s] 66%|██████▌   | 262968/400000 [00:29<00:15, 9124.12it/s] 66%|██████▌   | 263883/400000 [00:29<00:14, 9090.75it/s] 66%|██████▌   | 264805/400000 [00:29<00:14, 9127.60it/s] 66%|██████▋   | 265741/400000 [00:29<00:14, 9194.97it/s] 67%|██████▋   | 266662/400000 [00:29<00:14, 9133.47it/s] 67%|██████▋   | 267577/400000 [00:29<00:14, 9094.84it/s] 67%|██████▋   | 268488/400000 [00:30<00:14, 8950.54it/s] 67%|██████▋   | 269384/400000 [00:30<00:15, 8677.69it/s] 68%|██████▊   | 270271/400000 [00:30<00:14, 8734.34it/s] 68%|██████▊   | 271228/400000 [00:30<00:14, 8968.02it/s] 68%|██████▊   | 272128/400000 [00:30<00:14, 8946.44it/s] 68%|██████▊   | 273046/400000 [00:30<00:14, 9014.36it/s] 68%|██████▊   | 273951/400000 [00:30<00:13, 9022.31it/s] 69%|██████▊   | 274855/400000 [00:30<00:13, 9020.46it/s] 69%|██████▉   | 275758/400000 [00:30<00:13, 8930.95it/s] 69%|██████▉   | 276667/400000 [00:30<00:13, 8977.55it/s] 69%|██████▉   | 277593/400000 [00:31<00:13, 9058.90it/s] 70%|██████▉   | 278561/400000 [00:31<00:13, 9234.15it/s] 70%|██████▉   | 279486/400000 [00:31<00:13, 8975.71it/s] 70%|███████   | 280400/400000 [00:31<00:13, 9023.49it/s] 70%|███████   | 281305/400000 [00:31<00:13, 8954.40it/s] 71%|███████   | 282242/400000 [00:31<00:12, 9074.78it/s] 71%|███████   | 283151/400000 [00:31<00:12, 9009.80it/s] 71%|███████   | 284056/400000 [00:31<00:12, 9019.75it/s] 71%|███████   | 284979/400000 [00:31<00:12, 9079.04it/s] 71%|███████▏  | 285888/400000 [00:31<00:12, 9003.41it/s] 72%|███████▏  | 286848/400000 [00:32<00:12, 9172.38it/s] 72%|███████▏  | 287812/400000 [00:32<00:12, 9306.37it/s] 72%|███████▏  | 288744/400000 [00:32<00:12, 9231.36it/s] 72%|███████▏  | 289723/400000 [00:32<00:11, 9391.47it/s] 73%|███████▎  | 290664/400000 [00:32<00:12, 8970.50it/s] 73%|███████▎  | 291644/400000 [00:32<00:11, 9203.79it/s] 73%|███████▎  | 292570/400000 [00:32<00:11, 9085.75it/s] 73%|███████▎  | 293499/400000 [00:32<00:11, 9145.66it/s] 74%|███████▎  | 294436/400000 [00:32<00:11, 9211.49it/s] 74%|███████▍  | 295360/400000 [00:32<00:11, 9180.48it/s] 74%|███████▍  | 296354/400000 [00:33<00:11, 9394.80it/s] 74%|███████▍  | 297296/400000 [00:33<00:11, 9241.55it/s] 75%|███████▍  | 298223/400000 [00:33<00:11, 8991.40it/s] 75%|███████▍  | 299181/400000 [00:33<00:11, 9157.39it/s] 75%|███████▌  | 300100/400000 [00:33<00:10, 9120.44it/s] 75%|███████▌  | 301068/400000 [00:33<00:10, 9279.75it/s] 76%|███████▌  | 302031/400000 [00:33<00:10, 9381.57it/s] 76%|███████▌  | 302971/400000 [00:33<00:10, 9305.06it/s] 76%|███████▌  | 303903/400000 [00:33<00:10, 9276.89it/s] 76%|███████▌  | 304832/400000 [00:34<00:10, 9157.62it/s] 76%|███████▋  | 305770/400000 [00:34<00:10, 9222.32it/s] 77%|███████▋  | 306718/400000 [00:34<00:10, 9297.38it/s] 77%|███████▋  | 307716/400000 [00:34<00:09, 9489.56it/s] 77%|███████▋  | 308667/400000 [00:34<00:09, 9442.51it/s] 77%|███████▋  | 309613/400000 [00:34<00:09, 9429.46it/s] 78%|███████▊  | 310557/400000 [00:34<00:09, 9405.68it/s] 78%|███████▊  | 311499/400000 [00:34<00:09, 9382.59it/s] 78%|███████▊  | 312489/400000 [00:34<00:09, 9529.23it/s] 78%|███████▊  | 313443/400000 [00:34<00:09, 9483.09it/s] 79%|███████▊  | 314392/400000 [00:35<00:09, 9066.35it/s] 79%|███████▉  | 315303/400000 [00:35<00:09, 9077.69it/s] 79%|███████▉  | 316236/400000 [00:35<00:09, 9150.33it/s] 79%|███████▉  | 317154/400000 [00:35<00:09, 9038.50it/s] 80%|███████▉  | 318113/400000 [00:35<00:08, 9196.09it/s] 80%|███████▉  | 319035/400000 [00:35<00:08, 9109.35it/s] 80%|███████▉  | 319958/400000 [00:35<00:08, 9143.43it/s] 80%|████████  | 320876/400000 [00:35<00:08, 9153.45it/s] 80%|████████  | 321819/400000 [00:35<00:08, 9234.35it/s] 81%|████████  | 322744/400000 [00:35<00:08, 9234.23it/s] 81%|████████  | 323668/400000 [00:36<00:08, 9138.96it/s] 81%|████████  | 324603/400000 [00:36<00:08, 9199.69it/s] 81%|████████▏ | 325581/400000 [00:36<00:07, 9364.03it/s] 82%|████████▏ | 326519/400000 [00:36<00:07, 9342.28it/s] 82%|████████▏ | 327460/400000 [00:36<00:07, 9359.93it/s] 82%|████████▏ | 328397/400000 [00:36<00:07, 9229.29it/s] 82%|████████▏ | 329321/400000 [00:36<00:07, 9051.91it/s] 83%|████████▎ | 330236/400000 [00:36<00:07, 9079.90it/s] 83%|████████▎ | 331174/400000 [00:36<00:07, 9166.21it/s] 83%|████████▎ | 332092/400000 [00:36<00:07, 8751.09it/s] 83%|████████▎ | 332974/400000 [00:37<00:07, 8771.59it/s] 83%|████████▎ | 333875/400000 [00:37<00:07, 8839.40it/s] 84%|████████▎ | 334762/400000 [00:37<00:07, 8804.89it/s] 84%|████████▍ | 335682/400000 [00:37<00:07, 8917.51it/s] 84%|████████▍ | 336582/400000 [00:37<00:07, 8939.41it/s] 84%|████████▍ | 337478/400000 [00:37<00:06, 8944.66it/s] 85%|████████▍ | 338385/400000 [00:37<00:06, 8981.87it/s] 85%|████████▍ | 339354/400000 [00:37<00:06, 9181.48it/s] 85%|████████▌ | 340353/400000 [00:37<00:06, 9408.14it/s] 85%|████████▌ | 341321/400000 [00:37<00:06, 9485.87it/s] 86%|████████▌ | 342272/400000 [00:38<00:06, 9165.71it/s] 86%|████████▌ | 343193/400000 [00:38<00:06, 9135.88it/s] 86%|████████▌ | 344150/400000 [00:38<00:06, 9261.73it/s] 86%|████████▋ | 345101/400000 [00:38<00:05, 9333.10it/s] 87%|████████▋ | 346076/400000 [00:38<00:05, 9453.11it/s] 87%|████████▋ | 347023/400000 [00:38<00:05, 9336.82it/s] 87%|████████▋ | 347959/400000 [00:38<00:05, 9296.71it/s] 87%|████████▋ | 348890/400000 [00:38<00:05, 9259.61it/s] 87%|████████▋ | 349817/400000 [00:38<00:05, 9059.79it/s] 88%|████████▊ | 350725/400000 [00:38<00:05, 8999.82it/s] 88%|████████▊ | 351627/400000 [00:39<00:05, 8902.21it/s] 88%|████████▊ | 352519/400000 [00:39<00:05, 8896.24it/s] 88%|████████▊ | 353452/400000 [00:39<00:05, 9021.76it/s] 89%|████████▊ | 354400/400000 [00:39<00:04, 9153.07it/s] 89%|████████▉ | 355365/400000 [00:39<00:04, 9296.68it/s] 89%|████████▉ | 356296/400000 [00:39<00:04, 9160.06it/s] 89%|████████▉ | 357214/400000 [00:39<00:04, 8995.55it/s] 90%|████████▉ | 358198/400000 [00:39<00:04, 9230.33it/s] 90%|████████▉ | 359124/400000 [00:39<00:04, 9166.87it/s] 90%|█████████ | 360043/400000 [00:40<00:04, 9070.90it/s] 90%|█████████ | 360952/400000 [00:40<00:04, 8618.24it/s] 90%|█████████ | 361849/400000 [00:40<00:04, 8719.14it/s] 91%|█████████ | 362736/400000 [00:40<00:04, 8763.76it/s] 91%|█████████ | 363664/400000 [00:40<00:04, 8911.78it/s] 91%|█████████ | 364601/400000 [00:40<00:03, 9043.60it/s] 91%|█████████▏| 365508/400000 [00:40<00:03, 9016.03it/s] 92%|█████████▏| 366473/400000 [00:40<00:03, 9196.19it/s] 92%|█████████▏| 367469/400000 [00:40<00:03, 9411.63it/s] 92%|█████████▏| 368417/400000 [00:40<00:03, 9428.71it/s] 92%|█████████▏| 369374/400000 [00:41<00:03, 9469.55it/s] 93%|█████████▎| 370323/400000 [00:41<00:03, 9343.98it/s] 93%|█████████▎| 371259/400000 [00:41<00:03, 9068.25it/s] 93%|█████████▎| 372169/400000 [00:41<00:03, 8959.62it/s] 93%|█████████▎| 373110/400000 [00:41<00:02, 9088.71it/s] 94%|█████████▎| 374021/400000 [00:41<00:02, 9074.96it/s] 94%|█████████▎| 374930/400000 [00:41<00:02, 9021.10it/s] 94%|█████████▍| 375834/400000 [00:41<00:02, 9020.51it/s] 94%|█████████▍| 376786/400000 [00:41<00:02, 9164.09it/s] 94%|█████████▍| 377704/400000 [00:41<00:02, 9164.71it/s] 95%|█████████▍| 378622/400000 [00:42<00:02, 9116.06it/s] 95%|█████████▍| 379535/400000 [00:42<00:02, 9032.03it/s] 95%|█████████▌| 380439/400000 [00:42<00:02, 9029.27it/s] 95%|█████████▌| 381358/400000 [00:42<00:02, 9073.96it/s] 96%|█████████▌| 382266/400000 [00:42<00:01, 9052.49it/s] 96%|█████████▌| 383191/400000 [00:42<00:01, 9110.84it/s] 96%|█████████▌| 384103/400000 [00:42<00:01, 8745.44it/s] 96%|█████████▋| 385077/400000 [00:42<00:01, 9020.97it/s] 97%|█████████▋| 386032/400000 [00:42<00:01, 9171.61it/s] 97%|█████████▋| 386953/400000 [00:42<00:01, 9103.93it/s] 97%|█████████▋| 387867/400000 [00:43<00:01, 9114.48it/s] 97%|█████████▋| 388781/400000 [00:43<00:01, 9004.41it/s] 97%|█████████▋| 389684/400000 [00:43<00:01, 8989.86it/s] 98%|█████████▊| 390604/400000 [00:43<00:01, 9050.99it/s] 98%|█████████▊| 391511/400000 [00:43<00:00, 8981.23it/s] 98%|█████████▊| 392422/400000 [00:43<00:00, 9016.97it/s] 98%|█████████▊| 393325/400000 [00:43<00:00, 8792.94it/s] 99%|█████████▊| 394206/400000 [00:43<00:00, 8767.39it/s] 99%|█████████▉| 395126/400000 [00:43<00:00, 8892.61it/s] 99%|█████████▉| 396017/400000 [00:44<00:00, 8892.21it/s] 99%|█████████▉| 396922/400000 [00:44<00:00, 8937.03it/s] 99%|█████████▉| 397817/400000 [00:44<00:00, 8906.40it/s]100%|█████████▉| 398743/400000 [00:44<00:00, 9008.27it/s]100%|█████████▉| 399645/400000 [00:44<00:00, 8842.91it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8998.78it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fdd27c23cc0>, <torchtext.data.dataset.TabularDataset object at 0x7fdd27c23e10>, <torchtext.vocab.Vocab object at 0x7fdd27c23d30>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.43 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 24.07 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.76 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.76 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.69 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.69 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.83 file/s]2020-07-28 06:17:37.530742: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-28 06:17:37.534633: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-28 06:17:37.534790: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b5a975c0b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-28 06:17:37.534804: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 464791.86it/s] 95%|█████████▍| 9412608/9912422 [00:00<00:00, 662578.50it/s]9920512it [00:00, 47352723.53it/s]                           
0it [00:00, ?it/s]32768it [00:00, 692436.51it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 457574.65it/s]1654784it [00:00, 11681458.65it/s]                         
0it [00:00, ?it/s]8192it [00:00, 241454.77it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
