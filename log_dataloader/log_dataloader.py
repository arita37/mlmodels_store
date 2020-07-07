
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fab096040d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fab096040d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fab7494d1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fab7494d1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fab92c95e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fab92c95e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fab21c7b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fab21c7b620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fab21c7b620> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 137939.12it/s] 67%|██████▋   | 6610944/9912422 [00:00<00:16, 196876.54it/s]9920512it [00:00, 38739149.65it/s]                           
0it [00:00, ?it/s]32768it [00:00, 729018.56it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1012658.55it/s]1654784it [00:00, 12585870.64it/s]                           
0it [00:00, ?it/s]8192it [00:00, 281667.21it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fab1f124978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fab08cafc18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fab21c7b268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fab21c7b268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fab21c7b268> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:22,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:22,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:21,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:20,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:20,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:19,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:19,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:55,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:55,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:54,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:54,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:54,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:53,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:53,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:52,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:35,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:35,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:35,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:35,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:24,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:24,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:23,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:23,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:23,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:16,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:16,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:15,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:15,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:15,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:15,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11, 10.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 14.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 14.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 14.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 14.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 18.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 18.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 24.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 24.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 24.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 24.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 24.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 31.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 39.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 47.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 47.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 47.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 47.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 47.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 47.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 54.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 54.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 62.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 75.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.96 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.52s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 78.96 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.52s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.52s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.87 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.52s/ url]
0 examples [00:00, ? examples/s]2020-07-07 12:09:05.705907: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-07 12:09:05.719286: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-07 12:09:05.719459: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a983195d50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-07 12:09:05.719476: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  7.52 examples/s]99 examples [00:00, 10.70 examples/s]205 examples [00:00, 15.22 examples/s]314 examples [00:00, 21.62 examples/s]423 examples [00:00, 30.62 examples/s]532 examples [00:00, 43.22 examples/s]641 examples [00:00, 60.70 examples/s]747 examples [00:00, 84.63 examples/s]856 examples [00:00, 116.98 examples/s]963 examples [00:01, 159.59 examples/s]1073 examples [00:01, 214.57 examples/s]1181 examples [00:01, 282.32 examples/s]1288 examples [00:01, 362.20 examples/s]1397 examples [00:01, 452.44 examples/s]1504 examples [00:01, 546.86 examples/s]1613 examples [00:01, 642.58 examples/s]1721 examples [00:01, 731.15 examples/s]1829 examples [00:01, 789.86 examples/s]1936 examples [00:01, 856.56 examples/s]2045 examples [00:02, 913.27 examples/s]2153 examples [00:02, 956.13 examples/s]2262 examples [00:02, 991.03 examples/s]2369 examples [00:02, 1011.25 examples/s]2476 examples [00:02, 1010.23 examples/s]2586 examples [00:02, 1033.25 examples/s]2693 examples [00:02, 1008.46 examples/s]2800 examples [00:02, 1023.59 examples/s]2906 examples [00:02, 1032.64 examples/s]3015 examples [00:02, 1047.59 examples/s]3125 examples [00:03, 1060.19 examples/s]3234 examples [00:03, 1067.63 examples/s]3343 examples [00:03, 1071.80 examples/s]3451 examples [00:03, 1067.56 examples/s]3558 examples [00:03, 1064.58 examples/s]3665 examples [00:03, 1064.47 examples/s]3773 examples [00:03, 1068.52 examples/s]3883 examples [00:03, 1074.91 examples/s]3991 examples [00:03, 1069.38 examples/s]4100 examples [00:03, 1073.34 examples/s]4210 examples [00:04, 1079.75 examples/s]4319 examples [00:04, 1072.32 examples/s]4427 examples [00:04, 1073.87 examples/s]4535 examples [00:04, 1054.31 examples/s]4643 examples [00:04, 1061.79 examples/s]4752 examples [00:04, 1069.24 examples/s]4859 examples [00:04, 1054.92 examples/s]4969 examples [00:04, 1066.11 examples/s]5076 examples [00:04, 1040.70 examples/s]5181 examples [00:05, 1014.66 examples/s]5290 examples [00:05, 1035.62 examples/s]5394 examples [00:05, 1029.02 examples/s]5501 examples [00:05, 1039.21 examples/s]5606 examples [00:05, 1039.65 examples/s]5712 examples [00:05, 1044.48 examples/s]5822 examples [00:05, 1059.76 examples/s]5932 examples [00:05, 1071.17 examples/s]6042 examples [00:05, 1076.63 examples/s]6150 examples [00:05, 1070.32 examples/s]6258 examples [00:06, 1067.98 examples/s]6368 examples [00:06, 1076.20 examples/s]6477 examples [00:06, 1079.35 examples/s]6588 examples [00:06, 1086.42 examples/s]6697 examples [00:06, 1084.67 examples/s]6807 examples [00:06, 1088.52 examples/s]6916 examples [00:06, 1088.89 examples/s]7025 examples [00:06, 1088.89 examples/s]7136 examples [00:06, 1093.33 examples/s]7246 examples [00:06, 1095.03 examples/s]7356 examples [00:07, 1095.26 examples/s]7466 examples [00:07, 1093.48 examples/s]7576 examples [00:07, 1085.53 examples/s]7686 examples [00:07, 1088.91 examples/s]7795 examples [00:07, 1070.14 examples/s]7903 examples [00:07, 1066.24 examples/s]8010 examples [00:07, 1038.13 examples/s]8115 examples [00:07, 1037.96 examples/s]8223 examples [00:07, 1048.92 examples/s]8329 examples [00:07, 1040.25 examples/s]8434 examples [00:08, 1005.24 examples/s]8537 examples [00:08, 1010.69 examples/s]8639 examples [00:08, 1013.21 examples/s]8744 examples [00:08, 1023.83 examples/s]8847 examples [00:08, 1017.10 examples/s]8955 examples [00:08, 1035.13 examples/s]9063 examples [00:08, 1046.75 examples/s]9168 examples [00:08, 1044.93 examples/s]9277 examples [00:08, 1057.92 examples/s]9385 examples [00:08, 1062.11 examples/s]9495 examples [00:09, 1071.15 examples/s]9605 examples [00:09, 1078.36 examples/s]9715 examples [00:09, 1084.28 examples/s]9824 examples [00:09, 1075.82 examples/s]9933 examples [00:09, 1079.02 examples/s]10041 examples [00:09, 1022.47 examples/s]10147 examples [00:09, 1031.22 examples/s]10256 examples [00:09, 1047.79 examples/s]10366 examples [00:09, 1060.66 examples/s]10473 examples [00:10, 1057.51 examples/s]10579 examples [00:10, 1054.45 examples/s]10687 examples [00:10, 1059.98 examples/s]10794 examples [00:10, 1055.77 examples/s]10904 examples [00:10, 1066.31 examples/s]11012 examples [00:10, 1068.25 examples/s]11122 examples [00:10, 1074.76 examples/s]11230 examples [00:10, 1071.45 examples/s]11338 examples [00:10, 1073.25 examples/s]11448 examples [00:10, 1080.22 examples/s]11557 examples [00:11, 1074.12 examples/s]11665 examples [00:11, 1048.73 examples/s]11774 examples [00:11, 1060.05 examples/s]11885 examples [00:11, 1071.80 examples/s]11993 examples [00:11, 1073.27 examples/s]12102 examples [00:11, 1076.04 examples/s]12210 examples [00:11, 1074.29 examples/s]12318 examples [00:11, 1062.67 examples/s]12429 examples [00:11, 1074.93 examples/s]12537 examples [00:11, 1071.48 examples/s]12646 examples [00:12, 1074.70 examples/s]12756 examples [00:12, 1081.31 examples/s]12866 examples [00:12, 1084.12 examples/s]12977 examples [00:12, 1091.69 examples/s]13087 examples [00:12, 1087.14 examples/s]13196 examples [00:12, 1087.62 examples/s]13305 examples [00:12, 1054.42 examples/s]13411 examples [00:12, 1048.06 examples/s]13518 examples [00:12, 1052.68 examples/s]13628 examples [00:12, 1063.67 examples/s]13736 examples [00:13, 1067.33 examples/s]13843 examples [00:13, 1055.10 examples/s]13949 examples [00:13, 1050.20 examples/s]14055 examples [00:13, 1025.94 examples/s]14165 examples [00:13, 1045.36 examples/s]14271 examples [00:13, 1026.04 examples/s]14379 examples [00:13, 1039.59 examples/s]14486 examples [00:13, 1048.12 examples/s]14596 examples [00:13, 1060.58 examples/s]14705 examples [00:13, 1067.85 examples/s]14812 examples [00:14, 1058.94 examples/s]14918 examples [00:14, 1044.56 examples/s]15026 examples [00:14, 1054.45 examples/s]15132 examples [00:14, 1047.73 examples/s]15241 examples [00:14, 1058.41 examples/s]15347 examples [00:14, 1058.21 examples/s]15454 examples [00:14, 1059.68 examples/s]15561 examples [00:14, 1051.91 examples/s]15670 examples [00:14, 1060.75 examples/s]15777 examples [00:14, 1059.75 examples/s]15884 examples [00:15, 1047.91 examples/s]15992 examples [00:15, 1055.40 examples/s]16100 examples [00:15, 1060.52 examples/s]16208 examples [00:15, 1065.85 examples/s]16315 examples [00:15, 1055.02 examples/s]16423 examples [00:15, 1061.76 examples/s]16533 examples [00:15, 1071.43 examples/s]16642 examples [00:15, 1074.15 examples/s]16750 examples [00:15, 1074.02 examples/s]16858 examples [00:16, 1063.38 examples/s]16965 examples [00:16, 1060.20 examples/s]17076 examples [00:16, 1073.45 examples/s]17184 examples [00:16, 1075.10 examples/s]17295 examples [00:16, 1085.05 examples/s]17404 examples [00:16, 1085.65 examples/s]17513 examples [00:16, 1083.25 examples/s]17622 examples [00:16, 1053.00 examples/s]17731 examples [00:16, 1062.47 examples/s]17838 examples [00:16, 1062.44 examples/s]17948 examples [00:17, 1071.72 examples/s]18056 examples [00:17, 1064.58 examples/s]18163 examples [00:17, 1033.51 examples/s]18271 examples [00:17, 1046.07 examples/s]18376 examples [00:17, 1033.17 examples/s]18486 examples [00:17, 1049.92 examples/s]18594 examples [00:17, 1057.79 examples/s]18700 examples [00:17, 1056.13 examples/s]18808 examples [00:17, 1062.50 examples/s]18916 examples [00:17, 1066.39 examples/s]19023 examples [00:18, 1024.41 examples/s]19130 examples [00:18, 1035.91 examples/s]19239 examples [00:18, 1049.83 examples/s]19347 examples [00:18, 1058.18 examples/s]19457 examples [00:18, 1068.39 examples/s]19566 examples [00:18, 1073.17 examples/s]19674 examples [00:18, 1065.45 examples/s]19782 examples [00:18, 1068.72 examples/s]19892 examples [00:18, 1077.13 examples/s]20001 examples [00:18, 1028.44 examples/s]20111 examples [00:19, 1047.32 examples/s]20217 examples [00:19, 1046.04 examples/s]20325 examples [00:19, 1054.50 examples/s]20431 examples [00:19, 1046.05 examples/s]20540 examples [00:19, 1056.14 examples/s]20651 examples [00:19, 1069.16 examples/s]20759 examples [00:19, 1069.48 examples/s]20867 examples [00:19, 1062.31 examples/s]20976 examples [00:19, 1068.40 examples/s]21083 examples [00:20, 1068.61 examples/s]21190 examples [00:20, 1064.80 examples/s]21297 examples [00:20, 1037.18 examples/s]21401 examples [00:20, 996.36 examples/s] 21508 examples [00:20, 1016.42 examples/s]21618 examples [00:20, 1037.38 examples/s]21727 examples [00:20, 1051.87 examples/s]21834 examples [00:20, 1055.55 examples/s]21941 examples [00:20, 1057.89 examples/s]22050 examples [00:20, 1065.25 examples/s]22157 examples [00:21, 1049.76 examples/s]22266 examples [00:21, 1059.60 examples/s]22373 examples [00:21, 1058.09 examples/s]22483 examples [00:21, 1068.79 examples/s]22593 examples [00:21, 1075.20 examples/s]22701 examples [00:21, 1071.99 examples/s]22810 examples [00:21, 1076.19 examples/s]22918 examples [00:21, 1067.82 examples/s]23026 examples [00:21, 1068.63 examples/s]23136 examples [00:21, 1076.39 examples/s]23245 examples [00:22, 1080.17 examples/s]23354 examples [00:22, 1075.98 examples/s]23462 examples [00:22, 1069.45 examples/s]23572 examples [00:22, 1076.06 examples/s]23680 examples [00:22, 1064.53 examples/s]23788 examples [00:22, 1067.42 examples/s]23895 examples [00:22, 1055.10 examples/s]24001 examples [00:22, 1029.45 examples/s]24108 examples [00:22, 1038.98 examples/s]24213 examples [00:22, 1036.00 examples/s]24320 examples [00:23, 1045.87 examples/s]24428 examples [00:23, 1052.84 examples/s]24534 examples [00:23, 1021.58 examples/s]24640 examples [00:23, 1032.57 examples/s]24749 examples [00:23, 1048.55 examples/s]24856 examples [00:23, 1052.34 examples/s]24966 examples [00:23, 1064.80 examples/s]25073 examples [00:23, 1063.03 examples/s]25180 examples [00:23, 1044.02 examples/s]25287 examples [00:23, 1050.42 examples/s]25395 examples [00:24, 1059.05 examples/s]25504 examples [00:24, 1065.86 examples/s]25611 examples [00:24, 1056.55 examples/s]25717 examples [00:24, 1048.75 examples/s]25826 examples [00:24, 1058.90 examples/s]25936 examples [00:24, 1069.80 examples/s]26045 examples [00:24, 1074.15 examples/s]26153 examples [00:24, 1063.83 examples/s]26262 examples [00:24, 1068.72 examples/s]26371 examples [00:25, 1072.67 examples/s]26479 examples [00:25, 1066.71 examples/s]26586 examples [00:25, 1066.77 examples/s]26693 examples [00:25, 1066.87 examples/s]26802 examples [00:25, 1071.84 examples/s]26911 examples [00:25, 1074.73 examples/s]27019 examples [00:25, 1051.05 examples/s]27125 examples [00:25, 1035.78 examples/s]27229 examples [00:25, 1031.34 examples/s]27333 examples [00:25, 1024.53 examples/s]27436 examples [00:26, 1019.41 examples/s]27539 examples [00:26, 1020.90 examples/s]27647 examples [00:26, 1035.86 examples/s]27751 examples [00:26, 1008.31 examples/s]27854 examples [00:26, 1012.96 examples/s]27960 examples [00:26, 1024.21 examples/s]28066 examples [00:26, 1034.46 examples/s]28174 examples [00:26, 1045.91 examples/s]28281 examples [00:26, 1051.24 examples/s]28389 examples [00:26, 1057.21 examples/s]28495 examples [00:27, 1055.48 examples/s]28602 examples [00:27, 1058.12 examples/s]28709 examples [00:27, 1060.18 examples/s]28817 examples [00:27, 1065.44 examples/s]28924 examples [00:27, 1055.77 examples/s]29034 examples [00:27, 1066.61 examples/s]29142 examples [00:27, 1070.38 examples/s]29250 examples [00:27, 1062.10 examples/s]29357 examples [00:27, 1064.19 examples/s]29465 examples [00:27, 1067.74 examples/s]29574 examples [00:28, 1072.61 examples/s]29682 examples [00:28, 1058.48 examples/s]29788 examples [00:28, 1052.11 examples/s]29895 examples [00:28, 1056.54 examples/s]30001 examples [00:28, 1006.86 examples/s]30103 examples [00:28, 998.67 examples/s] 30204 examples [00:28, 1000.30 examples/s]30307 examples [00:28, 1007.84 examples/s]30411 examples [00:28, 1016.60 examples/s]30519 examples [00:28, 1032.31 examples/s]30627 examples [00:29, 1045.28 examples/s]30737 examples [00:29, 1060.00 examples/s]30844 examples [00:29, 1061.03 examples/s]30951 examples [00:29, 1052.98 examples/s]31057 examples [00:29, 1048.03 examples/s]31162 examples [00:29, 1040.32 examples/s]31271 examples [00:29, 1053.18 examples/s]31377 examples [00:29, 1052.15 examples/s]31484 examples [00:29, 1054.56 examples/s]31591 examples [00:29, 1058.24 examples/s]31697 examples [00:30, 1047.52 examples/s]31805 examples [00:30, 1055.33 examples/s]31914 examples [00:30, 1064.32 examples/s]32021 examples [00:30, 1057.94 examples/s]32129 examples [00:30, 1063.26 examples/s]32237 examples [00:30, 1067.36 examples/s]32346 examples [00:30, 1072.16 examples/s]32454 examples [00:30, 1035.59 examples/s]32561 examples [00:30, 1045.29 examples/s]32668 examples [00:31, 1051.66 examples/s]32777 examples [00:31, 1061.80 examples/s]32885 examples [00:31, 1066.72 examples/s]32993 examples [00:31, 1068.14 examples/s]33100 examples [00:31, 1068.61 examples/s]33207 examples [00:31, 1060.96 examples/s]33315 examples [00:31, 1064.44 examples/s]33422 examples [00:31, 1063.67 examples/s]33530 examples [00:31, 1066.33 examples/s]33638 examples [00:31, 1068.08 examples/s]33746 examples [00:32, 1068.72 examples/s]33855 examples [00:32, 1072.23 examples/s]33963 examples [00:32, 1072.23 examples/s]34073 examples [00:32, 1080.32 examples/s]34182 examples [00:32, 1066.88 examples/s]34289 examples [00:32, 1067.71 examples/s]34396 examples [00:32, 1004.61 examples/s]34498 examples [00:32, 1001.23 examples/s]34599 examples [00:32, 989.21 examples/s] 34705 examples [00:32, 1009.43 examples/s]34812 examples [00:33, 1025.80 examples/s]34921 examples [00:33, 1041.70 examples/s]35027 examples [00:33, 1045.02 examples/s]35133 examples [00:33, 1048.75 examples/s]35239 examples [00:33, 1045.80 examples/s]35344 examples [00:33, 1046.90 examples/s]35449 examples [00:33, 1045.69 examples/s]35556 examples [00:33, 1051.66 examples/s]35664 examples [00:33, 1059.27 examples/s]35772 examples [00:33, 1062.88 examples/s]35880 examples [00:34, 1067.15 examples/s]35987 examples [00:34, 1066.78 examples/s]36095 examples [00:34, 1067.89 examples/s]36202 examples [00:34, 1060.46 examples/s]36309 examples [00:34, 1045.18 examples/s]36416 examples [00:34, 1050.56 examples/s]36522 examples [00:34, 1042.26 examples/s]36627 examples [00:34, 1035.07 examples/s]36735 examples [00:34, 1045.27 examples/s]36840 examples [00:34, 1044.96 examples/s]36945 examples [00:35, 1044.06 examples/s]37050 examples [00:35, 1040.64 examples/s]37159 examples [00:35, 1054.28 examples/s]37268 examples [00:35, 1064.52 examples/s]37375 examples [00:35, 1062.77 examples/s]37483 examples [00:35, 1067.33 examples/s]37590 examples [00:35, 1039.41 examples/s]37696 examples [00:35, 1045.07 examples/s]37804 examples [00:35, 1055.28 examples/s]37911 examples [00:36, 1058.97 examples/s]38017 examples [00:36, 1058.23 examples/s]38125 examples [00:36, 1063.85 examples/s]38232 examples [00:36, 1063.35 examples/s]38340 examples [00:36, 1068.09 examples/s]38448 examples [00:36, 1069.52 examples/s]38555 examples [00:36, 1064.64 examples/s]38663 examples [00:36, 1067.74 examples/s]38770 examples [00:36, 1064.32 examples/s]38877 examples [00:36, 1059.72 examples/s]38984 examples [00:37, 1061.48 examples/s]39091 examples [00:37, 1059.75 examples/s]39197 examples [00:37, 1055.75 examples/s]39304 examples [00:37, 1058.75 examples/s]39412 examples [00:37, 1062.51 examples/s]39519 examples [00:37, 1035.20 examples/s]39623 examples [00:37, 1036.32 examples/s]39727 examples [00:37, 1031.78 examples/s]39831 examples [00:37, 1031.60 examples/s]39940 examples [00:37, 1047.73 examples/s]40045 examples [00:38, 1003.97 examples/s]40155 examples [00:38, 1028.71 examples/s]40264 examples [00:38, 1045.65 examples/s]40371 examples [00:38, 1050.65 examples/s]40481 examples [00:38, 1064.09 examples/s]40588 examples [00:38, 1056.63 examples/s]40694 examples [00:38, 1050.54 examples/s]40800 examples [00:38, 1031.31 examples/s]40906 examples [00:38, 1038.75 examples/s]41015 examples [00:38, 1051.62 examples/s]41124 examples [00:39, 1062.67 examples/s]41231 examples [00:39, 1063.39 examples/s]41340 examples [00:39, 1070.31 examples/s]41448 examples [00:39, 1067.39 examples/s]41555 examples [00:39, 1057.53 examples/s]41664 examples [00:39, 1065.70 examples/s]41772 examples [00:39, 1067.75 examples/s]41881 examples [00:39, 1072.28 examples/s]41989 examples [00:39, 1056.70 examples/s]42096 examples [00:39, 1060.03 examples/s]42203 examples [00:40, 1058.57 examples/s]42311 examples [00:40, 1064.14 examples/s]42418 examples [00:40, 1052.81 examples/s]42525 examples [00:40, 1056.62 examples/s]42633 examples [00:40, 1062.94 examples/s]42740 examples [00:40, 1063.56 examples/s]42848 examples [00:40, 1066.40 examples/s]42955 examples [00:40, 1054.83 examples/s]43062 examples [00:40, 1057.05 examples/s]43170 examples [00:40, 1063.67 examples/s]43278 examples [00:41, 1067.29 examples/s]43385 examples [00:41, 1066.13 examples/s]43492 examples [00:41, 1062.54 examples/s]43599 examples [00:41, 1054.67 examples/s]43705 examples [00:41, 1045.75 examples/s]43812 examples [00:41, 1049.84 examples/s]43918 examples [00:41, 1044.03 examples/s]44023 examples [00:41, 1011.98 examples/s]44126 examples [00:41, 1015.64 examples/s]44228 examples [00:42, 1014.61 examples/s]44333 examples [00:42, 1024.00 examples/s]44437 examples [00:42, 1026.43 examples/s]44546 examples [00:42, 1042.50 examples/s]44651 examples [00:42, 1038.85 examples/s]44758 examples [00:42, 1045.24 examples/s]44867 examples [00:42, 1055.65 examples/s]44973 examples [00:42, 1052.44 examples/s]45079 examples [00:42, 1051.25 examples/s]45187 examples [00:42, 1056.87 examples/s]45295 examples [00:43, 1063.35 examples/s]45402 examples [00:43, 1060.83 examples/s]45509 examples [00:43, 1061.40 examples/s]45619 examples [00:43, 1070.80 examples/s]45727 examples [00:43, 1071.68 examples/s]45835 examples [00:43, 1053.38 examples/s]45942 examples [00:43, 1056.74 examples/s]46048 examples [00:43, 1048.01 examples/s]46153 examples [00:43, 1034.98 examples/s]46258 examples [00:43, 1036.95 examples/s]46366 examples [00:44, 1049.14 examples/s]46475 examples [00:44, 1060.77 examples/s]46582 examples [00:44, 1049.31 examples/s]46690 examples [00:44, 1055.62 examples/s]46796 examples [00:44, 1038.86 examples/s]46903 examples [00:44, 1046.26 examples/s]47010 examples [00:44, 1051.05 examples/s]47116 examples [00:44, 1034.59 examples/s]47220 examples [00:44, 1035.64 examples/s]47324 examples [00:44, 1001.32 examples/s]47430 examples [00:45, 1016.70 examples/s]47536 examples [00:45, 1027.72 examples/s]47639 examples [00:45, 1016.52 examples/s]47744 examples [00:45, 1025.87 examples/s]47853 examples [00:45, 1041.80 examples/s]47962 examples [00:45, 1054.59 examples/s]48071 examples [00:45, 1063.30 examples/s]48178 examples [00:45, 1063.87 examples/s]48286 examples [00:45, 1066.73 examples/s]48395 examples [00:45, 1070.20 examples/s]48503 examples [00:46, 1067.96 examples/s]48611 examples [00:46, 1070.55 examples/s]48719 examples [00:46, 1063.22 examples/s]48826 examples [00:46, 1044.28 examples/s]48934 examples [00:46, 1051.45 examples/s]49043 examples [00:46, 1060.18 examples/s]49150 examples [00:46, 1059.68 examples/s]49257 examples [00:46, 1058.81 examples/s]49367 examples [00:46, 1070.04 examples/s]49477 examples [00:46, 1076.64 examples/s]49585 examples [00:47, 1073.32 examples/s]49693 examples [00:47, 1056.54 examples/s]49801 examples [00:47, 1062.02 examples/s]49910 examples [00:47, 1070.02 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 6933/50000 [00:00<00:00, 69329.57 examples/s] 40%|████      | 20075/50000 [00:00<00:00, 80778.28 examples/s] 66%|██████▌   | 32929/50000 [00:00<00:00, 90911.86 examples/s] 92%|█████████▏| 46080/50000 [00:00<00:00, 100189.69 examples/s]                                                                0 examples [00:00, ? examples/s]80 examples [00:00, 798.08 examples/s]176 examples [00:00, 838.97 examples/s]283 examples [00:00, 883.46 examples/s]387 examples [00:00, 924.10 examples/s]497 examples [00:00, 969.81 examples/s]607 examples [00:00, 1003.06 examples/s]718 examples [00:00, 1032.86 examples/s]828 examples [00:00, 1051.71 examples/s]939 examples [00:00, 1068.28 examples/s]1049 examples [00:01, 1076.53 examples/s]1160 examples [00:01, 1085.75 examples/s]1270 examples [00:01, 1088.52 examples/s]1381 examples [00:01, 1094.51 examples/s]1491 examples [00:01, 1095.81 examples/s]1601 examples [00:01, 1075.93 examples/s]1712 examples [00:01, 1084.60 examples/s]1822 examples [00:01, 1086.73 examples/s]1931 examples [00:01, 1084.67 examples/s]2040 examples [00:01, 1080.17 examples/s]2151 examples [00:02, 1087.24 examples/s]2261 examples [00:02, 1090.96 examples/s]2371 examples [00:02, 1085.63 examples/s]2480 examples [00:02, 1083.16 examples/s]2589 examples [00:02, 1078.42 examples/s]2697 examples [00:02, 1078.47 examples/s]2805 examples [00:02, 1068.85 examples/s]2916 examples [00:02, 1080.64 examples/s]3027 examples [00:02, 1087.87 examples/s]3136 examples [00:02, 1054.15 examples/s]3248 examples [00:03, 1070.44 examples/s]3356 examples [00:03, 1053.77 examples/s]3466 examples [00:03, 1065.35 examples/s]3575 examples [00:03, 1071.37 examples/s]3683 examples [00:03, 1067.26 examples/s]3792 examples [00:03, 1073.22 examples/s]3901 examples [00:03, 1076.76 examples/s]4011 examples [00:03, 1083.09 examples/s]4120 examples [00:03, 1058.27 examples/s]4226 examples [00:03, 1056.80 examples/s]4336 examples [00:04, 1068.05 examples/s]4445 examples [00:04, 1072.83 examples/s]4554 examples [00:04, 1076.27 examples/s]4663 examples [00:04, 1079.66 examples/s]4772 examples [00:04, 1081.34 examples/s]4881 examples [00:04, 1068.65 examples/s]4991 examples [00:04, 1076.19 examples/s]5099 examples [00:04, 1076.13 examples/s]5207 examples [00:04, 1071.48 examples/s]5315 examples [00:04, 1063.63 examples/s]5425 examples [00:05, 1072.54 examples/s]5536 examples [00:05, 1080.31 examples/s]5647 examples [00:05, 1086.41 examples/s]5756 examples [00:05, 1083.10 examples/s]5865 examples [00:05, 1072.07 examples/s]5974 examples [00:05, 1075.83 examples/s]6083 examples [00:05, 1078.61 examples/s]6193 examples [00:05, 1084.81 examples/s]6304 examples [00:05, 1090.32 examples/s]6414 examples [00:05, 1087.04 examples/s]6523 examples [00:06, 1075.81 examples/s]6631 examples [00:06, 1050.23 examples/s]6739 examples [00:06, 1056.14 examples/s]6848 examples [00:06, 1065.11 examples/s]6955 examples [00:06, 1057.86 examples/s]7061 examples [00:06, 1054.91 examples/s]7170 examples [00:06, 1064.92 examples/s]7279 examples [00:06, 1071.07 examples/s]7387 examples [00:06, 1067.20 examples/s]7496 examples [00:07, 1073.40 examples/s]7606 examples [00:07, 1081.25 examples/s]7716 examples [00:07, 1084.60 examples/s]7825 examples [00:07, 1074.73 examples/s]7934 examples [00:07, 1076.26 examples/s]8042 examples [00:07, 1038.23 examples/s]8150 examples [00:07, 1050.13 examples/s]8260 examples [00:07, 1063.37 examples/s]8371 examples [00:07, 1076.32 examples/s]8479 examples [00:07, 1075.30 examples/s]8589 examples [00:08, 1081.33 examples/s]8698 examples [00:08, 1083.08 examples/s]8807 examples [00:08, 1067.16 examples/s]8917 examples [00:08, 1075.62 examples/s]9025 examples [00:08, 1073.79 examples/s]9133 examples [00:08, 1075.22 examples/s]9241 examples [00:08, 1072.92 examples/s]9349 examples [00:08, 1072.70 examples/s]9458 examples [00:08, 1077.12 examples/s]9567 examples [00:08, 1079.07 examples/s]9675 examples [00:09, 1060.55 examples/s]9783 examples [00:09, 1065.27 examples/s]9890 examples [00:09, 1021.36 examples/s]9999 examples [00:09, 1040.36 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete2PS53P/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete2PS53P/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fab21c7b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fab21c7b620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fab21c7b620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7faab808d6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7faab81382b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fab21c7b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fab21c7b620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fab21c7b620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7faab8138160>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fab08caf3c8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7faa9f027158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7faa9f027158> 

  function with postional parmater data_info <function split_train_valid at 0x7faa9f027158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7faa9f027268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7faa9f027268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7faa9f027268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=b14649465b6085f97d457197e5607eaf4557c95aa88babb37779cd3d7b50cd47
  Stored in directory: /tmp/pip-ephem-wheel-cache-rryu6pb2/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.0
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:45:46, 11.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:27:57, 15.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:52:46, 22.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:37:25, 31.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.61M/862M [00:01<5:19:23, 44.8kB/s].vector_cache/glove.6B.zip:   1%|          | 8.36M/862M [00:01<3:42:25, 64.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:35:05, 91.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:01<1:47:56, 130kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.8M/862M [00:01<1:15:07, 186kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:01<52:32, 265kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:01<36:38, 378kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:02<25:42, 536kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<18:05, 759kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<12:46, 1.07MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:02<08:57, 1.52MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:02<06:25, 2.11MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<05:25, 2.48MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.2M/862M [00:03<04:04, 3.30MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:03<03:06, 4.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<10:09, 1.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<10:18, 1.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:05<08:22, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:05<06:21, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:05<04:42, 2.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<10:47, 1.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<09:59, 1.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:07<07:57, 1.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:07<06:01, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<06:44, 1.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<09:35, 1.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<12:00, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<10:18, 1.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<07:35, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<05:33, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<11:24, 1.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<10:04, 1.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<07:59, 1.65MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<06:03, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:11<04:27, 2.96MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<12:39, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<13:37, 965kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<11:26, 1.15MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<08:57, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:13<06:25, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:15<11:55, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<11:56, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<10:13, 1.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<08:04, 1.62MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:15<05:52, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:17<07:44, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<08:34, 1.52MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:17<06:41, 1.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:17<04:51, 2.67MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<08:11, 1.58MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:19<08:55, 1.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<07:01, 1.84MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:19<05:04, 2.54MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:21<11:38, 1.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:21<11:16, 1.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<08:33, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:21<06:13, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:23<07:56, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:23<08:32, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:23<06:37, 1.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:23<04:51, 2.63MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:25<07:15, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:25<08:01, 1.59MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:25<06:18, 2.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:35, 2.77MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:07, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<08:39, 1.46MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<06:41, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:56, 2.55MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:34, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<07:42, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<06:02, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<04:28, 2.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:23, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<07:30, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<05:53, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:17, 2.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:53, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<08:24, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<06:36, 1.88MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<04:48, 2.58MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<15:15, 812kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<11:46, 1.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<09:39, 1.27MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<07:02, 1.74MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:34, 1.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<11:15, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<09:17, 1.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<06:51, 1.78MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:14, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:45, 1.57MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<06:06, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<04:25, 2.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<11:41, 1.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<10:53, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<08:09, 1.48MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:55, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:45, 1.55MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<07:59, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<06:13, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<04:31, 2.65MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<14:56, 801kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<13:22, 895kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<10:05, 1.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<07:11, 1.65MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<12:06, 983kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<10:53, 1.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<08:12, 1.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<05:54, 2.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<18:32, 638kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<15:24, 768kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<11:21, 1.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<08:05, 1.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<21:40, 543kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<17:08, 686kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<12:33, 935kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<08:55, 1.31MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<12:42, 920kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<11:53, 983kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:54, 1.31MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:23, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<09:09, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<09:14, 1.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<07:03, 1.65MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:07, 2.26MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:14, 1.60MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:13, 1.60MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:30, 2.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<03:58, 2.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<13:56, 824kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<12:34, 913kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<09:22, 1.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<06:44, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<08:16, 1.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<08:25, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:28, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:39, 2.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:21, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<08:28, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:35, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<04:44, 2.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<11:55, 945kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<10:57, 1.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<08:12, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<05:56, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:25, 1.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:40, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:59, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<04:20, 2.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<13:00, 856kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<11:34, 962kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<08:38, 1.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<06:11, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<13:53, 797kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<12:17, 900kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<09:08, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<06:31, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<09:38, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:28, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:22, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:23, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:54, 1.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:23, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:24, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:09, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<06:30, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<05:02, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<03:38, 2.97MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<11:55, 905kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<09:48, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<07:09, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<05:07, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<19:37, 546kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<17:45, 604kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<13:24, 799kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<09:36, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<09:21, 1.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<08:34, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:28, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:17, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:18, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:53, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:10, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:31, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:21, 2.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:47, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:13, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:07, 2.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:37, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:49, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<03:51, 2.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<02:53, 3.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<05:23, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:38, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:24, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<04:48, 2.13MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:12, 1.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:06, 2.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:34, 2.23MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:01, 2.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<03:54, 2.60MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<02:50, 3.55MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<09:24, 1.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<08:25, 1.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:16, 1.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<04:31, 2.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<08:22, 1.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<07:40, 1.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:49, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:43, 1.74MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:51, 1.70MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:33, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<04:50, 2.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<05:15, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:03, 2.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<02:56, 3.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<12:27, 789kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<10:33, 931kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:49, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<07:05, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<08:37, 1.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:48, 1.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:56, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:47, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:47, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:29, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:45, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<06:57, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:45, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:14, 2.26MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:20, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:32, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:18, 2.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:35, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:58, 1.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<03:54, 2.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:18, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:46, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<03:42, 2.53MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<02:44, 3.41MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:29, 1.70MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:32, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:18, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:33, 2.04MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:51, 1.91MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:49, 2.42MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:12, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:36, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:38, 2.53MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:37, 3.48MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<30:01, 304kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<22:39, 403kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<16:15, 561kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<12:50, 707kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<12:24, 731kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<09:23, 965kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<06:43, 1.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:56, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:32, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:59, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:58, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:10, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:01, 2.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:17, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:39, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:39, 2.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:01, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:28, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:27, 2.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<02:31, 3.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:20, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<08:25, 1.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:35, 1.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<04:48, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:30, 1.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:25, 1.60MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:10, 2.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:21, 1.97MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:17, 1.37MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:12, 1.65MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:47, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:44, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:55, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:49, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:04, 2.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:26, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:29, 2.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<03:50, 2.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:52, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:53, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:36, 2.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:35, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:42, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:36, 2.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:36, 3.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<09:51, 837kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<08:23, 982kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:11, 1.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<04:25, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:00, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:23, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:46, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<03:26, 2.36MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:37, 1.22MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:43, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:14, 1.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:47, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:24, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:56, 1.35MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:30, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:29, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:31, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:27, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:30, 3.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<09:19, 848kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<07:54, 999kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:52, 1.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:25, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<05:10, 1.51MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:54, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:49, 2.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<06:46, 1.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<06:06, 1.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:33, 1.70MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<03:15, 2.36MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<09:13, 834kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<09:19, 825kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<07:13, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<05:10, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:35, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:14, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:57, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:49, 2.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<21:28, 352kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<17:51, 423kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<13:09, 574kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<09:20, 805kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<08:28, 883kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<07:11, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:16, 1.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:45, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<10:14, 724kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<08:27, 877kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<06:10, 1.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<04:24, 1.67MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<07:52, 933kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<06:50, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:02, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:37, 2.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<06:03, 1.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<05:30, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:10, 1.74MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:07, 1.75MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:09, 1.74MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:13, 2.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:27, 2.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:39, 1.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<02:49, 2.52MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:03, 3.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<07:11, 985kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:15, 1.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:40, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:26, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<05:44, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<04:38, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:22, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:08, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:06, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:07, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:15, 3.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<06:23, 1.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:42, 1.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:17, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:08, 1.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:05, 1.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:06, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<02:14, 3.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<10:25, 646kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<08:30, 791kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<06:14, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:28, 1.22MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:59, 1.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:18<03:44, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<02:40, 2.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<07:37, 865kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:29, 1.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<04:46, 1.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<03:25, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:56, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<06:36, 988kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:12, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<03:45, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:17, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:07, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:07, 2.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:15, 2.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:53, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:16, 1.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:55, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:48, 2.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:43, 940kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:48, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:17, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<03:04, 2.04MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<05:55, 1.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<05:17, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:58, 1.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:48, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:58, 1.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:56, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:52, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:25, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<03:27, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:40, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:53, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:16, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:32, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:34, 2.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:20, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:22, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:35, 2.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:51, 3.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<08:18, 711kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<08:01, 736kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<06:03, 973kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<04:19, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:28, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:09, 1.41MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:07, 1.87MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:14, 2.58MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<05:47, 996kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<06:13, 927kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:51, 1.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:30, 1.63MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:56, 1.45MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:44, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:49, 2.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<02:01, 2.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<07:43, 729kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<06:22, 882kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:41, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<04:12, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:52, 1.14MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:54, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:50, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:25, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:20, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:32, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:49, 2.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<10:28, 518kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<08:16, 655kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<05:58, 904kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<04:12, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<09:33, 560kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<07:37, 702kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:33, 960kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:45, 1.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:14, 1.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:09, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:15, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:50, 1.08MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<05:21, 974kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:12, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:02, 1.71MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:27, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:18, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:29, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:47, 2.83MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<06:44, 754kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:37, 902kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:09, 1.22MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:43, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:28, 1.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:38, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:40, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:45, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:06, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:31, 3.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<07:47, 626kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<06:10, 789kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:31, 1.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:12, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:50, 993kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:13, 1.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:07, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<02:12, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<12:46, 371kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<09:45, 485kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<07:01, 672kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<05:39, 825kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<04:46, 978kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:30, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:28, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<21:12, 217kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<16:33, 278kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<11:58, 383kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<08:25, 541kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<07:02, 643kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:42, 792kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<04:10, 1.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:39, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:20, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:31, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:29, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:16, 1.34MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:41, 1.63MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:57, 2.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:28, 1.75MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:31, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:54, 2.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:23, 3.08MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:31, 1.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:05, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:20, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:41, 2.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:44, 1.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:22, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:32, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:27, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:15, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:35, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:53, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:21, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:21, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:47, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:17, 3.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<05:02, 788kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:13, 940kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:05, 1.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<02:11, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:23, 888kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:45, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:45, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<01:57, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<06:55, 555kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<05:31, 693kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:59, 956kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<02:49, 1.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<04:36, 819kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<03:52, 971kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:51, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:36, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:12, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:34, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:52, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:14, 1.62MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:12, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:40, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:11, 2.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<27:15, 131kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<20:23, 175kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<14:33, 244kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<10:10, 346kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<07:59, 438kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<06:11, 564kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<04:26, 782kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<03:07, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<04:35, 745kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:49, 893kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:47, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:58, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<04:29, 747kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:43, 900kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:43, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<01:55, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<05:58, 550kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:45, 691kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:26, 951kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<02:24, 1.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<05:50, 552kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<04:38, 694kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<03:22, 950kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:51, 1.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:32, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:54, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:50, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<02:26, 1.26MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:58, 1.55MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:26, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:47, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:46, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:20, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:57, 3.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<05:02, 584kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<04:03, 726kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:55, 998kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:05, 1.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<02:25, 1.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:11, 1.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:38, 1.74MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<01:09, 2.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<08:21, 336kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<06:21, 441kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<04:31, 616kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<03:09, 869kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:46, 726kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:06, 879kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:15, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:35, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<04:20, 614kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<03:30, 761kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:33, 1.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:11, 1.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:59, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:28, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:03, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<03:21, 753kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:47, 906kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<02:02, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:25, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<05:17, 465kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<04:37, 532kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<03:26, 711kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:25, 997kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:17, 1.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:01, 1.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:29, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:03, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<02:05, 1.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<02:19, 996kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:49, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:19, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:30, 1.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:25, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:04, 2.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:45, 2.87MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:46, 579kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:01, 723kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:10, 994kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<01:31, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:13, 656kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:37, 806kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:54, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:39, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:31, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:09, 1.77MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:07, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:07, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:51, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:36, 3.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:31, 757kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:06, 906kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:32, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<01:04, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:49, 1.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:36, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:11, 1.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:06, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:05, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:49, 2.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:51, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:53, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:41, 2.42MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:45, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:48, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:37, 2.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:26, 3.56MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:12, 711kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:49, 858kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:19, 1.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:09, 1.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:20, 1.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:03, 1.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:45, 1.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:53, 1.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:52, 1.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:39, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:27, 2.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<03:06, 438kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<02:24, 563kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:43, 776kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:23, 933kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:12, 1.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:52, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:52<00:36, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<06:59, 175kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<05:19, 229kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<03:49, 318kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<02:37, 451kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<02:05, 552kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:40, 689kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:11, 950kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:50, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:00, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:53, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:39, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:27, 2.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<02:22, 429kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:50, 552kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:18, 765kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:52, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<02:36, 363kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:59, 475kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<01:24, 658kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:05, 811kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:54, 963kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:39, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:27, 1.82MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:44, 1.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:39, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:29, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:26, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:26, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:20, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:19, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:29, 1.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:23, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:17, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:17, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:18, 1.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:14, 2.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:09, 3.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:20, 1.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:19, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:14, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 2.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:48, 581kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:38, 724kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:27, 996kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:18, 1.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:21, 1.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:18, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:09, 2.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:14, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:13, 1.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:05, 2.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:25, 623kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:22, 680kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:16, 903kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:11, 1.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:14, 502kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:11, 648kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:07, 889kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:03, 1.25MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 773kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 921kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.24MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 325/400000 [00:00<02:02, 3249.91it/s]  0%|          | 1139/400000 [00:00<01:40, 3963.89it/s]  0%|          | 1963/400000 [00:00<01:24, 4694.47it/s]  1%|          | 2782/400000 [00:00<01:13, 5382.93it/s]  1%|          | 3631/400000 [00:00<01:05, 6046.81it/s]  1%|          | 4471/400000 [00:00<00:59, 6600.05it/s]  1%|▏         | 5311/400000 [00:00<00:55, 7052.71it/s]  2%|▏         | 6142/400000 [00:00<00:53, 7386.31it/s]  2%|▏         | 6959/400000 [00:00<00:51, 7602.95it/s]  2%|▏         | 7802/400000 [00:01<00:50, 7832.45it/s]  2%|▏         | 8656/400000 [00:01<00:48, 8030.91it/s]  2%|▏         | 9488/400000 [00:01<00:48, 8114.70it/s]  3%|▎         | 10311/400000 [00:01<00:48, 8094.43it/s]  3%|▎         | 11135/400000 [00:01<00:47, 8135.97it/s]  3%|▎         | 11982/400000 [00:01<00:47, 8232.99it/s]  3%|▎         | 12843/400000 [00:01<00:46, 8340.45it/s]  3%|▎         | 13688/400000 [00:01<00:46, 8371.95it/s]  4%|▎         | 14528/400000 [00:01<00:46, 8360.30it/s]  4%|▍         | 15366/400000 [00:01<00:46, 8290.02it/s]  4%|▍         | 16205/400000 [00:02<00:46, 8319.19it/s]  4%|▍         | 17067/400000 [00:02<00:45, 8405.83it/s]  4%|▍         | 17909/400000 [00:02<00:45, 8337.10it/s]  5%|▍         | 18745/400000 [00:02<00:45, 8343.66it/s]  5%|▍         | 19580/400000 [00:02<00:45, 8324.50it/s]  5%|▌         | 20413/400000 [00:02<00:45, 8312.19it/s]  5%|▌         | 21245/400000 [00:02<00:45, 8313.16it/s]  6%|▌         | 22077/400000 [00:02<00:46, 8182.44it/s]  6%|▌         | 22901/400000 [00:02<00:45, 8198.33it/s]  6%|▌         | 23722/400000 [00:02<00:46, 8116.91it/s]  6%|▌         | 24554/400000 [00:03<00:45, 8174.66it/s]  6%|▋         | 25408/400000 [00:03<00:45, 8274.86it/s]  7%|▋         | 26247/400000 [00:03<00:44, 8307.78it/s]  7%|▋         | 27087/400000 [00:03<00:44, 8332.69it/s]  7%|▋         | 27921/400000 [00:03<00:45, 8245.85it/s]  7%|▋         | 28747/400000 [00:03<00:45, 8228.69it/s]  7%|▋         | 29594/400000 [00:03<00:44, 8299.38it/s]  8%|▊         | 30425/400000 [00:03<00:45, 8154.11it/s]  8%|▊         | 31257/400000 [00:03<00:44, 8201.05it/s]  8%|▊         | 32078/400000 [00:03<00:45, 8150.65it/s]  8%|▊         | 32907/400000 [00:04<00:44, 8191.85it/s]  8%|▊         | 33737/400000 [00:04<00:44, 8221.65it/s]  9%|▊         | 34567/400000 [00:04<00:44, 8244.31it/s]  9%|▉         | 35394/400000 [00:04<00:44, 8250.46it/s]  9%|▉         | 36220/400000 [00:04<00:44, 8219.27it/s]  9%|▉         | 37049/400000 [00:04<00:44, 8240.36it/s]  9%|▉         | 37896/400000 [00:04<00:43, 8306.28it/s] 10%|▉         | 38737/400000 [00:04<00:43, 8334.56it/s] 10%|▉         | 39571/400000 [00:04<00:43, 8321.07it/s] 10%|█         | 40404/400000 [00:04<00:43, 8320.71it/s] 10%|█         | 41237/400000 [00:05<00:43, 8171.86it/s] 11%|█         | 42081/400000 [00:05<00:43, 8248.40it/s] 11%|█         | 42922/400000 [00:05<00:43, 8295.56it/s] 11%|█         | 43753/400000 [00:05<00:43, 8210.76it/s] 11%|█         | 44589/400000 [00:05<00:43, 8253.63it/s] 11%|█▏        | 45415/400000 [00:05<00:43, 8126.39it/s] 12%|█▏        | 46244/400000 [00:05<00:43, 8171.99it/s] 12%|█▏        | 47075/400000 [00:05<00:42, 8210.53it/s] 12%|█▏        | 47909/400000 [00:05<00:42, 8246.56it/s] 12%|█▏        | 48738/400000 [00:05<00:42, 8259.10it/s] 12%|█▏        | 49565/400000 [00:06<00:42, 8231.85it/s] 13%|█▎        | 50389/400000 [00:06<00:42, 8193.20it/s] 13%|█▎        | 51220/400000 [00:06<00:42, 8226.24it/s] 13%|█▎        | 52053/400000 [00:06<00:42, 8255.09it/s] 13%|█▎        | 52879/400000 [00:06<00:42, 8176.58it/s] 13%|█▎        | 53697/400000 [00:06<00:42, 8132.37it/s] 14%|█▎        | 54511/400000 [00:06<00:42, 8093.90it/s] 14%|█▍        | 55338/400000 [00:06<00:42, 8145.36it/s] 14%|█▍        | 56163/400000 [00:06<00:42, 8176.23it/s] 14%|█▍        | 56998/400000 [00:06<00:41, 8225.09it/s] 14%|█▍        | 57821/400000 [00:07<00:43, 7901.57it/s] 15%|█▍        | 58662/400000 [00:07<00:42, 8047.21it/s] 15%|█▍        | 59511/400000 [00:07<00:41, 8174.54it/s] 15%|█▌        | 60353/400000 [00:07<00:41, 8246.05it/s] 15%|█▌        | 61183/400000 [00:07<00:41, 8262.01it/s] 16%|█▌        | 62011/400000 [00:07<00:40, 8266.56it/s] 16%|█▌        | 62839/400000 [00:07<00:40, 8260.82it/s] 16%|█▌        | 63681/400000 [00:07<00:40, 8307.32it/s] 16%|█▌        | 64521/400000 [00:07<00:40, 8332.79it/s] 16%|█▋        | 65372/400000 [00:07<00:39, 8383.05it/s] 17%|█▋        | 66211/400000 [00:08<00:39, 8351.67it/s] 17%|█▋        | 67047/400000 [00:08<00:39, 8343.55it/s] 17%|█▋        | 67889/400000 [00:08<00:39, 8365.36it/s] 17%|█▋        | 68729/400000 [00:08<00:39, 8374.47it/s] 17%|█▋        | 69574/400000 [00:08<00:39, 8396.87it/s] 18%|█▊        | 70414/400000 [00:08<00:39, 8372.71it/s] 18%|█▊        | 71252/400000 [00:08<00:39, 8317.62it/s] 18%|█▊        | 72084/400000 [00:08<00:39, 8310.42it/s] 18%|█▊        | 72926/400000 [00:08<00:39, 8340.22it/s] 18%|█▊        | 73772/400000 [00:08<00:38, 8374.81it/s] 19%|█▊        | 74610/400000 [00:09<00:38, 8355.41it/s] 19%|█▉        | 75448/400000 [00:09<00:38, 8362.15it/s] 19%|█▉        | 76290/400000 [00:09<00:38, 8377.29it/s] 19%|█▉        | 77128/400000 [00:09<00:38, 8334.52it/s] 19%|█▉        | 77971/400000 [00:09<00:38, 8361.26it/s] 20%|█▉        | 78808/400000 [00:09<00:38, 8344.07it/s] 20%|█▉        | 79649/400000 [00:09<00:38, 8362.99it/s] 20%|██        | 80497/400000 [00:09<00:38, 8396.65it/s] 20%|██        | 81340/400000 [00:09<00:37, 8405.83it/s] 21%|██        | 82190/400000 [00:09<00:37, 8431.09it/s] 21%|██        | 83034/400000 [00:10<00:38, 8251.39it/s] 21%|██        | 83874/400000 [00:10<00:38, 8294.32it/s] 21%|██        | 84713/400000 [00:10<00:37, 8320.62it/s] 21%|██▏       | 85558/400000 [00:10<00:37, 8356.80it/s] 22%|██▏       | 86398/400000 [00:10<00:37, 8367.04it/s] 22%|██▏       | 87235/400000 [00:10<00:37, 8309.51it/s] 22%|██▏       | 88072/400000 [00:10<00:37, 8326.02it/s] 22%|██▏       | 88905/400000 [00:10<00:37, 8295.01it/s] 22%|██▏       | 89738/400000 [00:10<00:37, 8303.94it/s] 23%|██▎       | 90585/400000 [00:10<00:37, 8350.42it/s] 23%|██▎       | 91426/400000 [00:11<00:36, 8366.48it/s] 23%|██▎       | 92271/400000 [00:11<00:36, 8389.86it/s] 23%|██▎       | 93111/400000 [00:11<00:36, 8348.97it/s] 23%|██▎       | 93958/400000 [00:11<00:36, 8383.93it/s] 24%|██▎       | 94804/400000 [00:11<00:36, 8404.12it/s] 24%|██▍       | 95645/400000 [00:11<00:36, 8399.28it/s] 24%|██▍       | 96485/400000 [00:11<00:36, 8388.50it/s] 24%|██▍       | 97324/400000 [00:11<00:36, 8366.84it/s] 25%|██▍       | 98161/400000 [00:11<00:36, 8316.67it/s] 25%|██▍       | 99001/400000 [00:12<00:36, 8339.51it/s] 25%|██▍       | 99836/400000 [00:12<00:36, 8205.45it/s] 25%|██▌       | 100672/400000 [00:12<00:36, 8251.01it/s] 25%|██▌       | 101498/400000 [00:12<00:36, 8240.69it/s] 26%|██▌       | 102347/400000 [00:12<00:35, 8312.35it/s] 26%|██▌       | 103193/400000 [00:12<00:35, 8353.40it/s] 26%|██▌       | 104029/400000 [00:12<00:35, 8343.71it/s] 26%|██▌       | 104875/400000 [00:12<00:35, 8376.68it/s] 26%|██▋       | 105713/400000 [00:12<00:35, 8337.76it/s] 27%|██▋       | 106559/400000 [00:12<00:35, 8371.42it/s] 27%|██▋       | 107408/400000 [00:13<00:34, 8405.87it/s] 27%|██▋       | 108249/400000 [00:13<00:35, 8285.05it/s] 27%|██▋       | 109093/400000 [00:13<00:34, 8329.44it/s] 27%|██▋       | 109927/400000 [00:13<00:34, 8300.47it/s] 28%|██▊       | 110767/400000 [00:13<00:34, 8328.59it/s] 28%|██▊       | 111612/400000 [00:13<00:34, 8363.46it/s] 28%|██▊       | 112449/400000 [00:13<00:34, 8306.92it/s] 28%|██▊       | 113280/400000 [00:13<00:34, 8216.90it/s] 29%|██▊       | 114103/400000 [00:13<00:34, 8207.86it/s] 29%|██▊       | 114958/400000 [00:13<00:34, 8305.10it/s] 29%|██▉       | 115799/400000 [00:14<00:34, 8335.14it/s] 29%|██▉       | 116633/400000 [00:14<00:34, 8331.32it/s] 29%|██▉       | 117467/400000 [00:14<00:33, 8325.32it/s] 30%|██▉       | 118304/400000 [00:14<00:33, 8338.08it/s] 30%|██▉       | 119138/400000 [00:14<00:33, 8314.56it/s] 30%|██▉       | 119990/400000 [00:14<00:33, 8373.75it/s] 30%|███       | 120829/400000 [00:14<00:33, 8375.38it/s] 30%|███       | 121667/400000 [00:14<00:33, 8337.07it/s] 31%|███       | 122507/400000 [00:14<00:33, 8353.78it/s] 31%|███       | 123343/400000 [00:14<00:33, 8352.26it/s] 31%|███       | 124181/400000 [00:15<00:32, 8358.93it/s] 31%|███▏      | 125026/400000 [00:15<00:32, 8384.12it/s] 31%|███▏      | 125865/400000 [00:15<00:33, 8207.58it/s] 32%|███▏      | 126704/400000 [00:15<00:33, 8258.77it/s] 32%|███▏      | 127531/400000 [00:15<00:32, 8261.08it/s] 32%|███▏      | 128371/400000 [00:15<00:32, 8300.59it/s] 32%|███▏      | 129213/400000 [00:15<00:32, 8335.58it/s] 33%|███▎      | 130047/400000 [00:15<00:32, 8288.11it/s] 33%|███▎      | 130891/400000 [00:15<00:32, 8330.99it/s] 33%|███▎      | 131725/400000 [00:15<00:32, 8209.18it/s] 33%|███▎      | 132572/400000 [00:16<00:32, 8285.14it/s] 33%|███▎      | 133414/400000 [00:16<00:32, 8324.41it/s] 34%|███▎      | 134247/400000 [00:16<00:32, 8277.32it/s] 34%|███▍      | 135103/400000 [00:16<00:31, 8357.87it/s] 34%|███▍      | 135940/400000 [00:16<00:31, 8337.79it/s] 34%|███▍      | 136794/400000 [00:16<00:31, 8396.61it/s] 34%|███▍      | 137641/400000 [00:16<00:31, 8415.60it/s] 35%|███▍      | 138483/400000 [00:16<00:31, 8265.40it/s] 35%|███▍      | 139319/400000 [00:16<00:31, 8292.53it/s] 35%|███▌      | 140158/400000 [00:16<00:31, 8320.06it/s] 35%|███▌      | 140998/400000 [00:17<00:31, 8342.59it/s] 35%|███▌      | 141842/400000 [00:17<00:30, 8369.62it/s] 36%|███▌      | 142680/400000 [00:17<00:30, 8335.66it/s] 36%|███▌      | 143519/400000 [00:17<00:30, 8349.67it/s] 36%|███▌      | 144355/400000 [00:17<00:30, 8344.50it/s] 36%|███▋      | 145208/400000 [00:17<00:30, 8397.79it/s] 37%|███▋      | 146069/400000 [00:17<00:30, 8458.42it/s] 37%|███▋      | 146916/400000 [00:17<00:30, 8404.87it/s] 37%|███▋      | 147757/400000 [00:17<00:30, 8401.55it/s] 37%|███▋      | 148598/400000 [00:17<00:30, 8340.72it/s] 37%|███▋      | 149446/400000 [00:18<00:29, 8379.37it/s] 38%|███▊      | 150285/400000 [00:18<00:29, 8341.13it/s] 38%|███▊      | 151120/400000 [00:18<00:29, 8313.90it/s] 38%|███▊      | 151961/400000 [00:18<00:29, 8339.75it/s] 38%|███▊      | 152799/400000 [00:18<00:29, 8348.88it/s] 38%|███▊      | 153648/400000 [00:18<00:29, 8390.00it/s] 39%|███▊      | 154496/400000 [00:18<00:29, 8415.45it/s] 39%|███▉      | 155338/400000 [00:18<00:30, 8094.20it/s] 39%|███▉      | 156175/400000 [00:18<00:29, 8173.89it/s] 39%|███▉      | 157022/400000 [00:18<00:29, 8258.14it/s] 39%|███▉      | 157862/400000 [00:19<00:29, 8298.48it/s] 40%|███▉      | 158703/400000 [00:19<00:28, 8329.89it/s] 40%|███▉      | 159537/400000 [00:19<00:28, 8300.57it/s] 40%|████      | 160372/400000 [00:19<00:28, 8313.57it/s] 40%|████      | 161216/400000 [00:19<00:28, 8348.81it/s] 41%|████      | 162052/400000 [00:19<00:28, 8343.30it/s] 41%|████      | 162914/400000 [00:19<00:28, 8423.01it/s] 41%|████      | 163757/400000 [00:19<00:28, 8394.57it/s] 41%|████      | 164606/400000 [00:19<00:27, 8420.48it/s] 41%|████▏     | 165449/400000 [00:19<00:27, 8408.24it/s] 42%|████▏     | 166290/400000 [00:20<00:27, 8370.81it/s] 42%|████▏     | 167150/400000 [00:20<00:27, 8437.22it/s] 42%|████▏     | 167994/400000 [00:20<00:27, 8385.09it/s] 42%|████▏     | 168833/400000 [00:20<00:27, 8382.13it/s] 42%|████▏     | 169681/400000 [00:20<00:27, 8410.41it/s] 43%|████▎     | 170523/400000 [00:20<00:27, 8389.40it/s] 43%|████▎     | 171368/400000 [00:20<00:27, 8406.52it/s] 43%|████▎     | 172209/400000 [00:20<00:27, 8370.73it/s] 43%|████▎     | 173047/400000 [00:20<00:27, 8355.64it/s] 43%|████▎     | 173883/400000 [00:20<00:27, 8279.45it/s] 44%|████▎     | 174720/400000 [00:21<00:27, 8304.04it/s] 44%|████▍     | 175551/400000 [00:21<00:27, 8305.39it/s] 44%|████▍     | 176382/400000 [00:21<00:26, 8284.50it/s] 44%|████▍     | 177229/400000 [00:21<00:26, 8337.10it/s] 45%|████▍     | 178082/400000 [00:21<00:26, 8393.29it/s] 45%|████▍     | 178922/400000 [00:21<00:26, 8374.82it/s] 45%|████▍     | 179769/400000 [00:21<00:26, 8401.59it/s] 45%|████▌     | 180610/400000 [00:21<00:26, 8325.71it/s] 45%|████▌     | 181449/400000 [00:21<00:26, 8343.18it/s] 46%|████▌     | 182289/400000 [00:21<00:26, 8358.61it/s] 46%|████▌     | 183125/400000 [00:22<00:26, 8315.17it/s] 46%|████▌     | 183977/400000 [00:22<00:25, 8374.97it/s] 46%|████▌     | 184815/400000 [00:22<00:25, 8300.52it/s] 46%|████▋     | 185655/400000 [00:22<00:25, 8329.55it/s] 47%|████▋     | 186492/400000 [00:22<00:25, 8340.54it/s] 47%|████▋     | 187327/400000 [00:22<00:25, 8334.06it/s] 47%|████▋     | 188175/400000 [00:22<00:25, 8375.12it/s] 47%|████▋     | 189013/400000 [00:22<00:25, 8349.71it/s] 47%|████▋     | 189849/400000 [00:22<00:25, 8347.06it/s] 48%|████▊     | 190689/400000 [00:23<00:25, 8362.57it/s] 48%|████▊     | 191526/400000 [00:23<00:24, 8357.20it/s] 48%|████▊     | 192370/400000 [00:23<00:24, 8379.84it/s] 48%|████▊     | 193209/400000 [00:23<00:24, 8331.48it/s] 49%|████▊     | 194048/400000 [00:23<00:24, 8346.71it/s] 49%|████▊     | 194887/400000 [00:23<00:24, 8359.27it/s] 49%|████▉     | 195723/400000 [00:23<00:24, 8341.73it/s] 49%|████▉     | 196558/400000 [00:23<00:24, 8269.81it/s] 49%|████▉     | 197387/400000 [00:23<00:24, 8275.56it/s] 50%|████▉     | 198215/400000 [00:23<00:24, 8252.40it/s] 50%|████▉     | 199060/400000 [00:24<00:24, 8308.76it/s] 50%|████▉     | 199896/400000 [00:24<00:24, 8322.15it/s] 50%|█████     | 200733/400000 [00:24<00:23, 8334.25it/s] 50%|█████     | 201567/400000 [00:24<00:23, 8325.38it/s] 51%|█████     | 202400/400000 [00:24<00:23, 8236.62it/s] 51%|█████     | 203251/400000 [00:24<00:23, 8316.15it/s] 51%|█████     | 204083/400000 [00:24<00:23, 8281.77it/s] 51%|█████     | 204912/400000 [00:24<00:23, 8271.72it/s] 51%|█████▏    | 205753/400000 [00:24<00:23, 8310.37it/s] 52%|█████▏    | 206585/400000 [00:24<00:24, 8022.64it/s] 52%|█████▏    | 207424/400000 [00:25<00:23, 8127.58it/s] 52%|█████▏    | 208254/400000 [00:25<00:23, 8177.46it/s] 52%|█████▏    | 209092/400000 [00:25<00:23, 8235.35it/s] 52%|█████▏    | 209937/400000 [00:25<00:22, 8297.44it/s] 53%|█████▎    | 210769/400000 [00:25<00:22, 8301.93it/s] 53%|█████▎    | 211615/400000 [00:25<00:22, 8346.52it/s] 53%|█████▎    | 212463/400000 [00:25<00:22, 8383.16it/s] 53%|█████▎    | 213302/400000 [00:25<00:22, 8339.17it/s] 54%|█████▎    | 214151/400000 [00:25<00:22, 8383.77it/s] 54%|█████▎    | 214990/400000 [00:25<00:22, 8341.26it/s] 54%|█████▍    | 215837/400000 [00:26<00:21, 8379.09it/s] 54%|█████▍    | 216687/400000 [00:26<00:21, 8413.04it/s] 54%|█████▍    | 217529/400000 [00:26<00:21, 8339.15it/s] 55%|█████▍    | 218388/400000 [00:26<00:21, 8410.70it/s] 55%|█████▍    | 219230/400000 [00:26<00:21, 8321.35it/s] 55%|█████▌    | 220067/400000 [00:26<00:21, 8335.10it/s] 55%|█████▌    | 220912/400000 [00:26<00:21, 8368.20it/s] 55%|█████▌    | 221750/400000 [00:26<00:21, 8265.09it/s] 56%|█████▌    | 222606/400000 [00:26<00:21, 8350.87it/s] 56%|█████▌    | 223442/400000 [00:26<00:21, 8314.66it/s] 56%|█████▌    | 224285/400000 [00:27<00:21, 8348.03it/s] 56%|█████▋    | 225121/400000 [00:27<00:21, 8274.71it/s] 56%|█████▋    | 225954/400000 [00:27<00:20, 8289.43it/s] 57%|█████▋    | 226795/400000 [00:27<00:20, 8323.28it/s] 57%|█████▋    | 227628/400000 [00:27<00:20, 8290.56it/s] 57%|█████▋    | 228467/400000 [00:27<00:20, 8319.96it/s] 57%|█████▋    | 229306/400000 [00:27<00:20, 8338.84it/s] 58%|█████▊    | 230141/400000 [00:27<00:20, 8340.07it/s] 58%|█████▊    | 230993/400000 [00:27<00:20, 8392.95it/s] 58%|█████▊    | 231833/400000 [00:27<00:20, 8324.49it/s] 58%|█████▊    | 232676/400000 [00:28<00:20, 8355.83it/s] 58%|█████▊    | 233518/400000 [00:28<00:19, 8372.72it/s] 59%|█████▊    | 234356/400000 [00:28<00:19, 8353.06it/s] 59%|█████▉    | 235204/400000 [00:28<00:19, 8389.48it/s] 59%|█████▉    | 236044/400000 [00:28<00:19, 8296.28it/s] 59%|█████▉    | 236894/400000 [00:28<00:19, 8355.09it/s] 59%|█████▉    | 237732/400000 [00:28<00:19, 8360.72it/s] 60%|█████▉    | 238569/400000 [00:28<00:19, 8241.67it/s] 60%|█████▉    | 239410/400000 [00:28<00:19, 8289.75it/s] 60%|██████    | 240240/400000 [00:28<00:19, 8247.18it/s] 60%|██████    | 241079/400000 [00:29<00:19, 8288.22it/s] 60%|██████    | 241916/400000 [00:29<00:19, 8311.22it/s] 61%|██████    | 242748/400000 [00:29<00:18, 8313.42it/s] 61%|██████    | 243580/400000 [00:29<00:18, 8289.31it/s] 61%|██████    | 244410/400000 [00:29<00:18, 8290.52it/s] 61%|██████▏   | 245247/400000 [00:29<00:18, 8312.72it/s] 62%|██████▏   | 246084/400000 [00:29<00:18, 8329.40it/s] 62%|██████▏   | 246921/400000 [00:29<00:18, 8341.19it/s] 62%|██████▏   | 247756/400000 [00:29<00:18, 8341.53it/s] 62%|██████▏   | 248591/400000 [00:29<00:18, 8285.73it/s] 62%|██████▏   | 249424/400000 [00:30<00:18, 8297.88it/s] 63%|██████▎   | 250261/400000 [00:30<00:17, 8319.09it/s] 63%|██████▎   | 251097/400000 [00:30<00:17, 8330.25it/s] 63%|██████▎   | 251931/400000 [00:30<00:17, 8277.37it/s] 63%|██████▎   | 252759/400000 [00:30<00:17, 8266.30it/s] 63%|██████▎   | 253598/400000 [00:30<00:17, 8302.91it/s] 64%|██████▎   | 254429/400000 [00:30<00:17, 8278.82it/s] 64%|██████▍   | 255270/400000 [00:30<00:17, 8315.44it/s] 64%|██████▍   | 256102/400000 [00:30<00:17, 8313.08it/s] 64%|██████▍   | 256936/400000 [00:30<00:17, 8320.24it/s] 64%|██████▍   | 257770/400000 [00:31<00:17, 8323.90it/s] 65%|██████▍   | 258614/400000 [00:31<00:16, 8356.03it/s] 65%|██████▍   | 259450/400000 [00:31<00:17, 8203.70it/s] 65%|██████▌   | 260272/400000 [00:31<00:17, 8111.26it/s] 65%|██████▌   | 261099/400000 [00:31<00:17, 8157.35it/s] 65%|██████▌   | 261926/400000 [00:31<00:16, 8189.24it/s] 66%|██████▌   | 262746/400000 [00:31<00:16, 8180.12it/s] 66%|██████▌   | 263575/400000 [00:31<00:16, 8212.48it/s] 66%|██████▌   | 264399/400000 [00:31<00:16, 8218.81it/s] 66%|██████▋   | 265234/400000 [00:31<00:16, 8256.74it/s] 67%|██████▋   | 266060/400000 [00:32<00:16, 8228.53it/s] 67%|██████▋   | 266906/400000 [00:32<00:16, 8296.07it/s] 67%|██████▋   | 267736/400000 [00:32<00:16, 8182.41it/s] 67%|██████▋   | 268574/400000 [00:32<00:15, 8238.33it/s] 67%|██████▋   | 269417/400000 [00:32<00:15, 8294.25it/s] 68%|██████▊   | 270247/400000 [00:32<00:15, 8269.75it/s] 68%|██████▊   | 271086/400000 [00:32<00:15, 8303.14it/s] 68%|██████▊   | 271917/400000 [00:32<00:15, 8202.67it/s] 68%|██████▊   | 272761/400000 [00:32<00:15, 8272.19it/s] 68%|██████▊   | 273607/400000 [00:32<00:15, 8327.08it/s] 69%|██████▊   | 274441/400000 [00:33<00:15, 8317.34it/s] 69%|██████▉   | 275284/400000 [00:33<00:14, 8350.64it/s] 69%|██████▉   | 276120/400000 [00:33<00:15, 8184.18it/s] 69%|██████▉   | 276960/400000 [00:33<00:14, 8245.66it/s] 69%|██████▉   | 277796/400000 [00:33<00:14, 8277.69it/s] 70%|██████▉   | 278625/400000 [00:33<00:14, 8228.54it/s] 70%|██████▉   | 279449/400000 [00:33<00:14, 8149.06it/s] 70%|███████   | 280286/400000 [00:33<00:14, 8213.61it/s] 70%|███████   | 281123/400000 [00:33<00:14, 8257.95it/s] 70%|███████   | 281955/400000 [00:34<00:14, 8275.45it/s] 71%|███████   | 282783/400000 [00:34<00:14, 8262.07it/s] 71%|███████   | 283624/400000 [00:34<00:14, 8305.25it/s] 71%|███████   | 284470/400000 [00:34<00:13, 8349.20it/s] 71%|███████▏  | 285306/400000 [00:34<00:13, 8243.22it/s] 72%|███████▏  | 286153/400000 [00:34<00:13, 8307.85it/s] 72%|███████▏  | 286985/400000 [00:34<00:13, 8288.67it/s] 72%|███████▏  | 287839/400000 [00:34<00:13, 8359.73it/s] 72%|███████▏  | 288685/400000 [00:34<00:13, 8387.08it/s] 72%|███████▏  | 289524/400000 [00:34<00:13, 8361.00it/s] 73%|███████▎  | 290361/400000 [00:35<00:13, 8334.05it/s] 73%|███████▎  | 291198/400000 [00:35<00:13, 8342.14it/s] 73%|███████▎  | 292044/400000 [00:35<00:12, 8376.17it/s] 73%|███████▎  | 292882/400000 [00:35<00:12, 8312.48it/s] 73%|███████▎  | 293720/400000 [00:35<00:12, 8330.00it/s] 74%|███████▎  | 294570/400000 [00:35<00:12, 8380.25it/s] 74%|███████▍  | 295409/400000 [00:35<00:12, 8326.36it/s] 74%|███████▍  | 296249/400000 [00:35<00:12, 8346.00it/s] 74%|███████▍  | 297097/400000 [00:35<00:12, 8383.98it/s] 74%|███████▍  | 297946/400000 [00:35<00:12, 8414.27it/s] 75%|███████▍  | 298788/400000 [00:36<00:12, 8407.40it/s] 75%|███████▍  | 299629/400000 [00:36<00:12, 8357.49it/s] 75%|███████▌  | 300472/400000 [00:36<00:11, 8377.27it/s] 75%|███████▌  | 301323/400000 [00:36<00:11, 8415.54it/s] 76%|███████▌  | 302170/400000 [00:36<00:11, 8431.49it/s] 76%|███████▌  | 303014/400000 [00:36<00:11, 8393.37it/s] 76%|███████▌  | 303854/400000 [00:36<00:11, 8334.42it/s] 76%|███████▌  | 304698/400000 [00:36<00:11, 8363.10it/s] 76%|███████▋  | 305535/400000 [00:36<00:11, 8302.52it/s] 77%|███████▋  | 306387/400000 [00:36<00:11, 8365.27it/s] 77%|███████▋  | 307229/400000 [00:37<00:11, 8379.70it/s] 77%|███████▋  | 308068/400000 [00:37<00:11, 8338.50it/s] 77%|███████▋  | 308903/400000 [00:37<00:11, 8198.76it/s] 77%|███████▋  | 309724/400000 [00:37<00:11, 8089.30it/s] 78%|███████▊  | 310543/400000 [00:37<00:11, 8117.76it/s] 78%|███████▊  | 311389/400000 [00:37<00:10, 8215.23it/s] 78%|███████▊  | 312213/400000 [00:37<00:10, 8220.35it/s] 78%|███████▊  | 313053/400000 [00:37<00:10, 8272.52it/s] 78%|███████▊  | 313887/400000 [00:37<00:10, 8292.25it/s] 79%|███████▊  | 314741/400000 [00:37<00:10, 8364.70it/s] 79%|███████▉  | 315578/400000 [00:38<00:10, 8352.73it/s] 79%|███████▉  | 316414/400000 [00:38<00:10, 8322.76it/s] 79%|███████▉  | 317250/400000 [00:38<00:09, 8333.28it/s] 80%|███████▉  | 318084/400000 [00:38<00:09, 8333.16it/s] 80%|███████▉  | 318923/400000 [00:38<00:09, 8349.44it/s] 80%|███████▉  | 319759/400000 [00:38<00:09, 8290.25it/s] 80%|████████  | 320589/400000 [00:38<00:09, 8277.48it/s] 80%|████████  | 321417/400000 [00:38<00:09, 8239.60it/s] 81%|████████  | 322250/400000 [00:38<00:09, 8265.17it/s] 81%|████████  | 323089/400000 [00:38<00:09, 8300.93it/s] 81%|████████  | 323920/400000 [00:39<00:09, 8256.72it/s] 81%|████████  | 324746/400000 [00:39<00:09, 8249.43it/s] 81%|████████▏ | 325572/400000 [00:39<00:09, 8244.87it/s] 82%|████████▏ | 326408/400000 [00:39<00:08, 8276.53it/s] 82%|████████▏ | 327236/400000 [00:39<00:08, 8260.87it/s] 82%|████████▏ | 328063/400000 [00:39<00:08, 8255.44it/s] 82%|████████▏ | 328904/400000 [00:39<00:08, 8297.85it/s] 82%|████████▏ | 329734/400000 [00:39<00:08, 8255.39it/s] 83%|████████▎ | 330573/400000 [00:39<00:08, 8293.01it/s] 83%|████████▎ | 331403/400000 [00:39<00:08, 8277.54it/s] 83%|████████▎ | 332231/400000 [00:40<00:08, 8259.83it/s] 83%|████████▎ | 333059/400000 [00:40<00:08, 8265.53it/s] 83%|████████▎ | 333886/400000 [00:40<00:08, 8238.11it/s] 84%|████████▎ | 334727/400000 [00:40<00:07, 8286.84it/s] 84%|████████▍ | 335561/400000 [00:40<00:07, 8301.92it/s] 84%|████████▍ | 336399/400000 [00:40<00:07, 8323.64it/s] 84%|████████▍ | 337233/400000 [00:40<00:07, 8328.51it/s] 85%|████████▍ | 338066/400000 [00:40<00:07, 8312.15it/s] 85%|████████▍ | 338906/400000 [00:40<00:07, 8336.78it/s] 85%|████████▍ | 339746/400000 [00:40<00:07, 8355.61it/s] 85%|████████▌ | 340594/400000 [00:41<00:07, 8391.10it/s] 85%|████████▌ | 341434/400000 [00:41<00:06, 8384.47it/s] 86%|████████▌ | 342273/400000 [00:41<00:06, 8374.13it/s] 86%|████████▌ | 343111/400000 [00:41<00:06, 8333.69it/s] 86%|████████▌ | 343947/400000 [00:41<00:06, 8341.06it/s] 86%|████████▌ | 344782/400000 [00:41<00:06, 8338.47it/s] 86%|████████▋ | 345616/400000 [00:41<00:06, 8316.29it/s] 87%|████████▋ | 346460/400000 [00:41<00:06, 8350.14it/s] 87%|████████▋ | 347296/400000 [00:41<00:06, 8303.57it/s] 87%|████████▋ | 348127/400000 [00:41<00:06, 8284.68it/s] 87%|████████▋ | 348968/400000 [00:42<00:06, 8319.16it/s] 87%|████████▋ | 349801/400000 [00:42<00:06, 8313.94it/s] 88%|████████▊ | 350633/400000 [00:42<00:05, 8300.33it/s] 88%|████████▊ | 351473/400000 [00:42<00:05, 8328.93it/s] 88%|████████▊ | 352306/400000 [00:42<00:05, 8257.39it/s] 88%|████████▊ | 353153/400000 [00:42<00:05, 8319.88it/s] 88%|████████▊ | 353986/400000 [00:42<00:05, 8321.68it/s] 89%|████████▊ | 354819/400000 [00:42<00:05, 8296.66it/s] 89%|████████▉ | 355658/400000 [00:42<00:05, 8321.52it/s] 89%|████████▉ | 356494/400000 [00:42<00:05, 8331.44it/s] 89%|████████▉ | 357332/400000 [00:43<00:05, 8345.74it/s] 90%|████████▉ | 358167/400000 [00:43<00:05, 8269.32it/s] 90%|████████▉ | 358995/400000 [00:43<00:04, 8251.53it/s] 90%|████████▉ | 359844/400000 [00:43<00:04, 8319.56it/s] 90%|█████████ | 360683/400000 [00:43<00:04, 8339.35it/s] 90%|█████████ | 361526/400000 [00:43<00:04, 8364.82it/s] 91%|█████████ | 362363/400000 [00:43<00:04, 8281.00it/s] 91%|█████████ | 363192/400000 [00:43<00:04, 8080.46it/s] 91%|█████████ | 364037/400000 [00:43<00:04, 8185.83it/s] 91%|█████████ | 364880/400000 [00:43<00:04, 8256.00it/s] 91%|█████████▏| 365724/400000 [00:44<00:04, 8309.75it/s] 92%|█████████▏| 366556/400000 [00:44<00:04, 8281.48it/s] 92%|█████████▏| 367389/400000 [00:44<00:03, 8293.26it/s] 92%|█████████▏| 368219/400000 [00:44<00:03, 8277.48it/s] 92%|█████████▏| 369057/400000 [00:44<00:03, 8306.37it/s] 92%|█████████▏| 369910/400000 [00:44<00:03, 8371.31it/s] 93%|█████████▎| 370748/400000 [00:44<00:03, 8357.27it/s] 93%|█████████▎| 371584/400000 [00:44<00:03, 8279.79it/s] 93%|█████████▎| 372424/400000 [00:44<00:03, 8312.63it/s] 93%|█████████▎| 373256/400000 [00:44<00:03, 8293.74it/s] 94%|█████████▎| 374102/400000 [00:45<00:03, 8342.53it/s] 94%|█████████▎| 374937/400000 [00:45<00:03, 8317.75it/s] 94%|█████████▍| 375769/400000 [00:45<00:02, 8292.93it/s] 94%|█████████▍| 376614/400000 [00:45<00:02, 8339.41it/s] 94%|█████████▍| 377462/400000 [00:45<00:02, 8379.29it/s] 95%|█████████▍| 378306/400000 [00:45<00:02, 8397.25it/s] 95%|█████████▍| 379146/400000 [00:45<00:02, 8316.24it/s] 95%|█████████▍| 379978/400000 [00:45<00:02, 8301.83it/s] 95%|█████████▌| 380816/400000 [00:45<00:02, 8323.84it/s] 95%|█████████▌| 381654/400000 [00:46<00:02, 8338.58it/s] 96%|█████████▌| 382506/400000 [00:46<00:02, 8390.09it/s] 96%|█████████▌| 383346/400000 [00:46<00:01, 8365.55it/s] 96%|█████████▌| 384183/400000 [00:46<00:01, 8317.42it/s] 96%|█████████▋| 385033/400000 [00:46<00:01, 8368.79it/s] 96%|█████████▋| 385878/400000 [00:46<00:01, 8389.27it/s] 97%|█████████▋| 386727/400000 [00:46<00:01, 8418.56it/s] 97%|█████████▋| 387569/400000 [00:46<00:01, 8331.18it/s] 97%|█████████▋| 388403/400000 [00:46<00:01, 8272.91it/s] 97%|█████████▋| 389252/400000 [00:46<00:01, 8335.18it/s] 98%|█████████▊| 390090/400000 [00:47<00:01, 8346.22it/s] 98%|█████████▊| 390925/400000 [00:47<00:01, 8338.10it/s] 98%|█████████▊| 391764/400000 [00:47<00:00, 8351.73it/s] 98%|█████████▊| 392600/400000 [00:47<00:00, 8336.31it/s] 98%|█████████▊| 393442/400000 [00:47<00:00, 8358.46it/s] 99%|█████████▊| 394287/400000 [00:47<00:00, 8385.70it/s] 99%|█████████▉| 395135/400000 [00:47<00:00, 8411.94it/s] 99%|█████████▉| 395977/400000 [00:47<00:00, 8288.58it/s] 99%|█████████▉| 396807/400000 [00:47<00:00, 8239.55it/s] 99%|█████████▉| 397649/400000 [00:47<00:00, 8292.45it/s]100%|█████████▉| 398479/400000 [00:48<00:00, 8233.41it/s]100%|█████████▉| 399320/400000 [00:48<00:00, 8285.37it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8298.03it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7faa9f040b00>, <torchtext.data.dataset.TabularDataset object at 0x7faa9f040c50>, <torchtext.vocab.Vocab object at 0x7faa9f040b70>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.05 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.05 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.05 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.05 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.24 file/s]2020-07-07 12:18:13.800474: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-07 12:18:13.804821: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-07 12:18:13.804994: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d6a2086700 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-07 12:18:13.805009: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist_dataset_small.npy', 'cifar10', 'test', 'train', 'mnist2', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 138978.56it/s] 75%|███████▌  | 7462912/9912422 [00:00<00:12, 198382.10it/s]9920512it [00:00, 41573432.70it/s]                           
0it [00:00, ?it/s]32768it [00:00, 744345.62it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 159740.67it/s]1654784it [00:00, 11227383.04it/s]                         
0it [00:00, ?it/s]8192it [00:00, 255176.26it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
