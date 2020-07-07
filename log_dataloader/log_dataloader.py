
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f4b385ec048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f4b385ec048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f4ba41931e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f4ba41931e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f4bc24dbea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f4bc24dbea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4b514c1620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4b514c1620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4b514c1620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 453218.97it/s] 66%|██████▌   | 6504448/9912422 [00:00<00:05, 645513.08it/s]9920512it [00:00, 41537409.61it/s]                           
0it [00:00, ?it/s]32768it [00:00, 626708.04it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 479200.84it/s]1654784it [00:00, 12070683.49it/s]                         
0it [00:00, ?it/s]8192it [00:00, 203047.74it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4b4e969940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4b384f4be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f4b514c1268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f4b514c1268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f4b514c1268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:45,  3.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:45,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:45,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:44,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:44,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:44,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:44,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:43,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:43,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:30,  4.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:30,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:30,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:30,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:30,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:30,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:29,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:29,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:29,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:29,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:29,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:20,  6.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:20,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:20,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:20,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:20,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:20,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:19,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:19,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:19,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:19,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:13,  9.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:13,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:13,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:13,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:13,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:13,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:13,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:13,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12,  9.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 13.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:09, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:09, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 13.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:00<00:06, 17.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:06, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:06, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:06, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:06, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:06, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:06, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:06, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:00<00:04, 23.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:00<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:00<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:00<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:00<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:00<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:00<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:00<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:00<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 23.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 30.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 30.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 37.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 37.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 37.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 37.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 54.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 62.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 69.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 69.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 76.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 76.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 78.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 78.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 83.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:01<00:00, 83.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:01<00:00, 87.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:01<00:00, 87.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:01<00:00, 87.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:01<00:00, 87.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:01<00:00, 87.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:01<00:00, 87.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 87.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.01s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.01s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 87.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.01s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 87.62 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.01s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 87.62 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.64 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]
0 examples [00:00, ? examples/s]2020-07-07 00:11:07.122058: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-07 00:11:07.184848: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-07-07 00:11:07.185290: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a8b94f6c40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-07 00:11:07.185318: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.22 examples/s]101 examples [00:00,  3.16 examples/s]206 examples [00:00,  4.51 examples/s]312 examples [00:00,  6.44 examples/s]417 examples [00:00,  9.17 examples/s]514 examples [00:00, 13.05 examples/s]623 examples [00:01, 18.54 examples/s]732 examples [00:01, 26.29 examples/s]835 examples [00:01, 37.16 examples/s]941 examples [00:01, 52.29 examples/s]1051 examples [00:01, 73.20 examples/s]1157 examples [00:01, 101.55 examples/s]1268 examples [00:01, 139.59 examples/s]1377 examples [00:01, 189.02 examples/s]1484 examples [00:01, 248.62 examples/s]1590 examples [00:01, 322.49 examples/s]1695 examples [00:02, 407.07 examples/s]1803 examples [00:02, 500.07 examples/s]1908 examples [00:02, 592.17 examples/s]2016 examples [00:02, 684.93 examples/s]2124 examples [00:02, 769.20 examples/s]2231 examples [00:02, 827.79 examples/s]2338 examples [00:02, 886.29 examples/s]2444 examples [00:02, 918.64 examples/s]2548 examples [00:02, 937.40 examples/s]2651 examples [00:02, 948.89 examples/s]2759 examples [00:03, 983.51 examples/s]2862 examples [00:03, 992.17 examples/s]2965 examples [00:03, 1002.01 examples/s]3073 examples [00:03, 1023.07 examples/s]3181 examples [00:03, 1039.49 examples/s]3290 examples [00:03, 1053.76 examples/s]3398 examples [00:03, 1060.56 examples/s]3505 examples [00:03, 1014.76 examples/s]3611 examples [00:03, 1025.76 examples/s]3720 examples [00:04, 1043.08 examples/s]3825 examples [00:04, 1023.64 examples/s]3932 examples [00:04, 1034.31 examples/s]4038 examples [00:04, 1039.49 examples/s]4145 examples [00:04, 1046.61 examples/s]4250 examples [00:04, 1040.66 examples/s]4358 examples [00:04, 1049.47 examples/s]4466 examples [00:04, 1057.15 examples/s]4572 examples [00:04, 1054.78 examples/s]4678 examples [00:04, 1025.58 examples/s]4789 examples [00:05, 1047.60 examples/s]4895 examples [00:05, 1016.85 examples/s]4998 examples [00:05, 1014.82 examples/s]5105 examples [00:05, 1028.33 examples/s]5209 examples [00:05, 968.67 examples/s] 5310 examples [00:05, 979.90 examples/s]5409 examples [00:05, 976.10 examples/s]5519 examples [00:05, 1007.90 examples/s]5625 examples [00:05, 1021.81 examples/s]5733 examples [00:05, 1038.29 examples/s]5838 examples [00:06, 1027.42 examples/s]5947 examples [00:06, 1044.36 examples/s]6055 examples [00:06, 1052.08 examples/s]6161 examples [00:06, 1047.20 examples/s]6270 examples [00:06, 1057.41 examples/s]6376 examples [00:06, 1050.07 examples/s]6482 examples [00:06, 1050.92 examples/s]6590 examples [00:06, 1057.32 examples/s]6696 examples [00:06, 1044.56 examples/s]6801 examples [00:07, 1032.88 examples/s]6909 examples [00:07, 1046.57 examples/s]7014 examples [00:07, 1047.01 examples/s]7119 examples [00:07, 1039.82 examples/s]7224 examples [00:07, 1032.52 examples/s]7328 examples [00:07, 1034.47 examples/s]7434 examples [00:07, 1039.65 examples/s]7538 examples [00:07, 1011.53 examples/s]7640 examples [00:07, 1009.64 examples/s]7742 examples [00:07, 987.12 examples/s] 7848 examples [00:08, 1006.88 examples/s]7949 examples [00:08, 1006.61 examples/s]8057 examples [00:08, 1025.39 examples/s]8163 examples [00:08, 1034.61 examples/s]8270 examples [00:08, 1042.47 examples/s]8376 examples [00:08, 1047.58 examples/s]8481 examples [00:08, 1018.96 examples/s]8587 examples [00:08, 1029.20 examples/s]8691 examples [00:08, 1030.76 examples/s]8795 examples [00:08, 1008.87 examples/s]8899 examples [00:09, 1015.06 examples/s]9008 examples [00:09, 1034.97 examples/s]9115 examples [00:09, 1043.16 examples/s]9222 examples [00:09, 1048.82 examples/s]9328 examples [00:09, 1051.21 examples/s]9435 examples [00:09, 1054.69 examples/s]9543 examples [00:09, 1059.20 examples/s]9651 examples [00:09, 1064.79 examples/s]9758 examples [00:09, 1064.65 examples/s]9865 examples [00:09, 1063.82 examples/s]9972 examples [00:10, 1060.37 examples/s]10079 examples [00:10, 999.38 examples/s]10186 examples [00:10, 1017.35 examples/s]10290 examples [00:10, 1023.36 examples/s]10395 examples [00:10, 1028.41 examples/s]10499 examples [00:10, 998.89 examples/s] 10600 examples [00:10, 1000.12 examples/s]10707 examples [00:10, 1019.77 examples/s]10816 examples [00:10, 1037.54 examples/s]10921 examples [00:11, 1019.09 examples/s]11026 examples [00:11, 1027.15 examples/s]11129 examples [00:11, 995.93 examples/s] 11232 examples [00:11, 1005.21 examples/s]11339 examples [00:11, 1023.10 examples/s]11444 examples [00:11, 1021.51 examples/s]11551 examples [00:11, 1034.70 examples/s]11660 examples [00:11, 1049.61 examples/s]11766 examples [00:11, 1014.21 examples/s]11868 examples [00:11, 1011.21 examples/s]11970 examples [00:12, 1005.31 examples/s]12078 examples [00:12, 1026.46 examples/s]12186 examples [00:12, 1041.02 examples/s]12291 examples [00:12, 1037.89 examples/s]12400 examples [00:12, 1052.84 examples/s]12506 examples [00:12, 1051.03 examples/s]12612 examples [00:12, 1030.79 examples/s]12718 examples [00:12, 1038.64 examples/s]12822 examples [00:12, 1035.94 examples/s]12930 examples [00:12, 1047.17 examples/s]13035 examples [00:13, 1040.68 examples/s]13142 examples [00:13, 1048.03 examples/s]13251 examples [00:13, 1058.10 examples/s]13359 examples [00:13, 1061.69 examples/s]13468 examples [00:13, 1067.38 examples/s]13576 examples [00:13, 1068.45 examples/s]13683 examples [00:13, 1060.74 examples/s]13792 examples [00:13, 1066.71 examples/s]13900 examples [00:13, 1069.96 examples/s]14008 examples [00:13, 1002.38 examples/s]14112 examples [00:14, 1012.58 examples/s]14219 examples [00:14, 1028.39 examples/s]14324 examples [00:14, 1034.28 examples/s]14429 examples [00:14, 1038.37 examples/s]14537 examples [00:14, 1050.18 examples/s]14643 examples [00:14, 1040.13 examples/s]14748 examples [00:14, 1017.73 examples/s]14851 examples [00:14, 1001.08 examples/s]14952 examples [00:14, 995.48 examples/s] 15055 examples [00:15, 1005.20 examples/s]15156 examples [00:15, 990.87 examples/s] 15256 examples [00:15, 989.95 examples/s]15356 examples [00:15, 992.28 examples/s]15456 examples [00:15, 976.25 examples/s]15560 examples [00:15, 994.29 examples/s]15666 examples [00:15, 1009.69 examples/s]15772 examples [00:15, 1023.10 examples/s]15879 examples [00:15, 1035.43 examples/s]15985 examples [00:15, 1042.34 examples/s]16091 examples [00:16, 1047.00 examples/s]16198 examples [00:16, 1051.31 examples/s]16305 examples [00:16, 1055.37 examples/s]16411 examples [00:16, 1052.92 examples/s]16519 examples [00:16, 1058.90 examples/s]16625 examples [00:16, 1055.53 examples/s]16731 examples [00:16, 1007.59 examples/s]16833 examples [00:16, 1009.07 examples/s]16936 examples [00:16, 1014.63 examples/s]17041 examples [00:16, 1023.82 examples/s]17144 examples [00:17, 1023.50 examples/s]17252 examples [00:17, 1039.19 examples/s]17359 examples [00:17, 1047.09 examples/s]17464 examples [00:17, 1025.60 examples/s]17567 examples [00:17, 1009.29 examples/s]17672 examples [00:17, 1020.92 examples/s]17776 examples [00:17, 1024.89 examples/s]17879 examples [00:17, 1002.66 examples/s]17981 examples [00:17, 1007.45 examples/s]18082 examples [00:17, 993.98 examples/s] 18185 examples [00:18, 1004.33 examples/s]18288 examples [00:18, 1010.23 examples/s]18390 examples [00:18, 1004.16 examples/s]18495 examples [00:18, 1017.15 examples/s]18601 examples [00:18, 1029.54 examples/s]18708 examples [00:18, 1040.40 examples/s]18813 examples [00:18, 1020.34 examples/s]18919 examples [00:18, 1030.09 examples/s]19026 examples [00:18, 1039.87 examples/s]19134 examples [00:18, 1050.73 examples/s]19240 examples [00:19, 1030.97 examples/s]19344 examples [00:19, 1032.62 examples/s]19448 examples [00:19, 1016.99 examples/s]19553 examples [00:19, 1024.70 examples/s]19656 examples [00:19, 1018.35 examples/s]19762 examples [00:19, 1027.67 examples/s]19865 examples [00:19, 1010.72 examples/s]19970 examples [00:19, 1019.71 examples/s]20073 examples [00:19, 970.89 examples/s] 20179 examples [00:20, 994.29 examples/s]20286 examples [00:20, 1014.09 examples/s]20393 examples [00:20, 1030.03 examples/s]20497 examples [00:20, 1003.62 examples/s]20605 examples [00:20, 1025.09 examples/s]20713 examples [00:20, 1038.34 examples/s]20818 examples [00:20, 1025.67 examples/s]20922 examples [00:20, 1029.06 examples/s]21027 examples [00:20, 1034.66 examples/s]21131 examples [00:20, 1012.58 examples/s]21233 examples [00:21, 989.44 examples/s] 21341 examples [00:21, 1012.53 examples/s]21449 examples [00:21, 1030.37 examples/s]21554 examples [00:21, 1035.15 examples/s]21662 examples [00:21, 1046.89 examples/s]21770 examples [00:21, 1055.98 examples/s]21878 examples [00:21, 1058.17 examples/s]21984 examples [00:21, 1045.14 examples/s]22092 examples [00:21, 1055.31 examples/s]22199 examples [00:21, 1059.03 examples/s]22306 examples [00:22, 1061.60 examples/s]22415 examples [00:22, 1067.93 examples/s]22522 examples [00:22, 1067.03 examples/s]22629 examples [00:22, 1063.32 examples/s]22736 examples [00:22, 1063.67 examples/s]22843 examples [00:22, 1060.22 examples/s]22950 examples [00:22, 1053.10 examples/s]23056 examples [00:22, 1040.57 examples/s]23161 examples [00:22, 1013.08 examples/s]23271 examples [00:22, 1035.15 examples/s]23380 examples [00:23, 1050.40 examples/s]23486 examples [00:23, 1045.30 examples/s]23591 examples [00:23, 1045.34 examples/s]23698 examples [00:23, 1051.83 examples/s]23806 examples [00:23, 1059.39 examples/s]23913 examples [00:23, 1062.19 examples/s]24021 examples [00:23, 1066.76 examples/s]24129 examples [00:23, 1067.95 examples/s]24236 examples [00:23, 1068.49 examples/s]24343 examples [00:23, 1065.00 examples/s]24450 examples [00:24, 1020.65 examples/s]24553 examples [00:24, 1022.36 examples/s]24656 examples [00:24, 1020.80 examples/s]24764 examples [00:24, 1035.98 examples/s]24873 examples [00:24, 1050.56 examples/s]24979 examples [00:24, 1044.96 examples/s]25084 examples [00:24, 1029.23 examples/s]25189 examples [00:24, 1035.26 examples/s]25296 examples [00:24, 1044.41 examples/s]25404 examples [00:25, 1052.32 examples/s]25510 examples [00:25, 1026.56 examples/s]25616 examples [00:25, 1034.48 examples/s]25720 examples [00:25, 1022.76 examples/s]25823 examples [00:25, 1019.49 examples/s]25929 examples [00:25, 1030.00 examples/s]26035 examples [00:25, 1036.45 examples/s]26143 examples [00:25, 1048.35 examples/s]26250 examples [00:25, 1053.11 examples/s]26359 examples [00:25, 1061.04 examples/s]26466 examples [00:26, 1030.49 examples/s]26571 examples [00:26, 1035.87 examples/s]26678 examples [00:26, 1045.43 examples/s]26784 examples [00:26, 1046.78 examples/s]26893 examples [00:26, 1056.69 examples/s]27001 examples [00:26, 1060.42 examples/s]27108 examples [00:26, 1060.58 examples/s]27215 examples [00:26, 1059.84 examples/s]27323 examples [00:26, 1063.72 examples/s]27431 examples [00:26, 1065.92 examples/s]27538 examples [00:27, 1064.67 examples/s]27645 examples [00:27, 1047.58 examples/s]27752 examples [00:27, 1053.62 examples/s]27858 examples [00:27, 1051.78 examples/s]27966 examples [00:27, 1058.78 examples/s]28072 examples [00:27, 1058.59 examples/s]28178 examples [00:27, 1054.68 examples/s]28284 examples [00:27, 1049.77 examples/s]28389 examples [00:27, 1006.80 examples/s]28497 examples [00:27, 1025.63 examples/s]28605 examples [00:28, 1038.71 examples/s]28711 examples [00:28, 1043.24 examples/s]28816 examples [00:28, 1041.12 examples/s]28922 examples [00:28, 1044.97 examples/s]29033 examples [00:28, 1061.16 examples/s]29143 examples [00:28, 1070.00 examples/s]29251 examples [00:28, 1067.11 examples/s]29360 examples [00:28, 1072.70 examples/s]29471 examples [00:28, 1081.75 examples/s]29580 examples [00:29, 1076.07 examples/s]29689 examples [00:29, 1077.28 examples/s]29797 examples [00:29, 1048.04 examples/s]29905 examples [00:29, 1055.44 examples/s]30011 examples [00:29, 1009.64 examples/s]30113 examples [00:29, 978.01 examples/s] 30221 examples [00:29, 1004.38 examples/s]30330 examples [00:29, 1026.13 examples/s]30435 examples [00:29, 1032.83 examples/s]30544 examples [00:29, 1047.51 examples/s]30652 examples [00:30, 1055.65 examples/s]30760 examples [00:30, 1061.50 examples/s]30867 examples [00:30, 1027.92 examples/s]30976 examples [00:30, 1043.74 examples/s]31087 examples [00:30, 1060.84 examples/s]31195 examples [00:30, 1066.02 examples/s]31303 examples [00:30, 1069.60 examples/s]31411 examples [00:30, 1024.74 examples/s]31515 examples [00:30, 1028.32 examples/s]31622 examples [00:30, 1039.96 examples/s]31730 examples [00:31, 1051.56 examples/s]31838 examples [00:31, 1059.75 examples/s]31945 examples [00:31, 1042.49 examples/s]32050 examples [00:31, 1036.03 examples/s]32155 examples [00:31, 1039.24 examples/s]32264 examples [00:31, 1053.96 examples/s]32373 examples [00:31, 1062.37 examples/s]32480 examples [00:31, 1062.66 examples/s]32590 examples [00:31, 1071.51 examples/s]32702 examples [00:31, 1082.95 examples/s]32811 examples [00:32, 1077.34 examples/s]32919 examples [00:32, 1071.86 examples/s]33027 examples [00:32, 1065.96 examples/s]33134 examples [00:32, 1058.91 examples/s]33241 examples [00:32, 1059.20 examples/s]33350 examples [00:32, 1065.66 examples/s]33457 examples [00:32, 1046.98 examples/s]33562 examples [00:32, 1047.79 examples/s]33667 examples [00:32, 1046.13 examples/s]33776 examples [00:33, 1057.47 examples/s]33884 examples [00:33, 1063.66 examples/s]33991 examples [00:33, 1052.57 examples/s]34097 examples [00:33, 1008.43 examples/s]34203 examples [00:33, 1021.84 examples/s]34314 examples [00:33, 1044.72 examples/s]34423 examples [00:33, 1057.59 examples/s]34532 examples [00:33, 1064.87 examples/s]34639 examples [00:33, 1063.35 examples/s]34746 examples [00:33, 1063.71 examples/s]34855 examples [00:34, 1069.38 examples/s]34965 examples [00:34, 1076.73 examples/s]35073 examples [00:34, 1066.84 examples/s]35180 examples [00:34, 1036.34 examples/s]35287 examples [00:34, 1044.98 examples/s]35396 examples [00:34, 1055.47 examples/s]35502 examples [00:34, 1039.07 examples/s]35607 examples [00:34, 1039.71 examples/s]35715 examples [00:34, 1049.05 examples/s]35821 examples [00:34, 1043.97 examples/s]35926 examples [00:35, 1030.76 examples/s]36032 examples [00:35, 1038.29 examples/s]36136 examples [00:35, 1033.76 examples/s]36242 examples [00:35, 1039.38 examples/s]36350 examples [00:35, 1050.51 examples/s]36457 examples [00:35, 1054.63 examples/s]36563 examples [00:35, 1044.04 examples/s]36671 examples [00:35, 1053.98 examples/s]36777 examples [00:35, 1046.63 examples/s]36882 examples [00:35, 1030.28 examples/s]36992 examples [00:36, 1047.76 examples/s]37098 examples [00:36, 1049.59 examples/s]37206 examples [00:36, 1057.04 examples/s]37312 examples [00:36, 1033.16 examples/s]37419 examples [00:36, 1041.31 examples/s]37527 examples [00:36, 1052.21 examples/s]37633 examples [00:36, 1052.38 examples/s]37739 examples [00:36, 1051.47 examples/s]37847 examples [00:36, 1058.33 examples/s]37953 examples [00:36, 1055.21 examples/s]38060 examples [00:37, 1055.82 examples/s]38166 examples [00:37, 1054.94 examples/s]38272 examples [00:37, 1033.18 examples/s]38379 examples [00:37, 1042.70 examples/s]38489 examples [00:37, 1057.32 examples/s]38596 examples [00:37, 1059.01 examples/s]38705 examples [00:37, 1065.58 examples/s]38812 examples [00:37, 1063.14 examples/s]38919 examples [00:37, 1035.10 examples/s]39025 examples [00:38, 1039.79 examples/s]39130 examples [00:38, 1038.47 examples/s]39234 examples [00:38, 1035.81 examples/s]39341 examples [00:38, 1044.22 examples/s]39448 examples [00:38, 1050.44 examples/s]39555 examples [00:38, 1053.75 examples/s]39662 examples [00:38, 1058.22 examples/s]39768 examples [00:38, 1046.40 examples/s]39874 examples [00:38, 1049.99 examples/s]39980 examples [00:38, 1049.74 examples/s]40086 examples [00:39, 1002.07 examples/s]40195 examples [00:39, 1026.68 examples/s]40305 examples [00:39, 1047.44 examples/s]40411 examples [00:39, 1039.78 examples/s]40516 examples [00:39, 1006.60 examples/s]40622 examples [00:39, 1021.91 examples/s]40730 examples [00:39, 1037.61 examples/s]40835 examples [00:39, 1037.19 examples/s]40942 examples [00:39, 1044.45 examples/s]41047 examples [00:39, 1043.83 examples/s]41152 examples [00:40, 1042.00 examples/s]41257 examples [00:40, 1028.36 examples/s]41363 examples [00:40, 1035.12 examples/s]41467 examples [00:40, 1018.22 examples/s]41572 examples [00:40, 1024.77 examples/s]41680 examples [00:40, 1039.85 examples/s]41786 examples [00:40, 1045.07 examples/s]41891 examples [00:40, 1030.01 examples/s]41995 examples [00:40, 1031.42 examples/s]42099 examples [00:40, 1030.82 examples/s]42208 examples [00:41, 1045.14 examples/s]42314 examples [00:41, 1047.73 examples/s]42419 examples [00:41, 993.53 examples/s] 42526 examples [00:41, 1013.93 examples/s]42634 examples [00:41, 1030.60 examples/s]42741 examples [00:41, 1041.04 examples/s]42851 examples [00:41, 1055.89 examples/s]42961 examples [00:41, 1067.36 examples/s]43068 examples [00:41, 1059.92 examples/s]43175 examples [00:42, 1046.13 examples/s]43282 examples [00:42, 1050.80 examples/s]43391 examples [00:42, 1059.39 examples/s]43498 examples [00:42, 1057.27 examples/s]43604 examples [00:42, 1044.38 examples/s]43709 examples [00:42, 1032.06 examples/s]43816 examples [00:42, 1041.63 examples/s]43923 examples [00:42, 1049.32 examples/s]44029 examples [00:42, 1027.46 examples/s]44132 examples [00:42, 1005.00 examples/s]44237 examples [00:43, 1017.97 examples/s]44339 examples [00:43, 1008.63 examples/s]44446 examples [00:43, 1023.65 examples/s]44549 examples [00:43, 1024.92 examples/s]44658 examples [00:43, 1043.57 examples/s]44769 examples [00:43, 1059.89 examples/s]44878 examples [00:43, 1067.88 examples/s]44985 examples [00:43, 1050.28 examples/s]45091 examples [00:43, 1044.58 examples/s]45198 examples [00:43, 1050.78 examples/s]45307 examples [00:44, 1060.95 examples/s]45416 examples [00:44, 1067.88 examples/s]45523 examples [00:44, 1068.32 examples/s]45630 examples [00:44, 1063.79 examples/s]45737 examples [00:44, 1062.45 examples/s]45845 examples [00:44, 1067.53 examples/s]45954 examples [00:44, 1072.30 examples/s]46062 examples [00:44, 1049.40 examples/s]46168 examples [00:44, 1043.50 examples/s]46275 examples [00:44, 1051.06 examples/s]46383 examples [00:45, 1058.44 examples/s]46493 examples [00:45, 1070.11 examples/s]46601 examples [00:45, 1068.03 examples/s]46708 examples [00:45, 1045.92 examples/s]46813 examples [00:45, 1040.69 examples/s]46918 examples [00:45, 1014.33 examples/s]47026 examples [00:45, 1032.76 examples/s]47130 examples [00:45, 1034.72 examples/s]47236 examples [00:45, 1041.14 examples/s]47345 examples [00:45, 1052.95 examples/s]47451 examples [00:46, 1051.26 examples/s]47560 examples [00:46, 1061.76 examples/s]47667 examples [00:46, 1052.21 examples/s]47774 examples [00:46, 1056.13 examples/s]47882 examples [00:46, 1061.44 examples/s]47989 examples [00:46, 1063.48 examples/s]48096 examples [00:46, 1048.49 examples/s]48201 examples [00:46, 1030.58 examples/s]48306 examples [00:46, 1035.40 examples/s]48410 examples [00:47, 1031.91 examples/s]48514 examples [00:47, 1030.97 examples/s]48621 examples [00:47, 1040.25 examples/s]48727 examples [00:47, 1045.60 examples/s]48832 examples [00:47, 1015.75 examples/s]48940 examples [00:47, 1032.07 examples/s]49044 examples [00:47, 1025.10 examples/s]49152 examples [00:47, 1040.15 examples/s]49258 examples [00:47, 1045.52 examples/s]49363 examples [00:47, 1034.69 examples/s]49467 examples [00:48, 1024.17 examples/s]49573 examples [00:48, 1033.69 examples/s]49683 examples [00:48, 1050.52 examples/s]49789 examples [00:48, 1049.81 examples/s]49895 examples [00:48, 1045.61 examples/s]50000 examples [00:48, 1037.20 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6376/50000 [00:00<00:00, 63754.89 examples/s] 39%|███▉      | 19529/50000 [00:00<00:00, 75412.34 examples/s] 64%|██████▍   | 31983/50000 [00:00<00:00, 85533.94 examples/s] 89%|████████▉ | 44379/50000 [00:00<00:00, 94303.56 examples/s]                                                               0 examples [00:00, ? examples/s]88 examples [00:00, 877.45 examples/s]198 examples [00:00, 932.35 examples/s]299 examples [00:00, 953.90 examples/s]393 examples [00:00, 949.54 examples/s]505 examples [00:00, 994.94 examples/s]600 examples [00:00, 980.86 examples/s]710 examples [00:00, 1012.38 examples/s]817 examples [00:00, 1027.41 examples/s]923 examples [00:00, 1036.01 examples/s]1034 examples [00:01, 1055.75 examples/s]1145 examples [00:01, 1070.51 examples/s]1254 examples [00:01, 1076.25 examples/s]1361 examples [00:01, 1071.59 examples/s]1469 examples [00:01, 1073.69 examples/s]1582 examples [00:01, 1089.95 examples/s]1692 examples [00:01, 1091.35 examples/s]1802 examples [00:01, 1092.44 examples/s]1912 examples [00:01, 1069.84 examples/s]2019 examples [00:01, 1069.40 examples/s]2128 examples [00:02, 1072.85 examples/s]2236 examples [00:02, 1074.24 examples/s]2348 examples [00:02, 1085.01 examples/s]2460 examples [00:02, 1093.12 examples/s]2570 examples [00:02, 1063.25 examples/s]2681 examples [00:02, 1074.93 examples/s]2789 examples [00:02, 1060.61 examples/s]2896 examples [00:02, 1055.26 examples/s]3003 examples [00:02, 1057.67 examples/s]3109 examples [00:02, 1054.40 examples/s]3215 examples [00:03, 1022.89 examples/s]3319 examples [00:03, 1026.47 examples/s]3426 examples [00:03, 1037.10 examples/s]3535 examples [00:03, 1051.46 examples/s]3641 examples [00:03, 1049.50 examples/s]3752 examples [00:03, 1064.30 examples/s]3859 examples [00:03, 1060.35 examples/s]3968 examples [00:03, 1068.22 examples/s]4075 examples [00:03, 1067.35 examples/s]4182 examples [00:03, 1056.71 examples/s]4288 examples [00:04, 1030.60 examples/s]4395 examples [00:04, 1040.47 examples/s]4503 examples [00:04, 1049.59 examples/s]4609 examples [00:04, 1052.20 examples/s]4716 examples [00:04, 1056.05 examples/s]4825 examples [00:04, 1063.65 examples/s]4932 examples [00:04, 1057.98 examples/s]5038 examples [00:04, 1057.19 examples/s]5146 examples [00:04, 1063.46 examples/s]5255 examples [00:04, 1068.39 examples/s]5362 examples [00:05, 1050.04 examples/s]5473 examples [00:05, 1065.77 examples/s]5582 examples [00:05, 1072.20 examples/s]5691 examples [00:05, 1077.19 examples/s]5803 examples [00:05, 1087.66 examples/s]5914 examples [00:05, 1092.47 examples/s]6024 examples [00:05, 1085.53 examples/s]6133 examples [00:05, 1017.72 examples/s]6236 examples [00:05, 1018.84 examples/s]6345 examples [00:06, 1038.43 examples/s]6451 examples [00:06, 1044.45 examples/s]6556 examples [00:06, 1032.67 examples/s]6663 examples [00:06, 1042.83 examples/s]6769 examples [00:06, 1046.13 examples/s]6874 examples [00:06, 1025.08 examples/s]6983 examples [00:06, 1043.58 examples/s]7091 examples [00:06, 1052.79 examples/s]7197 examples [00:06, 1051.37 examples/s]7303 examples [00:06, 1051.36 examples/s]7411 examples [00:07, 1058.18 examples/s]7521 examples [00:07, 1068.75 examples/s]7632 examples [00:07, 1077.73 examples/s]7740 examples [00:07, 1055.71 examples/s]7848 examples [00:07, 1060.60 examples/s]7955 examples [00:07, 1024.89 examples/s]8062 examples [00:07, 1034.43 examples/s]8166 examples [00:07, 1033.76 examples/s]8270 examples [00:07, 1016.14 examples/s]8379 examples [00:07, 1037.10 examples/s]8486 examples [00:08, 1046.52 examples/s]8591 examples [00:08, 1044.12 examples/s]8696 examples [00:08, 1029.68 examples/s]8801 examples [00:08, 1033.56 examples/s]8905 examples [00:08, 1001.00 examples/s]9017 examples [00:08, 1033.22 examples/s]9126 examples [00:08, 1049.43 examples/s]9232 examples [00:08, 1040.75 examples/s]9337 examples [00:08, 1038.98 examples/s]9442 examples [00:08, 1041.00 examples/s]9547 examples [00:09, 1038.97 examples/s]9653 examples [00:09, 1043.06 examples/s]9758 examples [00:09, 1041.28 examples/s]9864 examples [00:09, 1045.86 examples/s]9969 examples [00:09, 1039.96 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRPXO3E/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRPXO3E/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4b514c1620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4b514c1620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4b514c1620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4ada923588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4ada960588>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4b514c1620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4b514c1620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4b514c1620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4b384f45c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4b384f4358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f4acf0af1e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f4acf0af1e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f4acf0af1e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f4acf0af2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f4acf0af2f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f4acf0af2f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=11ca16648083f009123cfa49b1dfebf174c6a8b69084c35daa00e5a98cab7c1d
  Stored in directory: /tmp/pip-ephem-wheel-cache-2p89ghzn/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:12:03, 13.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:58:32, 18.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:08:14, 26.2kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:24:18, 37.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<4:28:21, 53.3kB/s].vector_cache/glove.6B.zip:   1%|          | 8.14M/862M [00:01<3:06:57, 76.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:10:19, 109kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:30:44, 155kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:01<1:03:24, 221kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.3M/862M [00:01<44:11, 315kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.6M/862M [00:01<30:56, 449kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:01<21:35, 639kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:01<15:10, 904kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.2M/862M [00:02<10:38, 1.28MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:02<07:28, 1.81MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<05:31, 2.45MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.3M/862M [00:03<05:46, 2.33MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.8M/862M [00:03<04:52, 2.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:03<03:38, 3.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.5M/862M [00:05<06:13, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:05<07:47, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<06:13, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<04:33, 2.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:07<09:45, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.0M/862M [00:07<08:18, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:07<06:07, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:09<07:09, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:09<07:42, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<05:59, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<04:19, 3.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<22:59, 574kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:11<17:25, 756kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:11<12:28, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<11:47, 1.11MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:13<09:34, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:13<07:01, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:15<07:59, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<08:15, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:15<06:26, 2.02MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<06:35, 1.97MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:17<05:56, 2.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:17<04:28, 2.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<06:09, 2.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:19<05:38, 2.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:19<04:12, 3.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:20<03:45, 3.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:21<8:58:43, 23.9kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:21<6:17:34, 34.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:21<4:23:35, 48.5kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:23<3:11:45, 66.7kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:23<2:17:02, 93.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.1M/862M [00:23<1:36:24, 132kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:23<1:07:30, 189kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:25<52:13, 243kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:25<37:49, 336kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<26:42, 475kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<21:36, 585kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<17:42, 714kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<12:55, 977kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<09:11, 1.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<13:07, 958kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<10:29, 1.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<07:39, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<08:16, 1.51MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<08:19, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:24, 1.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<04:37, 2.69MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<10:21, 1.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<08:30, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<06:13, 1.99MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:15, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:36, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<05:50, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<04:14, 2.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<09:24, 1.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<07:52, 1.56MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:48, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:55, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<07:20, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:45, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:39<04:08, 2.94MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<11:47:11, 17.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<8:16:00, 24.5kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<5:46:43, 35.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<4:04:50, 49.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<2:52:31, 70.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<2:00:48, 99.9kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<1:27:10, 138kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<1:02:12, 193kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<43:45, 274kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<33:22, 358kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<25:47, 463kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<18:33, 643kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<13:11, 904kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<13:01, 913kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<10:19, 1.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<07:28, 1.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<08:00, 1.48MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<08:00, 1.48MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<06:06, 1.93MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<04:24, 2.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<09:58, 1.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<08:11, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<06:01, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:56, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:04, 1.92MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:30, 2.59MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:52, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:29, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:02, 2.30MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:56<03:40, 3.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<08:05, 1.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<06:50, 1.69MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<05:04, 2.27MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:14, 1.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:41, 1.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:12, 2.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<03:46, 3.03MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<09:35, 1.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<07:53, 1.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<05:48, 1.96MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<04:44, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<7:54:20, 23.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<5:32:18, 34.1kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<3:52:10, 48.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<2:45:39, 68.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<1:58:41, 95.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<1:23:36, 135kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<58:28, 192kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<48:07, 233kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<34:51, 321kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<24:36, 454kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<19:45, 564kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<14:54, 746kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<10:41, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<10:06, 1.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<08:15, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<06:00, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:44, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:56, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:24, 2.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:32, 1.97MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:01, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:44, 2.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:09, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:49, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<04:32, 2.39MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:18, 3.26MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:04, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:46, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:00, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:00, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:23, 1.68MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:56, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:39, 2.92MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:41, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:06, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<03:50, 2.77MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:08, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:45, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:33, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:53, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:29, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:22, 3.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:48, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:29, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:17, 2.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:07, 3.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<09:09, 1.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:29, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:29, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:15, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:25, 1.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<04:03, 2.54MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:14, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:45, 1.78MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<04:28, 2.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:17, 3.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:58, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:13, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:54, 2.60MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:05, 1.98MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:36, 2.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<03:28, 2.91MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:47, 2.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:22, 2.29MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:16, 3.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:39, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:18, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<04:12, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:32, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:12, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:11, 3.10MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<02:50, 3.47MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<6:42:13, 24.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<4:41:47, 34.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<3:16:48, 49.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<2:20:21, 69.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<1:40:23, 97.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<1:10:44, 138kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<49:23, 197kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<44:56, 216kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<32:25, 299kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<22:53, 423kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<18:13, 529kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<13:43, 702kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<09:49, 977kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<09:07, 1.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<08:20, 1.15MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<06:15, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<04:27, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<43:05, 221kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<31:08, 305kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<21:58, 431kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<17:32, 538kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<13:14, 712kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<09:28, 992kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:48, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:04, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:02, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<04:20, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<07:35, 1.23MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:16, 1.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:37, 2.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<05:22, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:43, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:28, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<03:14, 2.83MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<1:07:42, 135kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<48:18, 189kB/s]  .vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<33:57, 269kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<25:48, 352kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<18:58, 479kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<13:28, 672kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<11:31, 783kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<08:58, 1.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<06:29, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:46, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:16, 1.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:50, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<05:41, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:53, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<03:36, 2.45MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:35, 1.92MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:00, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:56, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<02:50, 3.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<8:27:49, 17.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<5:56:07, 24.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<4:08:45, 35.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<2:55:27, 49.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<2:04:32, 69.7kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<1:27:25, 99.1kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<1:01:05, 141kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<46:28, 185kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<33:22, 258kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<23:31, 365kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<18:24, 464kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<14:37, 584kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<10:36, 803kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<07:28, 1.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<17:43, 478kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<13:15, 638kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<09:28, 891kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<08:34, 980kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<07:38, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:42, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<04:05, 2.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<04:50, 1.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<5:32:02, 25.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<3:52:06, 35.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<2:43:07, 50.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<1:55:50, 71.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<1:21:22, 101kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<57:56, 142kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<41:21, 198kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<29:04, 281kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<22:09, 367kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<17:15, 471kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<12:26, 653kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<08:44, 924kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<19:19, 418kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<14:21, 561kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<10:13, 786kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<08:59, 890kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:06, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:09, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:28, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:26, 1.46MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:08, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<03:00, 2.62MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:24, 1.46MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:34, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<03:22, 2.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<04:10, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:32, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:34, 2.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<03:43, 2.07MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:24, 2.26MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<02:34, 2.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:35, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:04, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:14, 2.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:28, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:05, 2.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:20, 3.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:25, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:09, 2.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<02:23, 3.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:26, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:55, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:07, 2.38MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:15, 3.28MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<55:07, 134kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<39:11, 188kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<27:31, 267kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<20:54, 350kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<16:07, 453kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<11:38, 627kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<09:15, 782kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<07:13, 1.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<05:13, 1.38MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<05:18, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:27, 1.61MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:16, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:56, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:12, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:15, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<02:20, 3.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<07:50, 898kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:05, 1.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<04:23, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:07<03:09, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<30:17, 230kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<21:53, 318kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<15:26, 449kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<11:11, 617kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<4:53:16, 23.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<3:25:25, 33.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<2:23:01, 47.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<1:43:22, 66.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<1:13:51, 92.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<51:56, 131kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<36:13, 187kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<29:38, 228kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<28:09, 240kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<20:16, 331kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<14:54, 449kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<10:34, 631kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<08:49, 751kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:51, 967kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<04:56, 1.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:59, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:09, 1.57MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<03:04, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:41, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:54, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:00, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:12, 2.92MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:52, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:22, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:31, 2.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:15, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:55, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:11, 2.88MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:01, 2.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:23, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:41, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:52, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:17, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:33, 2.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<01:52, 3.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:49, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:18, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:27, 2.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:08, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:49, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:07, 2.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:53, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:18, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:37, 2.28MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<01:54, 3.13MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<43:43, 136kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<31:12, 190kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<21:52, 270kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<16:36, 354kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<12:12, 481kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<08:38, 677kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<07:23, 785kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<06:19, 919kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<04:39, 1.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<03:18, 1.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<06:33, 876kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<05:11, 1.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<03:44, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:56, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:55, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:58, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<02:09, 2.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:08, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:28, 1.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:33, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:04, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:43, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:02, 2.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:41, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:27, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:50, 2.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:37, 3.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<3:41:28, 24.4kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<2:35:05, 34.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<1:47:48, 49.7kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<1:17:51, 68.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<55:39, 95.9kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<39:10, 136kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<27:14, 194kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<25:25, 207kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<18:18, 287kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<12:52, 407kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<10:11, 510kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<08:08, 638kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<05:54, 877kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<04:11, 1.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:55, 1.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:58, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<02:53, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:12, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:45, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:02, 2.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:36, 1.92MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:19, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:43, 2.87MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:22, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:09, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:37, 3.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:17, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:05, 2.32MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:34, 3.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:14, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:02, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:32, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:11, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:01, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:30, 3.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:27, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:54, 2.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:23, 3.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:58, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:32, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:53, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:22, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:36, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:03, 2.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:28, 3.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<32:46, 136kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<23:22, 190kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<16:22, 269kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<12:23, 353kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<09:36, 455kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<06:53, 632kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<04:51, 892kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<05:16, 817kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:08, 1.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<02:58, 1.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<03:02, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:34, 1.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:52, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:17, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:01, 2.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:30, 2.74MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:01, 2.03MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:14, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:44, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<01:17, 3.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:10, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:56, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:27, 2.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:17, 3.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<2:33:40, 25.9kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<1:47:06, 36.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<1:14:50, 52.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<53:10, 73.4kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<37:18, 104kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<25:50, 149kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<23:56, 160kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<17:03, 225kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<11:56, 319kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<08:20, 452kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<20:05, 188kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<14:48, 254kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<10:30, 357kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<07:17, 507kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<15:29, 239kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<11:11, 330kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<07:52, 465kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<06:18, 575kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<05:09, 704kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:46, 955kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<02:39, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<31:18, 114kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<22:13, 160kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<15:31, 227kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<11:34, 302kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<08:48, 396kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<06:17, 552kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<04:24, 781kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<04:59, 685kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:50, 888kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<02:45, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:42, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:14, 1.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:38, 2.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:54, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:00, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:34, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:07, 2.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<23:46, 135kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<16:56, 190kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<11:49, 269kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<08:55, 353kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<06:51, 459kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<04:56, 634kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:54, 790kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<03:02, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<02:11, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:13, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:51, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:21, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:38, 1.80MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:44, 1.69MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:20, 2.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:57, 2.99MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<02:40, 1.08MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<02:09, 1.33MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:34, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:44, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:30, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:06, 2.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:24, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:15, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:56, 2.86MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:17, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:26, 1.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:07, 2.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<00:48, 3.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:07, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:45, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:16, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:28, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:32, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:11, 2.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:51, 2.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:12, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<1:38:01, 25.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<1:08:06, 36.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<47:11, 50.9kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<33:31, 71.5kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<23:28, 102kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<16:11, 145kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<12:26, 188kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<08:55, 260kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<06:14, 369kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<04:49, 470kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:35, 628kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<02:32, 880kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:20, 937kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:36, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<08:44, 243kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<06:18, 336kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<04:24, 475kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<03:31, 584kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:52, 715kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:06, 971kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<01:27, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<1:55:46, 17.2kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<1:20:59, 24.5kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<55:59, 35.0kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<38:55, 49.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<27:36, 69.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<19:17, 98.8kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<13:14, 141kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<10:21, 179kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<07:23, 250kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<05:08, 354kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<03:33, 503kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<06:58, 256kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<05:02, 352kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<03:31, 497kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:49, 608kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:19, 738kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:41, 1.00MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<01:10, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<06:46, 243kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<04:53, 335kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<03:24, 473kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<02:42, 584kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<02:12, 712kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:36, 966kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:06, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<06:35, 229kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<04:44, 317kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<03:18, 447kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<02:35, 555kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<02:06, 683kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<01:31, 931kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<01:03, 1.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<02:07, 647kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:37, 844kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<01:08, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:04, 1.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:53, 1.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:38, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:43, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:37, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:27, 2.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:35, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:38, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:30, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:31, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:28, 2.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:20, 3.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:28, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:26, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:19, 3.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:16, 3.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<40:12, 24.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<27:53, 34.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<18:39, 49.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<13:20, 67.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<09:31, 94.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<06:37, 133kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<04:25, 190kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<03:40, 225kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<02:37, 313kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<01:48, 443kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<01:13, 627kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:41, 447kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:20, 565kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:57, 780kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:38, 1.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:52, 790kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:40, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:28, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:27, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:26, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:19, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:13, 2.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:26, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:21, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:15, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:14, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:10, 2.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:12, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:13, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:09, 2.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:06, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:06, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:05, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:23<00:03, 3.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 3.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 2.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.42MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 3.31MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.33MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 818/400000 [00:00<00:48, 8176.38it/s]  0%|          | 1660/400000 [00:00<00:48, 8245.35it/s]  1%|          | 2523/400000 [00:00<00:47, 8354.55it/s]  1%|          | 3380/400000 [00:00<00:47, 8416.52it/s]  1%|          | 4204/400000 [00:00<00:47, 8360.53it/s]  1%|          | 4938/400000 [00:00<00:49, 8022.68it/s]  1%|▏         | 5677/400000 [00:00<00:50, 7821.28it/s]  2%|▏         | 6520/400000 [00:00<00:49, 7992.73it/s]  2%|▏         | 7373/400000 [00:00<00:48, 8143.86it/s]  2%|▏         | 8191/400000 [00:01<00:48, 8152.98it/s]  2%|▏         | 9053/400000 [00:01<00:47, 8286.87it/s]  2%|▏         | 9899/400000 [00:01<00:46, 8338.06it/s]  3%|▎         | 10723/400000 [00:01<00:46, 8305.77it/s]  3%|▎         | 11563/400000 [00:01<00:46, 8331.68it/s]  3%|▎         | 12412/400000 [00:01<00:46, 8376.95it/s]  3%|▎         | 13251/400000 [00:01<00:46, 8379.44it/s]  4%|▎         | 14087/400000 [00:01<00:46, 8226.14it/s]  4%|▎         | 14918/400000 [00:01<00:46, 8249.76it/s]  4%|▍         | 15759/400000 [00:01<00:46, 8295.72it/s]  4%|▍         | 16628/400000 [00:02<00:45, 8408.53it/s]  4%|▍         | 17489/400000 [00:02<00:45, 8466.95it/s]  5%|▍         | 18349/400000 [00:02<00:44, 8506.02it/s]  5%|▍         | 19201/400000 [00:02<00:44, 8508.06it/s]  5%|▌         | 20052/400000 [00:02<00:44, 8504.81it/s]  5%|▌         | 20916/400000 [00:02<00:44, 8544.69it/s]  5%|▌         | 21774/400000 [00:02<00:44, 8554.47it/s]  6%|▌         | 22630/400000 [00:02<00:44, 8395.12it/s]  6%|▌         | 23473/400000 [00:02<00:44, 8404.53it/s]  6%|▌         | 24319/400000 [00:02<00:44, 8418.94it/s]  6%|▋         | 25172/400000 [00:03<00:44, 8450.63it/s]  7%|▋         | 26018/400000 [00:03<00:45, 8278.16it/s]  7%|▋         | 26886/400000 [00:03<00:44, 8394.43it/s]  7%|▋         | 27737/400000 [00:03<00:44, 8426.58it/s]  7%|▋         | 28603/400000 [00:03<00:43, 8494.56it/s]  7%|▋         | 29468/400000 [00:03<00:43, 8539.47it/s]  8%|▊         | 30325/400000 [00:03<00:43, 8547.31it/s]  8%|▊         | 31187/400000 [00:03<00:43, 8566.87it/s]  8%|▊         | 32044/400000 [00:03<00:43, 8509.32it/s]  8%|▊         | 32896/400000 [00:03<00:43, 8386.47it/s]  8%|▊         | 33750/400000 [00:04<00:43, 8430.55it/s]  9%|▊         | 34606/400000 [00:04<00:43, 8466.76it/s]  9%|▉         | 35464/400000 [00:04<00:42, 8500.38it/s]  9%|▉         | 36316/400000 [00:04<00:42, 8504.62it/s]  9%|▉         | 37167/400000 [00:04<00:43, 8279.72it/s] 10%|▉         | 38033/400000 [00:04<00:43, 8389.40it/s] 10%|▉         | 38874/400000 [00:04<00:45, 7878.42it/s] 10%|▉         | 39731/400000 [00:04<00:44, 8072.13it/s] 10%|█         | 40568/400000 [00:04<00:44, 8157.67it/s] 10%|█         | 41401/400000 [00:04<00:43, 8207.64it/s] 11%|█         | 42257/400000 [00:05<00:43, 8309.20it/s] 11%|█         | 43114/400000 [00:05<00:42, 8383.97it/s] 11%|█         | 43966/400000 [00:05<00:42, 8422.26it/s] 11%|█         | 44810/400000 [00:05<00:42, 8375.72it/s] 11%|█▏        | 45649/400000 [00:05<00:42, 8357.96it/s] 12%|█▏        | 46497/400000 [00:05<00:42, 8391.16it/s] 12%|█▏        | 47337/400000 [00:05<00:42, 8378.12it/s] 12%|█▏        | 48176/400000 [00:05<00:42, 8360.28it/s] 12%|█▏        | 49023/400000 [00:05<00:41, 8392.71it/s] 12%|█▏        | 49863/400000 [00:05<00:41, 8375.46it/s] 13%|█▎        | 50729/400000 [00:06<00:41, 8456.43it/s] 13%|█▎        | 51600/400000 [00:06<00:40, 8530.60it/s] 13%|█▎        | 52468/400000 [00:06<00:40, 8572.92it/s] 13%|█▎        | 53326/400000 [00:06<00:40, 8524.96it/s] 14%|█▎        | 54179/400000 [00:06<00:40, 8460.95it/s] 14%|█▍        | 55026/400000 [00:06<00:40, 8450.04it/s] 14%|█▍        | 55872/400000 [00:06<00:41, 8296.92it/s] 14%|█▍        | 56737/400000 [00:06<00:40, 8399.71it/s] 14%|█▍        | 57601/400000 [00:06<00:40, 8468.86it/s] 15%|█▍        | 58449/400000 [00:06<00:40, 8460.81it/s] 15%|█▍        | 59296/400000 [00:07<00:40, 8459.08it/s] 15%|█▌        | 60148/400000 [00:07<00:40, 8475.71it/s] 15%|█▌        | 60999/400000 [00:07<00:39, 8483.32it/s] 15%|█▌        | 61848/400000 [00:07<00:39, 8480.19it/s] 16%|█▌        | 62697/400000 [00:07<00:39, 8477.51it/s] 16%|█▌        | 63545/400000 [00:07<00:40, 8287.59it/s] 16%|█▌        | 64406/400000 [00:07<00:40, 8381.64it/s] 16%|█▋        | 65270/400000 [00:07<00:39, 8456.18it/s] 17%|█▋        | 66138/400000 [00:07<00:39, 8519.88it/s] 17%|█▋        | 66991/400000 [00:07<00:39, 8413.78it/s] 17%|█▋        | 67852/400000 [00:08<00:39, 8470.91it/s] 17%|█▋        | 68715/400000 [00:08<00:38, 8516.53it/s] 17%|█▋        | 69570/400000 [00:08<00:38, 8526.05it/s] 18%|█▊        | 70429/400000 [00:08<00:38, 8544.46it/s] 18%|█▊        | 71284/400000 [00:08<00:38, 8532.76it/s] 18%|█▊        | 72141/400000 [00:08<00:38, 8542.10it/s] 18%|█▊        | 72996/400000 [00:08<00:39, 8347.61it/s] 18%|█▊        | 73850/400000 [00:08<00:38, 8403.49it/s] 19%|█▊        | 74692/400000 [00:08<00:38, 8384.16it/s] 19%|█▉        | 75532/400000 [00:09<00:39, 8285.71it/s] 19%|█▉        | 76365/400000 [00:09<00:39, 8296.96it/s] 19%|█▉        | 77198/400000 [00:09<00:38, 8306.23it/s] 20%|█▉        | 78064/400000 [00:09<00:38, 8408.94it/s] 20%|█▉        | 78906/400000 [00:09<00:38, 8336.46it/s] 20%|█▉        | 79741/400000 [00:09<00:39, 8125.55it/s] 20%|██        | 80556/400000 [00:09<00:39, 8024.05it/s] 20%|██        | 81405/400000 [00:09<00:39, 8156.55it/s] 21%|██        | 82223/400000 [00:09<00:39, 8082.05it/s] 21%|██        | 83082/400000 [00:09<00:38, 8225.28it/s] 21%|██        | 83909/400000 [00:10<00:38, 8237.25it/s] 21%|██        | 84755/400000 [00:10<00:37, 8299.99it/s] 21%|██▏       | 85590/400000 [00:10<00:37, 8314.27it/s] 22%|██▏       | 86423/400000 [00:10<00:37, 8293.52it/s] 22%|██▏       | 87261/400000 [00:10<00:37, 8318.19it/s] 22%|██▏       | 88104/400000 [00:10<00:37, 8349.92it/s] 22%|██▏       | 88940/400000 [00:10<00:38, 8121.52it/s] 22%|██▏       | 89801/400000 [00:10<00:37, 8259.76it/s] 23%|██▎       | 90663/400000 [00:10<00:36, 8361.96it/s] 23%|██▎       | 91501/400000 [00:10<00:36, 8363.95it/s] 23%|██▎       | 92339/400000 [00:11<00:37, 8149.31it/s] 23%|██▎       | 93166/400000 [00:11<00:37, 8183.08it/s] 24%|██▎       | 94034/400000 [00:11<00:36, 8325.35it/s] 24%|██▎       | 94869/400000 [00:11<00:37, 8226.24it/s] 24%|██▍       | 95693/400000 [00:11<00:38, 7981.10it/s] 24%|██▍       | 96544/400000 [00:11<00:37, 8132.66it/s] 24%|██▍       | 97390/400000 [00:11<00:36, 8226.86it/s] 25%|██▍       | 98252/400000 [00:11<00:36, 8340.89it/s] 25%|██▍       | 99109/400000 [00:11<00:35, 8406.07it/s] 25%|██▍       | 99951/400000 [00:11<00:35, 8378.84it/s] 25%|██▌       | 100816/400000 [00:12<00:35, 8455.64it/s] 25%|██▌       | 101663/400000 [00:12<00:35, 8441.70it/s] 26%|██▌       | 102528/400000 [00:12<00:34, 8502.48it/s] 26%|██▌       | 103395/400000 [00:12<00:34, 8551.07it/s] 26%|██▌       | 104251/400000 [00:12<00:34, 8523.41it/s] 26%|██▋       | 105111/400000 [00:12<00:34, 8545.36it/s] 26%|██▋       | 105966/400000 [00:12<00:37, 7814.86it/s] 27%|██▋       | 106760/400000 [00:12<00:38, 7716.31it/s] 27%|██▋       | 107599/400000 [00:12<00:36, 7906.30it/s] 27%|██▋       | 108439/400000 [00:13<00:36, 8046.52it/s] 27%|██▋       | 109304/400000 [00:13<00:35, 8216.98it/s] 28%|██▊       | 110155/400000 [00:13<00:34, 8302.65it/s] 28%|██▊       | 111013/400000 [00:13<00:34, 8381.19it/s] 28%|██▊       | 111865/400000 [00:13<00:34, 8419.47it/s] 28%|██▊       | 112709/400000 [00:13<00:34, 8338.89it/s] 28%|██▊       | 113551/400000 [00:13<00:34, 8361.02it/s] 29%|██▊       | 114389/400000 [00:13<00:34, 8346.88it/s] 29%|██▉       | 115246/400000 [00:13<00:33, 8412.23it/s] 29%|██▉       | 116088/400000 [00:13<00:34, 8273.45it/s] 29%|██▉       | 116939/400000 [00:14<00:33, 8342.70it/s] 29%|██▉       | 117808/400000 [00:14<00:33, 8441.59it/s] 30%|██▉       | 118656/400000 [00:14<00:33, 8451.32it/s] 30%|██▉       | 119510/400000 [00:14<00:33, 8476.54it/s] 30%|███       | 120359/400000 [00:14<00:33, 8462.94it/s] 30%|███       | 121206/400000 [00:14<00:33, 8435.43it/s] 31%|███       | 122050/400000 [00:14<00:33, 8405.53it/s] 31%|███       | 122891/400000 [00:14<00:33, 8274.76it/s] 31%|███       | 123757/400000 [00:14<00:32, 8386.68it/s] 31%|███       | 124617/400000 [00:14<00:32, 8448.51it/s] 31%|███▏      | 125463/400000 [00:15<00:32, 8405.28it/s] 32%|███▏      | 126318/400000 [00:15<00:32, 8445.33it/s] 32%|███▏      | 127163/400000 [00:15<00:32, 8357.18it/s] 32%|███▏      | 128000/400000 [00:15<00:32, 8311.49it/s] 32%|███▏      | 128860/400000 [00:15<00:32, 8394.02it/s] 32%|███▏      | 129729/400000 [00:15<00:31, 8479.85it/s] 33%|███▎      | 130586/400000 [00:15<00:31, 8504.01it/s] 33%|███▎      | 131437/400000 [00:15<00:32, 8381.30it/s] 33%|███▎      | 132301/400000 [00:15<00:31, 8455.46it/s] 33%|███▎      | 133148/400000 [00:15<00:32, 8269.30it/s] 33%|███▎      | 133982/400000 [00:16<00:32, 8289.61it/s] 34%|███▎      | 134853/400000 [00:16<00:31, 8409.63it/s] 34%|███▍      | 135723/400000 [00:16<00:31, 8492.99it/s] 34%|███▍      | 136589/400000 [00:16<00:30, 8540.96it/s] 34%|███▍      | 137453/400000 [00:16<00:30, 8567.80it/s] 35%|███▍      | 138315/400000 [00:16<00:30, 8581.70it/s] 35%|███▍      | 139196/400000 [00:16<00:30, 8646.81it/s] 35%|███▌      | 140062/400000 [00:16<00:30, 8519.07it/s] 35%|███▌      | 140915/400000 [00:16<00:30, 8473.90it/s] 35%|███▌      | 141763/400000 [00:16<00:30, 8466.58it/s] 36%|███▌      | 142623/400000 [00:17<00:30, 8505.66it/s] 36%|███▌      | 143504/400000 [00:17<00:29, 8593.62it/s] 36%|███▌      | 144364/400000 [00:17<00:29, 8557.15it/s] 36%|███▋      | 145221/400000 [00:17<00:30, 8451.36it/s] 37%|███▋      | 146093/400000 [00:17<00:29, 8529.79it/s] 37%|███▋      | 146967/400000 [00:17<00:29, 8591.55it/s] 37%|███▋      | 147833/400000 [00:17<00:29, 8611.47it/s] 37%|███▋      | 148695/400000 [00:17<00:29, 8554.96it/s] 37%|███▋      | 149551/400000 [00:17<00:29, 8533.67it/s] 38%|███▊      | 150405/400000 [00:17<00:29, 8475.28it/s] 38%|███▊      | 151253/400000 [00:18<00:29, 8464.64it/s] 38%|███▊      | 152101/400000 [00:18<00:29, 8466.30it/s] 38%|███▊      | 152949/400000 [00:18<00:29, 8469.46it/s] 38%|███▊      | 153797/400000 [00:18<00:29, 8425.72it/s] 39%|███▊      | 154671/400000 [00:18<00:28, 8517.35it/s] 39%|███▉      | 155551/400000 [00:18<00:28, 8600.06it/s] 39%|███▉      | 156412/400000 [00:18<00:28, 8576.58it/s] 39%|███▉      | 157270/400000 [00:18<00:28, 8395.89it/s] 40%|███▉      | 158143/400000 [00:18<00:28, 8493.36it/s] 40%|███▉      | 159031/400000 [00:18<00:28, 8603.96it/s] 40%|███▉      | 159893/400000 [00:19<00:28, 8492.19it/s] 40%|████      | 160751/400000 [00:19<00:28, 8518.03it/s] 40%|████      | 161604/400000 [00:19<00:28, 8491.59it/s] 41%|████      | 162492/400000 [00:19<00:27, 8603.53it/s] 41%|████      | 163367/400000 [00:19<00:27, 8644.68it/s] 41%|████      | 164233/400000 [00:19<00:27, 8624.40it/s] 41%|████▏     | 165096/400000 [00:19<00:27, 8575.49it/s] 41%|████▏     | 165954/400000 [00:19<00:27, 8543.63it/s] 42%|████▏     | 166809/400000 [00:19<00:27, 8490.03it/s] 42%|████▏     | 167664/400000 [00:19<00:27, 8506.54it/s] 42%|████▏     | 168515/400000 [00:20<00:27, 8462.05it/s] 42%|████▏     | 169362/400000 [00:20<00:27, 8378.17it/s] 43%|████▎     | 170201/400000 [00:20<00:27, 8339.85it/s] 43%|████▎     | 171064/400000 [00:20<00:27, 8423.61it/s] 43%|████▎     | 171923/400000 [00:20<00:26, 8472.82it/s] 43%|████▎     | 172771/400000 [00:20<00:26, 8419.62it/s] 43%|████▎     | 173638/400000 [00:20<00:26, 8492.01it/s] 44%|████▎     | 174488/400000 [00:20<00:27, 8280.96it/s] 44%|████▍     | 175353/400000 [00:20<00:26, 8385.95it/s] 44%|████▍     | 176213/400000 [00:21<00:26, 8447.56it/s] 44%|████▍     | 177063/400000 [00:21<00:26, 8462.84it/s] 44%|████▍     | 177923/400000 [00:21<00:26, 8503.51it/s] 45%|████▍     | 178774/400000 [00:21<00:26, 8475.05it/s] 45%|████▍     | 179632/400000 [00:21<00:25, 8505.68it/s] 45%|████▌     | 180491/400000 [00:21<00:25, 8529.52it/s] 45%|████▌     | 181351/400000 [00:21<00:25, 8549.24it/s] 46%|████▌     | 182217/400000 [00:21<00:25, 8579.91it/s] 46%|████▌     | 183076/400000 [00:21<00:25, 8516.59it/s] 46%|████▌     | 183933/400000 [00:21<00:25, 8532.18it/s] 46%|████▌     | 184787/400000 [00:22<00:25, 8518.71it/s] 46%|████▋     | 185639/400000 [00:22<00:25, 8474.30it/s] 47%|████▋     | 186487/400000 [00:22<00:25, 8399.30it/s] 47%|████▋     | 187328/400000 [00:22<00:25, 8401.35it/s] 47%|████▋     | 188179/400000 [00:22<00:25, 8431.84it/s] 47%|████▋     | 189023/400000 [00:22<00:25, 8410.06it/s] 47%|████▋     | 189865/400000 [00:22<00:25, 8350.07it/s] 48%|████▊     | 190722/400000 [00:22<00:24, 8411.81it/s] 48%|████▊     | 191564/400000 [00:22<00:25, 8280.27it/s] 48%|████▊     | 192433/400000 [00:22<00:24, 8397.64it/s] 48%|████▊     | 193308/400000 [00:23<00:24, 8497.76it/s] 49%|████▊     | 194159/400000 [00:23<00:24, 8498.12it/s] 49%|████▉     | 195046/400000 [00:23<00:23, 8603.69it/s] 49%|████▉     | 195921/400000 [00:23<00:23, 8644.88it/s] 49%|████▉     | 196793/400000 [00:23<00:23, 8666.21it/s] 49%|████▉     | 197664/400000 [00:23<00:23, 8678.16it/s] 50%|████▉     | 198558/400000 [00:23<00:23, 8753.33it/s] 50%|████▉     | 199441/400000 [00:23<00:22, 8774.66it/s] 50%|█████     | 200319/400000 [00:23<00:23, 8673.10it/s] 50%|█████     | 201188/400000 [00:23<00:22, 8677.33it/s] 51%|█████     | 202070/400000 [00:24<00:22, 8718.45it/s] 51%|█████     | 202943/400000 [00:24<00:22, 8708.97it/s] 51%|█████     | 203815/400000 [00:24<00:22, 8708.83it/s] 51%|█████     | 204687/400000 [00:24<00:22, 8591.06it/s] 51%|█████▏    | 205547/400000 [00:24<00:22, 8557.86it/s] 52%|█████▏    | 206404/400000 [00:24<00:22, 8466.81it/s] 52%|█████▏    | 207252/400000 [00:24<00:22, 8398.16it/s] 52%|█████▏    | 208095/400000 [00:24<00:22, 8406.91it/s] 52%|█████▏    | 208937/400000 [00:24<00:23, 8272.47it/s] 52%|█████▏    | 209828/400000 [00:24<00:22, 8452.27it/s] 53%|█████▎    | 210717/400000 [00:25<00:22, 8571.47it/s] 53%|█████▎    | 211597/400000 [00:25<00:21, 8637.83it/s] 53%|█████▎    | 212482/400000 [00:25<00:21, 8700.28it/s] 53%|█████▎    | 213353/400000 [00:25<00:21, 8687.01it/s] 54%|█████▎    | 214223/400000 [00:25<00:21, 8677.85it/s] 54%|█████▍    | 215094/400000 [00:25<00:21, 8685.71it/s] 54%|█████▍    | 215963/400000 [00:25<00:21, 8678.33it/s] 54%|█████▍    | 216832/400000 [00:25<00:21, 8655.89it/s] 54%|█████▍    | 217698/400000 [00:25<00:21, 8598.79it/s] 55%|█████▍    | 218580/400000 [00:25<00:20, 8661.98it/s] 55%|█████▍    | 219447/400000 [00:26<00:21, 8585.50it/s] 55%|█████▌    | 220309/400000 [00:26<00:20, 8594.47it/s] 55%|█████▌    | 221185/400000 [00:26<00:20, 8643.02it/s] 56%|█████▌    | 222058/400000 [00:26<00:20, 8668.04it/s] 56%|█████▌    | 222929/400000 [00:26<00:20, 8679.49it/s] 56%|█████▌    | 223798/400000 [00:26<00:20, 8655.78it/s] 56%|█████▌    | 224664/400000 [00:26<00:20, 8584.08it/s] 56%|█████▋    | 225523/400000 [00:26<00:20, 8546.79it/s] 57%|█████▋    | 226378/400000 [00:26<00:20, 8503.10it/s] 57%|█████▋    | 227240/400000 [00:26<00:20, 8535.23it/s] 57%|█████▋    | 228094/400000 [00:27<00:20, 8474.04it/s] 57%|█████▋    | 228968/400000 [00:27<00:20, 8550.25it/s] 57%|█████▋    | 229831/400000 [00:27<00:19, 8571.88it/s] 58%|█████▊    | 230689/400000 [00:27<00:20, 8401.46it/s] 58%|█████▊    | 231540/400000 [00:27<00:19, 8433.65it/s] 58%|█████▊    | 232401/400000 [00:27<00:19, 8484.37it/s] 58%|█████▊    | 233250/400000 [00:27<00:19, 8417.17it/s] 59%|█████▊    | 234107/400000 [00:27<00:19, 8459.86it/s] 59%|█████▊    | 234971/400000 [00:27<00:19, 8512.75it/s] 59%|█████▉    | 235846/400000 [00:27<00:19, 8582.16it/s] 59%|█████▉    | 236717/400000 [00:28<00:18, 8619.75it/s] 59%|█████▉    | 237580/400000 [00:28<00:19, 8542.15it/s] 60%|█████▉    | 238435/400000 [00:28<00:19, 8377.47it/s] 60%|█████▉    | 239294/400000 [00:28<00:19, 8437.68it/s] 60%|██████    | 240144/400000 [00:28<00:18, 8455.17it/s] 60%|██████    | 241012/400000 [00:28<00:18, 8520.09it/s] 60%|██████    | 241889/400000 [00:28<00:18, 8591.77it/s] 61%|██████    | 242750/400000 [00:28<00:18, 8596.83it/s] 61%|██████    | 243618/400000 [00:28<00:18, 8621.55it/s] 61%|██████    | 244481/400000 [00:28<00:18, 8589.81it/s] 61%|██████▏   | 245341/400000 [00:29<00:18, 8583.73it/s] 62%|██████▏   | 246228/400000 [00:29<00:17, 8666.16it/s] 62%|██████▏   | 247095/400000 [00:29<00:17, 8626.82it/s] 62%|██████▏   | 247958/400000 [00:29<00:17, 8556.44it/s] 62%|██████▏   | 248814/400000 [00:29<00:17, 8435.48it/s] 62%|██████▏   | 249686/400000 [00:29<00:17, 8518.26it/s] 63%|██████▎   | 250539/400000 [00:29<00:17, 8440.70it/s] 63%|██████▎   | 251384/400000 [00:29<00:17, 8441.85it/s] 63%|██████▎   | 252246/400000 [00:29<00:17, 8492.20it/s] 63%|██████▎   | 253096/400000 [00:29<00:17, 8490.83it/s] 63%|██████▎   | 253946/400000 [00:30<00:17, 8423.32it/s] 64%|██████▎   | 254789/400000 [00:30<00:17, 8345.23it/s] 64%|██████▍   | 255659/400000 [00:30<00:17, 8447.82it/s] 64%|██████▍   | 256528/400000 [00:30<00:16, 8517.38it/s] 64%|██████▍   | 257388/400000 [00:30<00:16, 8541.76it/s] 65%|██████▍   | 258257/400000 [00:30<00:16, 8584.03it/s] 65%|██████▍   | 259116/400000 [00:30<00:16, 8510.97it/s] 65%|██████▍   | 259968/400000 [00:30<00:16, 8381.32it/s] 65%|██████▌   | 260853/400000 [00:30<00:16, 8515.87it/s] 65%|██████▌   | 261718/400000 [00:31<00:16, 8554.81it/s] 66%|██████▌   | 262575/400000 [00:31<00:16, 8458.98it/s] 66%|██████▌   | 263448/400000 [00:31<00:16, 8533.87it/s] 66%|██████▌   | 264317/400000 [00:31<00:15, 8579.03it/s] 66%|██████▋   | 265190/400000 [00:31<00:15, 8621.08it/s] 67%|██████▋   | 266053/400000 [00:31<00:15, 8616.90it/s] 67%|██████▋   | 266915/400000 [00:31<00:15, 8603.98it/s] 67%|██████▋   | 267796/400000 [00:31<00:15, 8663.18it/s] 67%|██████▋   | 268671/400000 [00:31<00:15, 8687.02it/s] 67%|██████▋   | 269540/400000 [00:31<00:15, 8687.24it/s] 68%|██████▊   | 270409/400000 [00:32<00:15, 8353.66it/s] 68%|██████▊   | 271248/400000 [00:32<00:15, 8348.11it/s] 68%|██████▊   | 272116/400000 [00:32<00:15, 8444.32it/s] 68%|██████▊   | 272975/400000 [00:32<00:14, 8486.83it/s] 68%|██████▊   | 273855/400000 [00:32<00:14, 8576.03it/s] 69%|██████▊   | 274737/400000 [00:32<00:14, 8646.78it/s] 69%|██████▉   | 275603/400000 [00:32<00:14, 8577.78it/s] 69%|██████▉   | 276462/400000 [00:32<00:14, 8402.27it/s] 69%|██████▉   | 277304/400000 [00:32<00:14, 8285.82it/s] 70%|██████▉   | 278141/400000 [00:32<00:14, 8309.58it/s] 70%|██████▉   | 278998/400000 [00:33<00:14, 8383.32it/s] 70%|██████▉   | 279888/400000 [00:33<00:14, 8531.88it/s] 70%|███████   | 280765/400000 [00:33<00:13, 8600.17it/s] 70%|███████   | 281626/400000 [00:33<00:13, 8539.74it/s] 71%|███████   | 282497/400000 [00:33<00:13, 8587.36it/s] 71%|███████   | 283357/400000 [00:33<00:13, 8565.45it/s] 71%|███████   | 284214/400000 [00:33<00:13, 8536.92it/s] 71%|███████▏  | 285069/400000 [00:33<00:13, 8539.64it/s] 71%|███████▏  | 285924/400000 [00:33<00:13, 8419.06it/s] 72%|███████▏  | 286767/400000 [00:33<00:13, 8232.48it/s] 72%|███████▏  | 287620/400000 [00:34<00:13, 8317.87it/s] 72%|███████▏  | 288453/400000 [00:34<00:13, 8295.43it/s] 72%|███████▏  | 289312/400000 [00:34<00:13, 8381.24it/s] 73%|███████▎  | 290185/400000 [00:34<00:12, 8481.33it/s] 73%|███████▎  | 291057/400000 [00:34<00:12, 8549.51it/s] 73%|███████▎  | 291925/400000 [00:34<00:12, 8586.63it/s] 73%|███████▎  | 292802/400000 [00:34<00:12, 8640.19it/s] 73%|███████▎  | 293675/400000 [00:34<00:12, 8665.20it/s] 74%|███████▎  | 294542/400000 [00:34<00:12, 8470.86it/s] 74%|███████▍  | 295429/400000 [00:34<00:12, 8585.89it/s] 74%|███████▍  | 296303/400000 [00:35<00:12, 8629.00it/s] 74%|███████▍  | 297167/400000 [00:35<00:11, 8631.37it/s] 75%|███████▍  | 298056/400000 [00:35<00:11, 8704.84it/s] 75%|███████▍  | 298928/400000 [00:35<00:11, 8677.11it/s] 75%|███████▍  | 299797/400000 [00:35<00:11, 8594.62it/s] 75%|███████▌  | 300663/400000 [00:35<00:11, 8613.33it/s] 75%|███████▌  | 301525/400000 [00:35<00:11, 8559.32it/s] 76%|███████▌  | 302382/400000 [00:35<00:11, 8419.78it/s] 76%|███████▌  | 303225/400000 [00:35<00:11, 8198.13it/s] 76%|███████▌  | 304116/400000 [00:35<00:11, 8398.34it/s] 76%|███████▌  | 304994/400000 [00:36<00:11, 8506.70it/s] 76%|███████▋  | 305863/400000 [00:36<00:10, 8558.45it/s] 77%|███████▋  | 306726/400000 [00:36<00:10, 8578.86it/s] 77%|███████▋  | 307598/400000 [00:36<00:10, 8618.93it/s] 77%|███████▋  | 308487/400000 [00:36<00:10, 8697.54it/s] 77%|███████▋  | 309358/400000 [00:36<00:10, 8436.90it/s] 78%|███████▊  | 310204/400000 [00:36<00:10, 8377.08it/s] 78%|███████▊  | 311044/400000 [00:36<00:10, 8194.56it/s] 78%|███████▊  | 311897/400000 [00:36<00:10, 8292.30it/s] 78%|███████▊  | 312728/400000 [00:37<00:10, 8182.94it/s] 78%|███████▊  | 313591/400000 [00:37<00:10, 8309.52it/s] 79%|███████▊  | 314431/400000 [00:37<00:10, 8334.04it/s] 79%|███████▉  | 315276/400000 [00:37<00:10, 8366.00it/s] 79%|███████▉  | 316114/400000 [00:37<00:10, 8199.14it/s] 79%|███████▉  | 316936/400000 [00:37<00:10, 7895.44it/s] 79%|███████▉  | 317762/400000 [00:37<00:10, 7999.42it/s] 80%|███████▉  | 318615/400000 [00:37<00:09, 8148.95it/s] 80%|███████▉  | 319443/400000 [00:37<00:09, 8186.76it/s] 80%|████████  | 320286/400000 [00:37<00:09, 8258.06it/s] 80%|████████  | 321157/400000 [00:38<00:09, 8388.11it/s] 80%|████████  | 321998/400000 [00:38<00:09, 8368.97it/s] 81%|████████  | 322836/400000 [00:38<00:09, 8252.54it/s] 81%|████████  | 323688/400000 [00:38<00:09, 8330.53it/s] 81%|████████  | 324539/400000 [00:38<00:09, 8383.13it/s] 81%|████████▏ | 325379/400000 [00:38<00:08, 8387.31it/s] 82%|████████▏ | 326219/400000 [00:38<00:08, 8304.49it/s] 82%|████████▏ | 327051/400000 [00:38<00:08, 8254.94it/s] 82%|████████▏ | 327877/400000 [00:38<00:08, 8153.18it/s] 82%|████████▏ | 328719/400000 [00:38<00:08, 8230.56it/s] 82%|████████▏ | 329572/400000 [00:39<00:08, 8315.69it/s] 83%|████████▎ | 330443/400000 [00:39<00:08, 8428.97it/s] 83%|████████▎ | 331322/400000 [00:39<00:08, 8532.33it/s] 83%|████████▎ | 332191/400000 [00:39<00:07, 8576.79it/s] 83%|████████▎ | 333050/400000 [00:39<00:07, 8559.93it/s] 83%|████████▎ | 333921/400000 [00:39<00:07, 8601.94it/s] 84%|████████▎ | 334782/400000 [00:39<00:07, 8556.12it/s] 84%|████████▍ | 335641/400000 [00:39<00:07, 8565.89it/s] 84%|████████▍ | 336502/400000 [00:39<00:07, 8577.87it/s] 84%|████████▍ | 337393/400000 [00:39<00:07, 8673.05it/s] 85%|████████▍ | 338261/400000 [00:40<00:07, 8574.42it/s] 85%|████████▍ | 339119/400000 [00:40<00:07, 8348.61it/s] 85%|████████▍ | 339960/400000 [00:40<00:07, 8366.68it/s] 85%|████████▌ | 340804/400000 [00:40<00:07, 8388.34it/s] 85%|████████▌ | 341644/400000 [00:40<00:07, 8201.61it/s] 86%|████████▌ | 342519/400000 [00:40<00:06, 8357.24it/s] 86%|████████▌ | 343359/400000 [00:40<00:06, 8367.94it/s] 86%|████████▌ | 344227/400000 [00:40<00:06, 8458.41it/s] 86%|████████▋ | 345074/400000 [00:40<00:06, 8340.85it/s] 86%|████████▋ | 345921/400000 [00:40<00:06, 8376.59it/s] 87%|████████▋ | 346784/400000 [00:41<00:06, 8450.25it/s] 87%|████████▋ | 347630/400000 [00:41<00:06, 8447.25it/s] 87%|████████▋ | 348476/400000 [00:41<00:06, 8446.21it/s] 87%|████████▋ | 349321/400000 [00:41<00:06, 8408.76it/s] 88%|████████▊ | 350203/400000 [00:41<00:05, 8527.70it/s] 88%|████████▊ | 351057/400000 [00:41<00:05, 8489.54it/s] 88%|████████▊ | 351924/400000 [00:41<00:05, 8540.44it/s] 88%|████████▊ | 352786/400000 [00:41<00:05, 8561.54it/s] 88%|████████▊ | 353656/400000 [00:41<00:05, 8600.12it/s] 89%|████████▊ | 354547/400000 [00:41<00:05, 8689.74it/s] 89%|████████▉ | 355429/400000 [00:42<00:05, 8725.84it/s] 89%|████████▉ | 356303/400000 [00:42<00:05, 8729.38it/s] 89%|████████▉ | 357177/400000 [00:42<00:04, 8603.02it/s] 90%|████████▉ | 358038/400000 [00:42<00:04, 8533.62it/s] 90%|████████▉ | 358892/400000 [00:42<00:04, 8375.36it/s] 90%|████████▉ | 359731/400000 [00:42<00:04, 8241.01it/s] 90%|█████████ | 360591/400000 [00:42<00:04, 8345.31it/s] 90%|█████████ | 361427/400000 [00:42<00:04, 8140.42it/s] 91%|█████████ | 362245/400000 [00:42<00:04, 8150.41it/s] 91%|█████████ | 363093/400000 [00:43<00:04, 8244.41it/s] 91%|█████████ | 363944/400000 [00:43<00:04, 8320.74it/s] 91%|█████████ | 364798/400000 [00:43<00:04, 8383.63it/s] 91%|█████████▏| 365669/400000 [00:43<00:04, 8478.50it/s] 92%|█████████▏| 366524/400000 [00:43<00:03, 8498.12it/s] 92%|█████████▏| 367402/400000 [00:43<00:03, 8579.63it/s] 92%|█████████▏| 368261/400000 [00:43<00:03, 8538.35it/s] 92%|█████████▏| 369124/400000 [00:43<00:03, 8563.10it/s] 92%|█████████▏| 369986/400000 [00:43<00:03, 8579.57it/s] 93%|█████████▎| 370845/400000 [00:43<00:03, 8557.67it/s] 93%|█████████▎| 371701/400000 [00:44<00:03, 8507.23it/s] 93%|█████████▎| 372569/400000 [00:44<00:03, 8556.46it/s] 93%|█████████▎| 373425/400000 [00:44<00:03, 8533.03it/s] 94%|█████████▎| 374279/400000 [00:44<00:03, 8528.36it/s] 94%|█████████▍| 375132/400000 [00:44<00:02, 8500.33it/s] 94%|█████████▍| 375995/400000 [00:44<00:02, 8537.18it/s] 94%|█████████▍| 376849/400000 [00:44<00:02, 8507.70it/s] 94%|█████████▍| 377723/400000 [00:44<00:02, 8574.14it/s] 95%|█████████▍| 378591/400000 [00:44<00:02, 8605.55it/s] 95%|█████████▍| 379452/400000 [00:44<00:02, 8484.19it/s] 95%|█████████▌| 380301/400000 [00:45<00:02, 8451.46it/s] 95%|█████████▌| 381182/400000 [00:45<00:02, 8553.51it/s] 96%|█████████▌| 382074/400000 [00:45<00:02, 8658.85it/s] 96%|█████████▌| 382955/400000 [00:45<00:01, 8702.10it/s] 96%|█████████▌| 383826/400000 [00:45<00:01, 8549.63it/s] 96%|█████████▌| 384682/400000 [00:45<00:01, 8387.86it/s] 96%|█████████▋| 385552/400000 [00:45<00:01, 8477.21it/s] 97%|█████████▋| 386401/400000 [00:45<00:01, 8410.57it/s] 97%|█████████▋| 387277/400000 [00:45<00:01, 8510.06it/s] 97%|█████████▋| 388129/400000 [00:45<00:01, 8456.82it/s] 97%|█████████▋| 389027/400000 [00:46<00:01, 8605.33it/s] 97%|█████████▋| 389889/400000 [00:46<00:01, 8607.73it/s] 98%|█████████▊| 390751/400000 [00:46<00:01, 8518.35it/s] 98%|█████████▊| 391604/400000 [00:46<00:01, 8173.64it/s] 98%|█████████▊| 392425/400000 [00:46<00:00, 7824.15it/s] 98%|█████████▊| 393255/400000 [00:46<00:00, 7941.30it/s] 99%|█████████▊| 394054/400000 [00:46<00:00, 7945.69it/s] 99%|█████████▊| 394901/400000 [00:46<00:00, 8093.63it/s] 99%|█████████▉| 395754/400000 [00:46<00:00, 8219.66it/s] 99%|█████████▉| 396579/400000 [00:46<00:00, 8160.92it/s] 99%|█████████▉| 397397/400000 [00:47<00:00, 8125.27it/s]100%|█████████▉| 398225/400000 [00:47<00:00, 8170.14it/s]100%|█████████▉| 399043/400000 [00:47<00:00, 7957.25it/s]100%|█████████▉| 399860/400000 [00:47<00:00, 8019.42it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8435.65it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f4acf0cbb38>, <torchtext.data.dataset.TabularDataset object at 0x7f4acf0cbc88>, <torchtext.vocab.Vocab object at 0x7f4acf0cbba8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.11 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.11 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.11 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.41 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.41 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.99 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.99 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.86 file/s]2020-07-07 00:20:21.062561: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-07 00:20:21.067090: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-07-07 00:20:21.067278: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b0f31b83f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-07 00:20:21.067294: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 133855.38it/s] 71%|███████   | 7012352/9912422 [00:00<00:15, 191065.11it/s]9920512it [00:00, 40554140.55it/s]                           
0it [00:00, ?it/s]32768it [00:00, 593742.64it/s]
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 944597.62it/s]1654784it [00:00, 12114189.08it/s]                         
0it [00:00, ?it/s]8192it [00:00, 157080.27it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
