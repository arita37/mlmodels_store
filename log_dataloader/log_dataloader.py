
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7efcc9a340d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7efcc9a340d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7efd355db2f0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7efd355db2f0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7efd53920e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7efd53920e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efce2902730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efce2902730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efce2902730> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3522560/9912422 [00:00<00:00, 35190563.01it/s]9920512it [00:00, 35290941.29it/s]                             
0it [00:00, ?it/s]32768it [00:00, 429665.91it/s]
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 966661.42it/s]1654784it [00:00, 12225446.48it/s]                         
0it [00:00, ?it/s]8192it [00:00, 198395.61it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efcc04b4ac8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efcc04c4d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7efce2902378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7efce2902378> 

  function with postional parmater data_info <function tf_dataset_download at 0x7efce2902378> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:16,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:15,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:14,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:36,  4.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:36,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:35,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:17,  7.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:17,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 10.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 10.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:11, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:11, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:08, 14.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:08, 14.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:08, 14.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 14.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 14.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 14.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 14.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 14.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 18.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 18.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 18.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 18.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 18.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 18.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 18.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 24.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 24.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 24.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 24.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 24.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 24.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 24.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 24.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 29.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 29.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 29.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 29.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 29.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 29.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 29.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 29.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 34.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 34.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 34.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 34.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 34.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 36.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 36.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 36.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 36.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 36.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:02, 33.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:02, 33.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:02, 33.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:02, 31.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:02, 30.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:02, 30.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:02, 30.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:02, 30.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:02, 30.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:02, 29.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:02, 29.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:02, 29.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:02, 29.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:02, 29.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:02, 25.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:02, 22.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:02, 22.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:02, 22.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:02, 22.64 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:02, 19.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:02, 19.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:02, 19.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:02, 19.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:03, 15.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:03, 15.50 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:03, 15.50 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:03, 13.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:03, 13.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:03, 13.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:03<00:03, 12.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:03, 12.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:03, 12.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:03, 12.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:03, 12.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:03, 12.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:04<00:03, 12.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:04<00:03, 12.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:04<00:03, 12.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:04<00:03, 11.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:04<00:03, 11.99 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:04<00:03, 11.99 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:04<00:03, 12.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:04<00:03, 12.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:04<00:03, 12.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:04<00:03, 11.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:04<00:03, 11.38 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:04<00:03, 11.38 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:04<00:03, 11.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:04<00:03, 11.48 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:02, 11.48 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:04<00:02, 11.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:04<00:02, 11.97 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:04<00:02, 11.97 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:05<00:02, 11.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:05<00:02, 11.84 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:05<00:02, 11.84 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:05<00:02, 10.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:05<00:02, 10.85 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:05<00:02, 10.85 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:05<00:02, 10.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:05<00:02, 10.34 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:05<00:02, 10.34 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:05<00:02, 10.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:05<00:02, 10.19 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:05<00:02, 10.19 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:05<00:02, 10.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:05<00:02, 10.03 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:06<00:02, 10.03 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:06<00:02,  9.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:06<00:02,  9.93 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:06<00:02,  9.93 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:06<00:01,  9.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:06<00:01,  9.98 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:06<00:01,  9.98 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:06<00:01, 10.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:06<00:01, 10.13 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:06<00:01, 10.13 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:06<00:01, 10.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:06<00:01, 10.31 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:06<00:01, 10.31 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:06<00:01, 10.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:06<00:01, 10.42 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:07<00:01, 10.42 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:07<00:01, 10.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:07<00:01, 10.17 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:07<00:00, 10.17 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:07<00:00,  9.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:07<00:00,  9.44 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:07<00:00,  9.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:07<00:00,  9.16 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:07<00:00,  8.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:07<00:00,  8.89 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:07<00:00,  8.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:07<00:00,  8.93 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:07<00:00,  8.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:07<00:00,  8.92 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:07<00:00,  8.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:07<00:00,  8.96 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:08<00:00,  9.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:08<00:00,  9.04 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:08<00:00,  9.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:08<00:00,  9.26 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:08<00:00,  9.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:08<00:00,  9.19 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:08<00:00,  9.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00,  9.35 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.37s/ url]Dl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00,  9.35 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00,  9.35 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:08<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.70s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:10<00:00,  8.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:10<00:00,  9.35 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.70s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.70s/ file]
Dl Size...: 100%|██████████| 162/162 [00:10<00:00, 15.14 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:10<00:00, 10.70s/ url]
0 examples [00:00, ? examples/s]2020-07-16 18:10:43.740676: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-16 18:10:43.756104: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294675000 Hz
2020-07-16 18:10:43.756442: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556dbdf78830 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-16 18:10:43.756464: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
5 examples [00:00, 49.54 examples/s]93 examples [00:00, 69.10 examples/s]182 examples [00:00, 95.54 examples/s]269 examples [00:00, 130.22 examples/s]359 examples [00:00, 175.16 examples/s]449 examples [00:00, 230.97 examples/s]541 examples [00:00, 297.65 examples/s]630 examples [00:00, 371.32 examples/s]718 examples [00:00, 448.67 examples/s]804 examples [00:01, 523.52 examples/s]892 examples [00:01, 594.90 examples/s]980 examples [00:01, 657.67 examples/s]1067 examples [00:01, 709.37 examples/s]1154 examples [00:01, 740.60 examples/s]1240 examples [00:01, 770.52 examples/s]1327 examples [00:01, 795.68 examples/s]1415 examples [00:01, 818.22 examples/s]1502 examples [00:01, 828.78 examples/s]1588 examples [00:01, 829.32 examples/s]1674 examples [00:02, 835.97 examples/s]1760 examples [00:02, 841.04 examples/s]1846 examples [00:02, 844.30 examples/s]1933 examples [00:02, 849.27 examples/s]2019 examples [00:02, 845.66 examples/s]2108 examples [00:02, 857.15 examples/s]2195 examples [00:02, 859.86 examples/s]2282 examples [00:02, 862.26 examples/s]2369 examples [00:02, 842.88 examples/s]2457 examples [00:02, 851.39 examples/s]2545 examples [00:03, 859.12 examples/s]2633 examples [00:03, 862.73 examples/s]2721 examples [00:03, 864.60 examples/s]2808 examples [00:03, 866.04 examples/s]2895 examples [00:03, 863.29 examples/s]2982 examples [00:03, 854.35 examples/s]3068 examples [00:03, 850.20 examples/s]3160 examples [00:03, 868.44 examples/s]3251 examples [00:03, 878.51 examples/s]3342 examples [00:03, 883.90 examples/s]3436 examples [00:04, 897.35 examples/s]3530 examples [00:04, 907.44 examples/s]3621 examples [00:04, 903.29 examples/s]3712 examples [00:04, 885.74 examples/s]3801 examples [00:04, 875.14 examples/s]3891 examples [00:04, 881.62 examples/s]3980 examples [00:04, 870.89 examples/s]4068 examples [00:04, 859.78 examples/s]4158 examples [00:04, 870.65 examples/s]4246 examples [00:04, 859.59 examples/s]4333 examples [00:05, 855.17 examples/s]4422 examples [00:05, 863.25 examples/s]4511 examples [00:05, 869.92 examples/s]4599 examples [00:05, 857.17 examples/s]4685 examples [00:05, 851.93 examples/s]4775 examples [00:05, 863.68 examples/s]4862 examples [00:05, 863.21 examples/s]4950 examples [00:05, 867.83 examples/s]5040 examples [00:05, 875.40 examples/s]5128 examples [00:06, 873.45 examples/s]5219 examples [00:06, 882.61 examples/s]5308 examples [00:06, 881.36 examples/s]5398 examples [00:06, 885.98 examples/s]5487 examples [00:06, 879.87 examples/s]5576 examples [00:06, 882.05 examples/s]5667 examples [00:06, 889.46 examples/s]5759 examples [00:06, 897.58 examples/s]5849 examples [00:06, 872.36 examples/s]5937 examples [00:06, 856.72 examples/s]6027 examples [00:07, 866.95 examples/s]6118 examples [00:07, 877.70 examples/s]6210 examples [00:07, 886.69 examples/s]6301 examples [00:07, 890.86 examples/s]6393 examples [00:07, 897.65 examples/s]6483 examples [00:07, 888.29 examples/s]6572 examples [00:07, 880.00 examples/s]6661 examples [00:07, 876.19 examples/s]6750 examples [00:07, 878.19 examples/s]6838 examples [00:07, 872.19 examples/s]6926 examples [00:08, 869.28 examples/s]7015 examples [00:08, 874.62 examples/s]7106 examples [00:08, 884.03 examples/s]7196 examples [00:08, 887.08 examples/s]7287 examples [00:08, 892.34 examples/s]7377 examples [00:08, 883.90 examples/s]7466 examples [00:08, 879.05 examples/s]7558 examples [00:08, 890.22 examples/s]7652 examples [00:08, 902.08 examples/s]7745 examples [00:08, 909.90 examples/s]7837 examples [00:09, 904.38 examples/s]7928 examples [00:09, 902.90 examples/s]8019 examples [00:09, 898.95 examples/s]8109 examples [00:09, 891.26 examples/s]8199 examples [00:09, 892.62 examples/s]8289 examples [00:09, 880.56 examples/s]8378 examples [00:09, 863.07 examples/s]8468 examples [00:09, 873.79 examples/s]8556 examples [00:09, 874.81 examples/s]8644 examples [00:09, 872.51 examples/s]8732 examples [00:10, 849.61 examples/s]8821 examples [00:10, 860.03 examples/s]8911 examples [00:10, 869.23 examples/s]9000 examples [00:10, 873.59 examples/s]9089 examples [00:10, 876.28 examples/s]9177 examples [00:10, 873.62 examples/s]9270 examples [00:10, 887.39 examples/s]9363 examples [00:10, 898.68 examples/s]9455 examples [00:10, 902.50 examples/s]9547 examples [00:10, 905.10 examples/s]9638 examples [00:11, 870.96 examples/s]9727 examples [00:11, 875.71 examples/s]9819 examples [00:11, 886.25 examples/s]9912 examples [00:11, 896.59 examples/s]10002 examples [00:11, 828.42 examples/s]10091 examples [00:11, 845.36 examples/s]10177 examples [00:11, 844.17 examples/s]10270 examples [00:11, 865.70 examples/s]10366 examples [00:11, 891.00 examples/s]10461 examples [00:12, 905.92 examples/s]10553 examples [00:12, 901.99 examples/s]10648 examples [00:12, 913.74 examples/s]10742 examples [00:12, 919.96 examples/s]10835 examples [00:12, 907.66 examples/s]10926 examples [00:12, 908.23 examples/s]11017 examples [00:12, 893.10 examples/s]11113 examples [00:12, 909.93 examples/s]11206 examples [00:12, 915.55 examples/s]11300 examples [00:12, 921.06 examples/s]11395 examples [00:13, 928.37 examples/s]11488 examples [00:13, 910.80 examples/s]11580 examples [00:13, 905.08 examples/s]11671 examples [00:13, 902.35 examples/s]11762 examples [00:13, 889.95 examples/s]11855 examples [00:13, 900.27 examples/s]11947 examples [00:13, 904.13 examples/s]12043 examples [00:13, 919.17 examples/s]12136 examples [00:13, 916.46 examples/s]12228 examples [00:13, 917.08 examples/s]12320 examples [00:14, 916.79 examples/s]12412 examples [00:14, 885.21 examples/s]12502 examples [00:14, 887.14 examples/s]12591 examples [00:14, 883.57 examples/s]12683 examples [00:14, 892.72 examples/s]12773 examples [00:14, 892.99 examples/s]12863 examples [00:14, 883.55 examples/s]12953 examples [00:14, 887.78 examples/s]13046 examples [00:14, 897.09 examples/s]13141 examples [00:15, 912.26 examples/s]13235 examples [00:15, 918.50 examples/s]13327 examples [00:15, 891.31 examples/s]13418 examples [00:15, 894.13 examples/s]13508 examples [00:15, 892.70 examples/s]13598 examples [00:15, 893.01 examples/s]13692 examples [00:15, 905.56 examples/s]13783 examples [00:15, 904.57 examples/s]13877 examples [00:15, 913.98 examples/s]13972 examples [00:15, 922.73 examples/s]14068 examples [00:16, 932.76 examples/s]14162 examples [00:16, 928.14 examples/s]14255 examples [00:16, 920.44 examples/s]14348 examples [00:16, 919.80 examples/s]14441 examples [00:16, 909.08 examples/s]14532 examples [00:16, 904.09 examples/s]14623 examples [00:16, 903.24 examples/s]14714 examples [00:16, 891.82 examples/s]14806 examples [00:16, 899.93 examples/s]14900 examples [00:16, 911.05 examples/s]14996 examples [00:17, 922.64 examples/s]15089 examples [00:17, 922.81 examples/s]15182 examples [00:17, 920.25 examples/s]15275 examples [00:17, 897.88 examples/s]15365 examples [00:17, 895.53 examples/s]15458 examples [00:17, 903.39 examples/s]15551 examples [00:17, 910.55 examples/s]15643 examples [00:17, 909.37 examples/s]15734 examples [00:17, 887.87 examples/s]15826 examples [00:17, 895.73 examples/s]15916 examples [00:18, 892.39 examples/s]16006 examples [00:18, 877.43 examples/s]16096 examples [00:18, 883.99 examples/s]16186 examples [00:18, 886.97 examples/s]16279 examples [00:18, 898.74 examples/s]16374 examples [00:18, 912.93 examples/s]16467 examples [00:18, 917.26 examples/s]16559 examples [00:18, 893.93 examples/s]16649 examples [00:18, 894.55 examples/s]16739 examples [00:18, 886.45 examples/s]16829 examples [00:19, 888.80 examples/s]16919 examples [00:19, 890.05 examples/s]17011 examples [00:19, 898.60 examples/s]17101 examples [00:19, 883.24 examples/s]17190 examples [00:19, 859.45 examples/s]17278 examples [00:19, 863.94 examples/s]17365 examples [00:19, 849.81 examples/s]17452 examples [00:19, 854.20 examples/s]17541 examples [00:19, 864.33 examples/s]17629 examples [00:20, 867.92 examples/s]17716 examples [00:20, 856.04 examples/s]17803 examples [00:20, 859.03 examples/s]17891 examples [00:20, 864.36 examples/s]17979 examples [00:20, 866.17 examples/s]18071 examples [00:20, 881.02 examples/s]18162 examples [00:20, 879.78 examples/s]18251 examples [00:20, 878.99 examples/s]18339 examples [00:20, 865.48 examples/s]18426 examples [00:20, 859.33 examples/s]18515 examples [00:21, 868.20 examples/s]18602 examples [00:21, 867.59 examples/s]18692 examples [00:21, 875.41 examples/s]18781 examples [00:21, 878.58 examples/s]18869 examples [00:21, 872.21 examples/s]18958 examples [00:21, 875.44 examples/s]19048 examples [00:21, 880.43 examples/s]19137 examples [00:21, 879.42 examples/s]19227 examples [00:21, 882.97 examples/s]19316 examples [00:21, 879.18 examples/s]19408 examples [00:22, 889.00 examples/s]19497 examples [00:22, 886.82 examples/s]19591 examples [00:22, 898.80 examples/s]19681 examples [00:22, 883.89 examples/s]19770 examples [00:22, 874.37 examples/s]19860 examples [00:22, 879.29 examples/s]19950 examples [00:22, 885.09 examples/s]20039 examples [00:22, 835.10 examples/s]20128 examples [00:22, 850.32 examples/s]20217 examples [00:22, 860.27 examples/s]20308 examples [00:23, 872.18 examples/s]20397 examples [00:23, 877.34 examples/s]20485 examples [00:23, 876.31 examples/s]20573 examples [00:23, 875.59 examples/s]20663 examples [00:23, 880.02 examples/s]20752 examples [00:23, 882.91 examples/s]20841 examples [00:23, 882.90 examples/s]20930 examples [00:23, 856.13 examples/s]21019 examples [00:23, 864.18 examples/s]21111 examples [00:24, 877.95 examples/s]21203 examples [00:24, 888.37 examples/s]21293 examples [00:24, 889.90 examples/s]21383 examples [00:24, 892.48 examples/s]21473 examples [00:24, 888.23 examples/s]21566 examples [00:24, 899.97 examples/s]21658 examples [00:24, 904.97 examples/s]21749 examples [00:24, 899.52 examples/s]21840 examples [00:24, 877.42 examples/s]21928 examples [00:24, 874.42 examples/s]22016 examples [00:25, 867.74 examples/s]22104 examples [00:25, 869.69 examples/s]22193 examples [00:25, 874.36 examples/s]22285 examples [00:25, 886.11 examples/s]22376 examples [00:25, 892.91 examples/s]22466 examples [00:25, 892.20 examples/s]22560 examples [00:25, 905.87 examples/s]22651 examples [00:25, 904.69 examples/s]22742 examples [00:25, 905.42 examples/s]22833 examples [00:25, 900.66 examples/s]22926 examples [00:26, 907.60 examples/s]23017 examples [00:26, 905.62 examples/s]23108 examples [00:26, 906.42 examples/s]23199 examples [00:26, 895.68 examples/s]23289 examples [00:26, 894.04 examples/s]23381 examples [00:26, 901.28 examples/s]23472 examples [00:26, 900.81 examples/s]23563 examples [00:26, 888.55 examples/s]23652 examples [00:26, 886.09 examples/s]23743 examples [00:26, 890.06 examples/s]23833 examples [00:27, 892.89 examples/s]23923 examples [00:27, 894.74 examples/s]24014 examples [00:27, 898.96 examples/s]24104 examples [00:27, 896.29 examples/s]24194 examples [00:27, 895.19 examples/s]24286 examples [00:27, 900.56 examples/s]24378 examples [00:27, 904.20 examples/s]24471 examples [00:27, 911.64 examples/s]24564 examples [00:27, 916.17 examples/s]24656 examples [00:27, 912.44 examples/s]24749 examples [00:28, 916.06 examples/s]24843 examples [00:28, 921.10 examples/s]24938 examples [00:28, 927.17 examples/s]25031 examples [00:28, 915.87 examples/s]25125 examples [00:28, 920.78 examples/s]25218 examples [00:28, 900.47 examples/s]25309 examples [00:28, 903.25 examples/s]25404 examples [00:28, 913.80 examples/s]25497 examples [00:28, 917.00 examples/s]25589 examples [00:28, 917.50 examples/s]25681 examples [00:29, 910.83 examples/s]25773 examples [00:29, 913.21 examples/s]25865 examples [00:29, 902.68 examples/s]25956 examples [00:29, 884.65 examples/s]26045 examples [00:29, 884.63 examples/s]26134 examples [00:29, 883.64 examples/s]26223 examples [00:29, 882.12 examples/s]26313 examples [00:29, 884.81 examples/s]26402 examples [00:29, 878.12 examples/s]26491 examples [00:29, 880.99 examples/s]26580 examples [00:30, 876.31 examples/s]26669 examples [00:30, 877.76 examples/s]26757 examples [00:30, 855.45 examples/s]26843 examples [00:30, 841.87 examples/s]26929 examples [00:30, 845.79 examples/s]27014 examples [00:30, 847.04 examples/s]27099 examples [00:30, 847.28 examples/s]27184 examples [00:30, 836.98 examples/s]27268 examples [00:30, 826.05 examples/s]27351 examples [00:31, 827.01 examples/s]27441 examples [00:31, 845.39 examples/s]27535 examples [00:31, 870.52 examples/s]27628 examples [00:31, 886.04 examples/s]27717 examples [00:31, 874.22 examples/s]27805 examples [00:31, 863.60 examples/s]27892 examples [00:31, 845.69 examples/s]27977 examples [00:31, 839.72 examples/s]28064 examples [00:31, 847.24 examples/s]28149 examples [00:31, 846.88 examples/s]28235 examples [00:32, 848.74 examples/s]28322 examples [00:32, 851.68 examples/s]28415 examples [00:32, 872.42 examples/s]28510 examples [00:32, 893.87 examples/s]28601 examples [00:32, 896.43 examples/s]28692 examples [00:32, 898.33 examples/s]28782 examples [00:32, 897.96 examples/s]28872 examples [00:32, 898.16 examples/s]28963 examples [00:32, 898.96 examples/s]29053 examples [00:32, 886.88 examples/s]29142 examples [00:33, 883.50 examples/s]29231 examples [00:33, 884.27 examples/s]29321 examples [00:33, 887.02 examples/s]29414 examples [00:33, 897.49 examples/s]29504 examples [00:33, 895.85 examples/s]29595 examples [00:33, 898.90 examples/s]29685 examples [00:33, 883.00 examples/s]29774 examples [00:33, 870.49 examples/s]29862 examples [00:33, 854.87 examples/s]29948 examples [00:33, 854.34 examples/s]30034 examples [00:34, 806.13 examples/s]30123 examples [00:34, 828.69 examples/s]30209 examples [00:34, 835.90 examples/s]30296 examples [00:34, 844.93 examples/s]30381 examples [00:34, 837.53 examples/s]30468 examples [00:34, 846.85 examples/s]30558 examples [00:34, 861.29 examples/s]30648 examples [00:34, 869.83 examples/s]30736 examples [00:34, 868.60 examples/s]30825 examples [00:35, 874.14 examples/s]30915 examples [00:35, 881.53 examples/s]31007 examples [00:35, 891.54 examples/s]31097 examples [00:35, 887.61 examples/s]31186 examples [00:35, 886.73 examples/s]31275 examples [00:35, 884.89 examples/s]31367 examples [00:35, 894.15 examples/s]31458 examples [00:35, 897.31 examples/s]31550 examples [00:35, 901.72 examples/s]31642 examples [00:35, 904.69 examples/s]31733 examples [00:36, 900.62 examples/s]31826 examples [00:36, 907.79 examples/s]31920 examples [00:36, 916.29 examples/s]32012 examples [00:36, 909.76 examples/s]32106 examples [00:36, 915.96 examples/s]32198 examples [00:36, 914.38 examples/s]32292 examples [00:36, 921.80 examples/s]32387 examples [00:36, 928.89 examples/s]32480 examples [00:36, 918.04 examples/s]32572 examples [00:36, 915.23 examples/s]32664 examples [00:37, 899.94 examples/s]32755 examples [00:37, 895.76 examples/s]32849 examples [00:37, 907.50 examples/s]32944 examples [00:37, 919.75 examples/s]33037 examples [00:37, 916.79 examples/s]33129 examples [00:37, 908.27 examples/s]33220 examples [00:37, 908.28 examples/s]33312 examples [00:37, 908.96 examples/s]33403 examples [00:37, 903.43 examples/s]33494 examples [00:37, 893.50 examples/s]33584 examples [00:38, 893.23 examples/s]33675 examples [00:38, 897.48 examples/s]33766 examples [00:38, 898.58 examples/s]33859 examples [00:38, 906.06 examples/s]33950 examples [00:38, 888.25 examples/s]34039 examples [00:38, 885.93 examples/s]34130 examples [00:38, 890.13 examples/s]34222 examples [00:38, 897.95 examples/s]34314 examples [00:38, 903.78 examples/s]34405 examples [00:38, 893.29 examples/s]34495 examples [00:39, 894.63 examples/s]34587 examples [00:39, 900.37 examples/s]34678 examples [00:39, 900.39 examples/s]34769 examples [00:39, 891.33 examples/s]34859 examples [00:39, 883.18 examples/s]34948 examples [00:39, 884.88 examples/s]35040 examples [00:39, 893.10 examples/s]35130 examples [00:39, 893.74 examples/s]35220 examples [00:39, 890.30 examples/s]35310 examples [00:39, 884.26 examples/s]35399 examples [00:40, 874.64 examples/s]35491 examples [00:40, 885.90 examples/s]35583 examples [00:40, 893.58 examples/s]35673 examples [00:40, 883.04 examples/s]35762 examples [00:40, 868.79 examples/s]35849 examples [00:40, 855.14 examples/s]35939 examples [00:40, 866.32 examples/s]36026 examples [00:40, 856.87 examples/s]36116 examples [00:40, 867.53 examples/s]36209 examples [00:41, 884.85 examples/s]36298 examples [00:41, 879.84 examples/s]36390 examples [00:41, 890.57 examples/s]36483 examples [00:41, 899.51 examples/s]36576 examples [00:41, 906.03 examples/s]36667 examples [00:41, 897.80 examples/s]36757 examples [00:41, 896.51 examples/s]36850 examples [00:41, 905.76 examples/s]36941 examples [00:41, 905.71 examples/s]37034 examples [00:41, 912.67 examples/s]37126 examples [00:42, 911.90 examples/s]37218 examples [00:42, 895.71 examples/s]37308 examples [00:42, 895.49 examples/s]37398 examples [00:42, 895.93 examples/s]37488 examples [00:42, 880.62 examples/s]37577 examples [00:42, 870.17 examples/s]37665 examples [00:42, 868.99 examples/s]37753 examples [00:42, 870.50 examples/s]37841 examples [00:42, 870.01 examples/s]37935 examples [00:42, 888.28 examples/s]38024 examples [00:43, 887.95 examples/s]38113 examples [00:43, 882.22 examples/s]38206 examples [00:43, 895.07 examples/s]38299 examples [00:43, 898.10 examples/s]38392 examples [00:43, 905.63 examples/s]38483 examples [00:43, 894.60 examples/s]38573 examples [00:43, 874.92 examples/s]38661 examples [00:43, 869.47 examples/s]38751 examples [00:43, 878.16 examples/s]38839 examples [00:43, 871.69 examples/s]38927 examples [00:44, 854.50 examples/s]39015 examples [00:44, 861.54 examples/s]39105 examples [00:44, 870.41 examples/s]39193 examples [00:44, 847.18 examples/s]39278 examples [00:44, 847.32 examples/s]39365 examples [00:44, 851.65 examples/s]39453 examples [00:44, 857.75 examples/s]39544 examples [00:44, 869.74 examples/s]39632 examples [00:44, 832.26 examples/s]39721 examples [00:45, 847.41 examples/s]39807 examples [00:45, 849.18 examples/s]39896 examples [00:45, 860.65 examples/s]39984 examples [00:45, 863.46 examples/s]40071 examples [00:45, 786.08 examples/s]40157 examples [00:45, 805.56 examples/s]40242 examples [00:45, 815.71 examples/s]40328 examples [00:45, 827.19 examples/s]40413 examples [00:45, 808.89 examples/s]40500 examples [00:45, 826.10 examples/s]40585 examples [00:46, 831.95 examples/s]40669 examples [00:46, 801.19 examples/s]40755 examples [00:46, 816.03 examples/s]40838 examples [00:46, 818.36 examples/s]40926 examples [00:46, 834.16 examples/s]41019 examples [00:46, 860.43 examples/s]41108 examples [00:46, 866.71 examples/s]41195 examples [00:46, 866.89 examples/s]41284 examples [00:46, 871.52 examples/s]41374 examples [00:46, 878.72 examples/s]41464 examples [00:47, 884.70 examples/s]41553 examples [00:47, 882.69 examples/s]41646 examples [00:47, 894.19 examples/s]41736 examples [00:47, 891.62 examples/s]41826 examples [00:47, 882.75 examples/s]41915 examples [00:47, 881.23 examples/s]42004 examples [00:47, 867.98 examples/s]42092 examples [00:47, 870.61 examples/s]42180 examples [00:47, 870.79 examples/s]42270 examples [00:47, 877.38 examples/s]42360 examples [00:48, 880.91 examples/s]42449 examples [00:48, 881.31 examples/s]42538 examples [00:48, 881.24 examples/s]42627 examples [00:48, 858.86 examples/s]42720 examples [00:48, 876.89 examples/s]42808 examples [00:48, 873.73 examples/s]42896 examples [00:48, 869.31 examples/s]42988 examples [00:48, 882.20 examples/s]43079 examples [00:48, 889.83 examples/s]43169 examples [00:49, 878.73 examples/s]43261 examples [00:49, 889.91 examples/s]43353 examples [00:49, 896.22 examples/s]43443 examples [00:49, 890.08 examples/s]43533 examples [00:49, 889.44 examples/s]43624 examples [00:49, 894.05 examples/s]43716 examples [00:49, 900.59 examples/s]43807 examples [00:49, 894.36 examples/s]43898 examples [00:49, 896.62 examples/s]43988 examples [00:49, 891.59 examples/s]44078 examples [00:50, 864.86 examples/s]44165 examples [00:50, 862.49 examples/s]44254 examples [00:50, 870.35 examples/s]44345 examples [00:50, 880.12 examples/s]44438 examples [00:50, 892.18 examples/s]44528 examples [00:50, 878.49 examples/s]44617 examples [00:50, 878.04 examples/s]44705 examples [00:50, 878.55 examples/s]44798 examples [00:50, 892.89 examples/s]44889 examples [00:50, 896.79 examples/s]44979 examples [00:51, 891.21 examples/s]45072 examples [00:51, 901.18 examples/s]45163 examples [00:51, 898.50 examples/s]45253 examples [00:51, 898.77 examples/s]45345 examples [00:51, 903.05 examples/s]45436 examples [00:51, 902.67 examples/s]45529 examples [00:51, 908.97 examples/s]45620 examples [00:51, 890.07 examples/s]45712 examples [00:51, 898.51 examples/s]45802 examples [00:51, 883.20 examples/s]45895 examples [00:52, 894.60 examples/s]45985 examples [00:52, 880.38 examples/s]46077 examples [00:52, 889.97 examples/s]46169 examples [00:52, 897.54 examples/s]46263 examples [00:52, 907.91 examples/s]46354 examples [00:52, 905.28 examples/s]46446 examples [00:52, 909.11 examples/s]46537 examples [00:52, 908.61 examples/s]46628 examples [00:52, 908.79 examples/s]46719 examples [00:52, 901.33 examples/s]46813 examples [00:53, 911.70 examples/s]46905 examples [00:53, 905.74 examples/s]46996 examples [00:53, 900.70 examples/s]47087 examples [00:53, 902.64 examples/s]47181 examples [00:53, 912.04 examples/s]47273 examples [00:53, 914.03 examples/s]47365 examples [00:53, 909.40 examples/s]47456 examples [00:53, 902.46 examples/s]47548 examples [00:53, 905.11 examples/s]47642 examples [00:54, 912.99 examples/s]47735 examples [00:54, 916.11 examples/s]47827 examples [00:54, 916.25 examples/s]47919 examples [00:54, 910.55 examples/s]48011 examples [00:54, 901.14 examples/s]48104 examples [00:54, 907.10 examples/s]48195 examples [00:54, 900.71 examples/s]48286 examples [00:54, 877.85 examples/s]48376 examples [00:54, 883.19 examples/s]48468 examples [00:54, 891.28 examples/s]48558 examples [00:55, 891.77 examples/s]48650 examples [00:55, 897.45 examples/s]48740 examples [00:55, 887.94 examples/s]48829 examples [00:55, 884.00 examples/s]48918 examples [00:55, 879.35 examples/s]49009 examples [00:55, 885.73 examples/s]49098 examples [00:55, 881.98 examples/s]49187 examples [00:55, 862.27 examples/s]49279 examples [00:55, 876.77 examples/s]49368 examples [00:55, 878.11 examples/s]49456 examples [00:56, 866.38 examples/s]49545 examples [00:56, 872.32 examples/s]49633 examples [00:56, 851.02 examples/s]49724 examples [00:56, 865.76 examples/s]49815 examples [00:56, 875.80 examples/s]49909 examples [00:56, 892.03 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5449/50000 [00:00<00:00, 54486.15 examples/s] 34%|███▍      | 17201/50000 [00:00<00:00, 64933.11 examples/s] 58%|█████▊    | 28834/50000 [00:00<00:00, 74854.17 examples/s] 81%|████████  | 40485/50000 [00:00<00:00, 83847.17 examples/s]                                                               0 examples [00:00, ? examples/s]67 examples [00:00, 669.94 examples/s]149 examples [00:00, 707.12 examples/s]237 examples [00:00, 750.30 examples/s]323 examples [00:00, 778.51 examples/s]418 examples [00:00, 821.54 examples/s]508 examples [00:00, 843.09 examples/s]600 examples [00:00, 863.18 examples/s]692 examples [00:00, 878.36 examples/s]780 examples [00:00, 877.94 examples/s]866 examples [00:01, 841.57 examples/s]949 examples [00:01, 813.88 examples/s]1030 examples [00:01, 805.73 examples/s]1123 examples [00:01, 838.96 examples/s]1212 examples [00:01, 852.20 examples/s]1302 examples [00:01, 865.96 examples/s]1394 examples [00:01, 880.71 examples/s]1486 examples [00:01, 891.12 examples/s]1577 examples [00:01, 895.04 examples/s]1668 examples [00:01, 897.11 examples/s]1758 examples [00:02, 892.71 examples/s]1848 examples [00:02, 883.73 examples/s]1937 examples [00:02, 879.44 examples/s]2027 examples [00:02, 884.55 examples/s]2116 examples [00:02, 882.76 examples/s]2205 examples [00:02, 884.41 examples/s]2294 examples [00:02, 879.04 examples/s]2383 examples [00:02, 880.78 examples/s]2472 examples [00:02, 867.85 examples/s]2562 examples [00:02, 876.79 examples/s]2651 examples [00:03, 878.34 examples/s]2739 examples [00:03, 861.39 examples/s]2827 examples [00:03, 863.78 examples/s]2915 examples [00:03, 868.38 examples/s]3004 examples [00:03, 872.46 examples/s]3094 examples [00:03, 879.97 examples/s]3187 examples [00:03, 894.14 examples/s]3280 examples [00:03, 904.16 examples/s]3373 examples [00:03, 910.68 examples/s]3466 examples [00:03, 913.87 examples/s]3558 examples [00:04, 897.90 examples/s]3651 examples [00:04, 906.24 examples/s]3745 examples [00:04, 914.12 examples/s]3840 examples [00:04, 924.49 examples/s]3936 examples [00:04, 933.48 examples/s]4030 examples [00:04, 930.53 examples/s]4124 examples [00:04, 914.00 examples/s]4216 examples [00:04, 914.79 examples/s]4308 examples [00:04, 896.23 examples/s]4398 examples [00:04, 891.28 examples/s]4488 examples [00:05, 886.52 examples/s]4578 examples [00:05, 889.84 examples/s]4673 examples [00:05, 904.93 examples/s]4766 examples [00:05, 911.92 examples/s]4862 examples [00:05, 925.75 examples/s]4956 examples [00:05, 928.08 examples/s]5053 examples [00:05, 937.99 examples/s]5150 examples [00:05, 945.29 examples/s]5246 examples [00:05, 948.30 examples/s]5344 examples [00:05, 955.02 examples/s]5440 examples [00:06, 954.82 examples/s]5536 examples [00:06, 938.93 examples/s]5634 examples [00:06, 948.33 examples/s]5732 examples [00:06, 953.78 examples/s]5828 examples [00:06, 946.67 examples/s]5923 examples [00:06, 937.34 examples/s]6017 examples [00:06, 906.28 examples/s]6113 examples [00:06, 920.23 examples/s]6206 examples [00:06, 916.00 examples/s]6298 examples [00:07, 912.39 examples/s]6390 examples [00:07, 899.99 examples/s]6481 examples [00:07, 898.91 examples/s]6571 examples [00:07, 888.34 examples/s]6662 examples [00:07, 892.84 examples/s]6754 examples [00:07, 899.38 examples/s]6844 examples [00:07, 897.45 examples/s]6934 examples [00:07, 884.35 examples/s]7023 examples [00:07, 883.19 examples/s]7113 examples [00:07, 886.43 examples/s]7204 examples [00:08, 890.00 examples/s]7296 examples [00:08, 898.70 examples/s]7386 examples [00:08, 892.89 examples/s]7476 examples [00:08, 887.01 examples/s]7565 examples [00:08, 879.98 examples/s]7654 examples [00:08, 879.94 examples/s]7743 examples [00:08, 875.09 examples/s]7833 examples [00:08, 881.69 examples/s]7922 examples [00:08, 880.70 examples/s]8011 examples [00:08, 883.26 examples/s]8102 examples [00:09, 889.10 examples/s]8191 examples [00:09, 886.78 examples/s]8280 examples [00:09, 873.14 examples/s]8374 examples [00:09, 890.73 examples/s]8467 examples [00:09, 901.56 examples/s]8561 examples [00:09, 910.76 examples/s]8653 examples [00:09, 904.81 examples/s]8746 examples [00:09, 910.11 examples/s]8838 examples [00:09, 907.39 examples/s]8929 examples [00:09, 906.64 examples/s]9021 examples [00:10, 908.59 examples/s]9112 examples [00:10, 902.68 examples/s]9203 examples [00:10, 901.61 examples/s]9295 examples [00:10, 905.69 examples/s]9386 examples [00:10, 904.27 examples/s]9477 examples [00:10, 903.18 examples/s]9568 examples [00:10, 896.39 examples/s]9659 examples [00:10, 898.62 examples/s]9751 examples [00:10, 902.26 examples/s]9842 examples [00:10, 903.80 examples/s]9935 examples [00:11, 910.69 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYCRB5L/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYCRB5L/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efce2902730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efce2902730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efce2902730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efcc04b4ac8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efc6cb1d390>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efce2902730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efce2902730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efce2902730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efc6cb1d240>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efcc04e10f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7efc5a345378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7efc5a345378> 

  function with postional parmater data_info <function split_train_valid at 0x7efc5a345378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7efc5a345488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7efc5a345488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7efc5a345488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=1466c48a64ee129852c1b7e53c7703c963b9a71866f19871a95dd70d2311173e
  Stored in directory: /tmp/pip-ephem-wheel-cache-9ov716ib/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip: 0.00B [01:12, ?B/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json [Errno 104] Connection reset by peer

  




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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 298, in compute
    out_tmp = preprocessor_func(data_info=self.data_info, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 241, in create_tabular_dataset
    TEXT.build_vocab(tabular_train, vectors=pretrained_emb)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/data/field.py", line 309, in build_vocab
    self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 101, in __init__
    self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 182, in load_vectors
    vectors[idx] = pretrained_aliases[vector](**kwargs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 484, in __init__
    super(GloVe, self).__init__(name, url=url, **kwargs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 323, in __init__
    self.cache(name, cache, url=url, max_vectors=max_vectors)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/vocab.py", line 358, in cache
    urlretrieve(url, dest, reporthook=reporthook(t))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 248, in urlretrieve
    with contextlib.closing(urlopen(url, data)) as fp:
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 223, in urlopen
    return opener.open(url, data, timeout)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 532, in open
    response = meth(req, response)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 642, in http_response
    'http', request, response, code, msg, hdrs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 564, in error
    result = self._call_chain(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 504, in _call_chain
    result = func(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 756, in http_error_302
    return self.parent.open(new, timeout=req.timeout)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 532, in open
    response = meth(req, response)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 642, in http_response
    'http', request, response, code, msg, hdrs)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 564, in error
    result = self._call_chain(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 504, in _call_chain
    result = func(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 756, in http_error_302
    return self.parent.open(new, timeout=req.timeout)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 526, in open
    response = self._open(req, data)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 544, in _open
    '_open', req)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 504, in _call_chain
    result = func(*args)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 1377, in http_open
    return self.do_open(http.client.HTTPConnection, req)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/urllib/request.py", line 1352, in do_open
    r = h.getresponse()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/http/client.py", line 1364, in getresponse
    response.begin()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/http/client.py", line 307, in begin
    version, status, reason = self._read_status()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/http/client.py", line 268, in _read_status
    line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/socket.py", line 586, in readinto
    return self._sock.recv_into(b)
ConnectionResetError: [Errno 104] Connection reset by peer
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 14.39 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 10.66 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 10.66 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.58 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.58 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.23 file/s]2020-07-16 18:13:25.006197: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-16 18:13:25.010616: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294675000 Hz
2020-07-16 18:13:25.010803: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5612e098d840 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-16 18:13:25.010820: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<02:28, 66463.57it/s]9920512it [00:00, 34515585.79it/s]                        
0it [00:00, ?it/s]32768it [00:00, 413895.46it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158202.02it/s]1654784it [00:00, 11188254.35it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202680.04it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
