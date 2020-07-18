
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f97ccf69a60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f97ccf69a60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f9837bd4510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f9837bd4510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f9856e23ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f9856e23ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f97e4f08950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f97e4f08950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f97e4f08950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3080192/9912422 [00:00<00:00, 30798865.30it/s]9920512it [00:00, 33784481.55it/s]                             
0it [00:00, ?it/s]32768it [00:00, 469063.72it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475208.85it/s]1654784it [00:00, 11784155.32it/s]                         
0it [00:00, ?it/s]8192it [00:00, 259728.47it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f97e2fe2208>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f97e3477908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f97e4f08598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f97e4f08598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f97e4f08598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:34,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:34,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:33,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:33,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:32,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:05,  2.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:05,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:04,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:04,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:03,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:03,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:03,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:02,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:02,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:43,  3.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:43,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:42,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:42,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:29,  4.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:29,  4.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:28,  4.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:28,  4.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:20,  6.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:20,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:20,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:20,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:19,  6.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:14,  8.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:14,  8.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:14,  8.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:13,  8.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:13,  8.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  8.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:13,  8.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:13,  8.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 12.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 12.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 12.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 12.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 12.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 12.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 16.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 20.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 26.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 32.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 38.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 38.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 38.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 38.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 44.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 44.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 44.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 44.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 44.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 44.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 44.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 44.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 48.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 48.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 48.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 48.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 48.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 48.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 48.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 52.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 56.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 59.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 61.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 70.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 70.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 70.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 70.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 70.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 70.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 70.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 69.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.91s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.91s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.91s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.91s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 69.25 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.90 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ url]
0 examples [00:00, ? examples/s]2020-07-18 00:09:45.605316: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-18 00:09:45.624367: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-07-18 00:09:45.624582: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562015366930 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-18 00:09:45.624605: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
32 examples [00:00, 317.79 examples/s]124 examples [00:00, 395.16 examples/s]212 examples [00:00, 473.08 examples/s]304 examples [00:00, 553.50 examples/s]398 examples [00:00, 630.10 examples/s]494 examples [00:00, 701.90 examples/s]592 examples [00:00, 766.76 examples/s]689 examples [00:00, 816.94 examples/s]778 examples [00:00, 837.49 examples/s]867 examples [00:01, 829.62 examples/s]959 examples [00:01, 853.93 examples/s]1057 examples [00:01, 887.25 examples/s]1148 examples [00:01, 871.63 examples/s]1238 examples [00:01, 878.37 examples/s]1338 examples [00:01, 910.13 examples/s]1436 examples [00:01, 929.88 examples/s]1535 examples [00:01, 946.78 examples/s]1631 examples [00:01, 946.76 examples/s]1727 examples [00:01, 926.51 examples/s]1821 examples [00:02, 906.61 examples/s]1913 examples [00:02, 866.42 examples/s]2001 examples [00:02, 859.54 examples/s]2096 examples [00:02, 884.55 examples/s]2185 examples [00:02, 856.37 examples/s]2278 examples [00:02, 875.19 examples/s]2372 examples [00:02, 891.62 examples/s]2463 examples [00:02, 896.36 examples/s]2555 examples [00:02, 901.46 examples/s]2651 examples [00:02, 916.35 examples/s]2743 examples [00:03, 866.87 examples/s]2840 examples [00:03, 894.46 examples/s]2939 examples [00:03, 920.03 examples/s]3035 examples [00:03, 930.24 examples/s]3132 examples [00:03, 939.79 examples/s]3229 examples [00:03, 946.17 examples/s]3327 examples [00:03, 954.56 examples/s]3424 examples [00:03, 958.83 examples/s]3521 examples [00:03, 961.97 examples/s]3621 examples [00:04, 972.33 examples/s]3719 examples [00:04, 963.63 examples/s]3819 examples [00:04, 973.43 examples/s]3917 examples [00:04, 960.11 examples/s]4014 examples [00:04, 962.13 examples/s]4113 examples [00:04, 967.70 examples/s]4213 examples [00:04, 976.37 examples/s]4311 examples [00:04, 973.15 examples/s]4413 examples [00:04, 984.97 examples/s]4512 examples [00:04, 937.52 examples/s]4607 examples [00:05, 918.56 examples/s]4700 examples [00:05, 921.02 examples/s]4794 examples [00:05, 924.05 examples/s]4888 examples [00:05, 927.86 examples/s]4983 examples [00:05, 933.07 examples/s]5077 examples [00:05, 934.30 examples/s]5178 examples [00:05, 954.72 examples/s]5274 examples [00:05, 941.17 examples/s]5369 examples [00:05, 911.39 examples/s]5461 examples [00:05, 912.25 examples/s]5555 examples [00:06, 919.73 examples/s]5651 examples [00:06, 930.56 examples/s]5752 examples [00:06, 952.55 examples/s]5848 examples [00:06, 954.69 examples/s]5944 examples [00:06, 878.36 examples/s]6039 examples [00:06, 891.38 examples/s]6132 examples [00:06, 900.87 examples/s]6224 examples [00:06, 904.99 examples/s]6323 examples [00:06, 927.25 examples/s]6419 examples [00:06, 934.15 examples/s]6515 examples [00:07, 940.87 examples/s]6617 examples [00:07, 961.35 examples/s]6714 examples [00:07, 956.39 examples/s]6810 examples [00:07, 924.89 examples/s]6904 examples [00:07, 926.69 examples/s]7003 examples [00:07, 944.31 examples/s]7098 examples [00:07, 941.62 examples/s]7193 examples [00:07, 920.91 examples/s]7286 examples [00:07, 913.82 examples/s]7378 examples [00:08, 873.37 examples/s]7474 examples [00:08, 896.42 examples/s]7570 examples [00:08, 913.63 examples/s]7666 examples [00:08, 926.37 examples/s]7762 examples [00:08, 934.73 examples/s]7861 examples [00:08, 949.92 examples/s]7961 examples [00:08, 963.62 examples/s]8060 examples [00:08, 970.85 examples/s]8159 examples [00:08, 975.77 examples/s]8257 examples [00:08, 968.97 examples/s]8356 examples [00:09, 974.39 examples/s]8455 examples [00:09, 978.32 examples/s]8554 examples [00:09, 979.77 examples/s]8654 examples [00:09, 984.01 examples/s]8753 examples [00:09, 980.83 examples/s]8852 examples [00:09, 969.90 examples/s]8950 examples [00:09, 959.55 examples/s]9047 examples [00:09, 951.78 examples/s]9143 examples [00:09, 920.29 examples/s]9236 examples [00:09, 913.33 examples/s]9330 examples [00:10, 920.61 examples/s]9423 examples [00:10, 916.86 examples/s]9515 examples [00:10, 911.00 examples/s]9607 examples [00:10, 909.03 examples/s]9698 examples [00:10, 862.60 examples/s]9792 examples [00:10, 882.31 examples/s]9886 examples [00:10, 898.02 examples/s]9980 examples [00:10, 909.32 examples/s]10072 examples [00:10, 882.30 examples/s]10166 examples [00:11, 898.36 examples/s]10264 examples [00:11, 920.13 examples/s]10363 examples [00:11, 938.80 examples/s]10462 examples [00:11, 948.59 examples/s]10560 examples [00:11, 957.61 examples/s]10656 examples [00:11, 933.03 examples/s]10750 examples [00:11, 914.78 examples/s]10842 examples [00:11, 861.70 examples/s]10934 examples [00:11, 876.97 examples/s]11026 examples [00:11, 889.38 examples/s]11116 examples [00:12, 884.98 examples/s]11214 examples [00:12, 909.28 examples/s]11313 examples [00:12, 930.69 examples/s]11411 examples [00:12, 943.60 examples/s]11508 examples [00:12, 950.77 examples/s]11604 examples [00:12, 943.47 examples/s]11699 examples [00:12, 943.40 examples/s]11795 examples [00:12, 945.72 examples/s]11890 examples [00:12, 932.18 examples/s]11984 examples [00:12, 907.29 examples/s]12082 examples [00:13, 926.99 examples/s]12181 examples [00:13, 942.35 examples/s]12278 examples [00:13, 948.60 examples/s]12375 examples [00:13, 954.40 examples/s]12471 examples [00:13, 937.46 examples/s]12565 examples [00:13, 910.19 examples/s]12660 examples [00:13, 919.63 examples/s]12758 examples [00:13, 935.23 examples/s]12854 examples [00:13, 942.01 examples/s]12949 examples [00:14, 933.14 examples/s]13043 examples [00:14, 907.78 examples/s]13137 examples [00:14, 915.52 examples/s]13232 examples [00:14, 924.03 examples/s]13330 examples [00:14, 939.20 examples/s]13425 examples [00:14, 942.02 examples/s]13520 examples [00:14, 936.83 examples/s]13614 examples [00:14, 920.91 examples/s]13709 examples [00:14, 927.38 examples/s]13805 examples [00:14, 935.48 examples/s]13900 examples [00:15, 938.63 examples/s]13995 examples [00:15, 941.79 examples/s]14090 examples [00:15, 936.00 examples/s]14184 examples [00:15, 926.89 examples/s]14277 examples [00:15, 862.24 examples/s]14365 examples [00:15, 865.97 examples/s]14453 examples [00:15, 868.38 examples/s]14541 examples [00:15, 870.64 examples/s]14640 examples [00:15, 898.81 examples/s]14737 examples [00:15, 916.83 examples/s]14833 examples [00:16, 929.13 examples/s]14927 examples [00:16, 922.91 examples/s]15020 examples [00:16, 918.46 examples/s]15119 examples [00:16, 938.24 examples/s]15220 examples [00:16, 958.07 examples/s]15318 examples [00:16, 963.29 examples/s]15415 examples [00:16, 960.22 examples/s]15512 examples [00:16, 911.87 examples/s]15604 examples [00:16, 908.60 examples/s]15696 examples [00:17, 897.08 examples/s]15792 examples [00:17, 914.93 examples/s]15890 examples [00:17, 929.34 examples/s]15989 examples [00:17, 945.32 examples/s]16091 examples [00:17, 964.22 examples/s]16192 examples [00:17, 973.75 examples/s]16290 examples [00:17, 972.30 examples/s]16388 examples [00:17, 952.92 examples/s]16484 examples [00:17, 930.60 examples/s]16578 examples [00:17, 920.50 examples/s]16671 examples [00:18, 913.09 examples/s]16763 examples [00:18, 903.52 examples/s]16854 examples [00:18, 904.17 examples/s]16945 examples [00:18, 874.38 examples/s]17037 examples [00:18, 885.31 examples/s]17126 examples [00:18, 865.82 examples/s]17215 examples [00:18, 870.76 examples/s]17308 examples [00:18, 886.26 examples/s]17401 examples [00:18, 896.78 examples/s]17495 examples [00:18, 905.99 examples/s]17586 examples [00:19, 906.37 examples/s]17679 examples [00:19, 908.77 examples/s]17778 examples [00:19, 931.04 examples/s]17876 examples [00:19, 944.88 examples/s]17971 examples [00:19, 934.40 examples/s]18068 examples [00:19, 944.37 examples/s]18163 examples [00:19, 943.46 examples/s]18260 examples [00:19, 950.77 examples/s]18357 examples [00:19, 955.13 examples/s]18457 examples [00:19, 965.75 examples/s]18554 examples [00:20, 956.86 examples/s]18650 examples [00:20, 940.67 examples/s]18745 examples [00:20, 924.99 examples/s]18838 examples [00:20, 906.47 examples/s]18929 examples [00:20, 903.42 examples/s]19030 examples [00:20, 930.67 examples/s]19124 examples [00:20, 925.79 examples/s]19217 examples [00:20, 926.52 examples/s]19316 examples [00:20, 942.84 examples/s]19415 examples [00:21, 956.41 examples/s]19513 examples [00:21, 961.95 examples/s]19610 examples [00:21, 954.86 examples/s]19707 examples [00:21, 959.15 examples/s]19803 examples [00:21, 958.03 examples/s]19901 examples [00:21, 958.12 examples/s]19997 examples [00:21, 955.19 examples/s]20093 examples [00:21, 917.47 examples/s]20191 examples [00:21, 935.25 examples/s]20288 examples [00:21, 945.33 examples/s]20386 examples [00:22, 952.69 examples/s]20482 examples [00:22, 883.94 examples/s]20572 examples [00:22, 872.47 examples/s]20669 examples [00:22, 898.86 examples/s]20767 examples [00:22, 920.73 examples/s]20860 examples [00:22, 906.72 examples/s]20953 examples [00:22, 912.83 examples/s]21046 examples [00:22, 917.03 examples/s]21138 examples [00:22, 913.85 examples/s]21230 examples [00:22, 890.22 examples/s]21320 examples [00:23, 877.00 examples/s]21415 examples [00:23, 896.34 examples/s]21515 examples [00:23, 923.04 examples/s]21611 examples [00:23, 932.71 examples/s]21705 examples [00:23, 927.98 examples/s]21799 examples [00:23, 921.28 examples/s]21892 examples [00:23, 915.88 examples/s]21991 examples [00:23, 935.97 examples/s]22090 examples [00:23, 951.47 examples/s]22189 examples [00:23, 961.70 examples/s]22289 examples [00:24, 971.83 examples/s]22387 examples [00:24, 969.73 examples/s]22486 examples [00:24, 975.18 examples/s]22586 examples [00:24, 979.60 examples/s]22685 examples [00:24, 969.00 examples/s]22782 examples [00:24, 934.63 examples/s]22879 examples [00:24, 942.74 examples/s]22978 examples [00:24, 954.98 examples/s]23077 examples [00:24, 962.94 examples/s]23177 examples [00:25, 971.30 examples/s]23277 examples [00:25, 976.93 examples/s]23375 examples [00:25, 974.99 examples/s]23473 examples [00:25, 910.50 examples/s]23569 examples [00:25, 923.62 examples/s]23663 examples [00:25, 922.87 examples/s]23757 examples [00:25, 925.13 examples/s]23853 examples [00:25, 933.53 examples/s]23954 examples [00:25, 953.08 examples/s]24050 examples [00:25, 923.58 examples/s]24149 examples [00:26, 942.42 examples/s]24244 examples [00:26, 922.50 examples/s]24337 examples [00:26, 923.95 examples/s]24431 examples [00:26, 928.64 examples/s]24529 examples [00:26, 942.58 examples/s]24626 examples [00:26, 947.64 examples/s]24724 examples [00:26, 954.78 examples/s]24821 examples [00:26, 957.83 examples/s]24921 examples [00:26, 969.28 examples/s]25020 examples [00:26, 973.16 examples/s]25119 examples [00:27, 977.38 examples/s]25217 examples [00:27, 961.21 examples/s]25314 examples [00:27, 939.16 examples/s]25409 examples [00:27, 918.21 examples/s]25502 examples [00:27, 878.87 examples/s]25591 examples [00:27, 861.36 examples/s]25678 examples [00:27, 854.95 examples/s]25764 examples [00:27, 844.14 examples/s]25855 examples [00:27, 861.01 examples/s]25953 examples [00:28, 892.98 examples/s]26044 examples [00:28, 896.23 examples/s]26134 examples [00:28, 896.20 examples/s]26226 examples [00:28, 902.29 examples/s]26320 examples [00:28, 910.98 examples/s]26413 examples [00:28, 915.82 examples/s]26505 examples [00:28, 912.52 examples/s]26597 examples [00:28, 908.38 examples/s]26688 examples [00:28, 905.67 examples/s]26779 examples [00:28, 904.89 examples/s]26871 examples [00:29, 907.39 examples/s]26965 examples [00:29, 916.27 examples/s]27057 examples [00:29, 917.22 examples/s]27152 examples [00:29, 924.14 examples/s]27245 examples [00:29, 917.84 examples/s]27337 examples [00:29, 916.31 examples/s]27431 examples [00:29, 922.94 examples/s]27527 examples [00:29, 932.73 examples/s]27625 examples [00:29, 944.95 examples/s]27722 examples [00:29, 949.50 examples/s]27822 examples [00:30, 961.50 examples/s]27922 examples [00:30, 970.11 examples/s]28020 examples [00:30, 961.10 examples/s]28117 examples [00:30, 912.49 examples/s]28209 examples [00:30, 903.75 examples/s]28300 examples [00:30, 903.31 examples/s]28391 examples [00:30, 903.82 examples/s]28482 examples [00:30, 890.11 examples/s]28572 examples [00:30, 879.09 examples/s]28661 examples [00:30, 881.43 examples/s]28750 examples [00:31, 878.04 examples/s]28839 examples [00:31, 880.72 examples/s]28928 examples [00:31, 865.54 examples/s]29019 examples [00:31, 876.04 examples/s]29107 examples [00:31, 865.99 examples/s]29196 examples [00:31, 871.76 examples/s]29286 examples [00:31, 880.02 examples/s]29376 examples [00:31, 883.74 examples/s]29465 examples [00:31, 885.25 examples/s]29556 examples [00:32, 890.93 examples/s]29646 examples [00:32, 885.00 examples/s]29735 examples [00:32, 874.45 examples/s]29823 examples [00:32, 875.21 examples/s]29915 examples [00:32, 887.78 examples/s]30004 examples [00:32, 854.76 examples/s]30094 examples [00:32, 866.09 examples/s]30189 examples [00:32, 888.88 examples/s]30279 examples [00:32, 891.79 examples/s]30369 examples [00:32, 862.27 examples/s]30456 examples [00:33, 861.91 examples/s]30555 examples [00:33, 895.45 examples/s]30653 examples [00:33, 917.04 examples/s]30746 examples [00:33, 919.91 examples/s]30842 examples [00:33, 931.33 examples/s]30936 examples [00:33, 932.50 examples/s]31033 examples [00:33, 942.06 examples/s]31128 examples [00:33, 939.65 examples/s]31224 examples [00:33, 944.10 examples/s]31321 examples [00:33, 951.38 examples/s]31417 examples [00:34, 950.27 examples/s]31513 examples [00:34, 945.08 examples/s]31608 examples [00:34, 938.51 examples/s]31702 examples [00:34, 882.45 examples/s]31793 examples [00:34, 889.41 examples/s]31883 examples [00:34, 886.88 examples/s]31973 examples [00:34, 887.55 examples/s]32063 examples [00:34, 884.73 examples/s]32163 examples [00:34, 914.90 examples/s]32264 examples [00:34, 941.22 examples/s]32363 examples [00:35, 955.21 examples/s]32459 examples [00:35, 945.28 examples/s]32554 examples [00:35, 941.62 examples/s]32649 examples [00:35, 897.99 examples/s]32740 examples [00:35, 873.20 examples/s]32831 examples [00:35, 880.96 examples/s]32920 examples [00:35, 880.91 examples/s]33009 examples [00:35, 874.55 examples/s]33105 examples [00:35, 898.06 examples/s]33205 examples [00:36, 924.64 examples/s]33302 examples [00:36, 935.84 examples/s]33402 examples [00:36, 953.37 examples/s]33501 examples [00:36, 962.07 examples/s]33598 examples [00:36, 957.87 examples/s]33696 examples [00:36, 962.55 examples/s]33794 examples [00:36, 965.79 examples/s]33891 examples [00:36, 960.80 examples/s]33988 examples [00:36, 960.73 examples/s]34088 examples [00:36, 969.75 examples/s]34186 examples [00:37, 966.36 examples/s]34286 examples [00:37, 974.66 examples/s]34384 examples [00:37, 967.50 examples/s]34482 examples [00:37, 968.96 examples/s]34580 examples [00:37, 971.52 examples/s]34678 examples [00:37, 951.56 examples/s]34774 examples [00:37, 918.73 examples/s]34867 examples [00:37, 878.15 examples/s]34956 examples [00:37, 853.17 examples/s]35042 examples [00:38, 823.78 examples/s]35132 examples [00:38, 842.95 examples/s]35228 examples [00:38, 874.03 examples/s]35326 examples [00:38, 901.98 examples/s]35418 examples [00:38, 906.33 examples/s]35514 examples [00:38, 920.42 examples/s]35607 examples [00:38, 906.60 examples/s]35702 examples [00:38, 918.69 examples/s]35798 examples [00:38, 930.18 examples/s]35894 examples [00:38, 938.21 examples/s]35993 examples [00:39, 950.84 examples/s]36094 examples [00:39, 965.49 examples/s]36192 examples [00:39, 969.72 examples/s]36292 examples [00:39, 976.41 examples/s]36390 examples [00:39, 965.45 examples/s]36487 examples [00:39, 958.81 examples/s]36583 examples [00:39, 935.50 examples/s]36677 examples [00:39, 932.53 examples/s]36771 examples [00:39, 926.03 examples/s]36868 examples [00:39, 936.57 examples/s]36965 examples [00:40, 945.82 examples/s]37062 examples [00:40, 952.73 examples/s]37158 examples [00:40, 932.54 examples/s]37252 examples [00:40, 877.83 examples/s]37341 examples [00:40, 863.91 examples/s]37428 examples [00:40, 812.28 examples/s]37512 examples [00:40, 818.58 examples/s]37595 examples [00:40, 821.40 examples/s]37687 examples [00:40, 848.27 examples/s]37781 examples [00:41, 871.71 examples/s]37875 examples [00:41, 889.78 examples/s]37969 examples [00:41, 902.56 examples/s]38061 examples [00:41, 905.94 examples/s]38153 examples [00:41, 909.82 examples/s]38245 examples [00:41, 900.27 examples/s]38336 examples [00:41, 895.32 examples/s]38426 examples [00:41, 889.58 examples/s]38518 examples [00:41, 896.77 examples/s]38608 examples [00:41, 894.46 examples/s]38699 examples [00:42, 898.63 examples/s]38796 examples [00:42, 917.92 examples/s]38895 examples [00:42, 935.10 examples/s]38989 examples [00:42, 923.71 examples/s]39082 examples [00:42, 901.79 examples/s]39173 examples [00:42, 868.49 examples/s]39267 examples [00:42, 887.70 examples/s]39363 examples [00:42, 907.83 examples/s]39457 examples [00:42, 915.24 examples/s]39552 examples [00:42, 924.26 examples/s]39645 examples [00:43, 896.45 examples/s]39739 examples [00:43, 908.65 examples/s]39836 examples [00:43, 925.47 examples/s]39933 examples [00:43, 938.23 examples/s]40028 examples [00:43, 891.07 examples/s]40118 examples [00:43, 891.44 examples/s]40215 examples [00:43, 913.25 examples/s]40307 examples [00:43, 912.91 examples/s]40400 examples [00:43, 916.92 examples/s]40492 examples [00:43, 901.89 examples/s]40591 examples [00:44, 924.64 examples/s]40690 examples [00:44, 940.96 examples/s]40787 examples [00:44, 949.16 examples/s]40885 examples [00:44, 957.13 examples/s]40981 examples [00:44, 944.26 examples/s]41076 examples [00:44, 928.73 examples/s]41174 examples [00:44, 942.14 examples/s]41269 examples [00:44, 921.80 examples/s]41364 examples [00:44, 930.02 examples/s]41461 examples [00:45, 941.45 examples/s]41560 examples [00:45, 950.97 examples/s]41661 examples [00:45, 965.33 examples/s]41761 examples [00:45, 973.76 examples/s]41862 examples [00:45, 981.75 examples/s]41961 examples [00:45, 924.78 examples/s]42057 examples [00:45, 934.55 examples/s]42152 examples [00:45, 930.01 examples/s]42246 examples [00:45, 898.64 examples/s]42337 examples [00:45, 882.09 examples/s]42426 examples [00:46, 867.10 examples/s]42516 examples [00:46, 876.01 examples/s]42604 examples [00:46, 873.02 examples/s]42692 examples [00:46, 861.14 examples/s]42780 examples [00:46, 865.80 examples/s]42870 examples [00:46, 873.99 examples/s]42966 examples [00:46, 896.90 examples/s]43066 examples [00:46, 924.66 examples/s]43164 examples [00:46, 937.16 examples/s]43259 examples [00:46, 935.21 examples/s]43353 examples [00:47, 915.49 examples/s]43445 examples [00:47, 910.23 examples/s]43537 examples [00:47, 899.74 examples/s]43628 examples [00:47, 878.69 examples/s]43722 examples [00:47, 896.14 examples/s]43812 examples [00:47, 873.25 examples/s]43900 examples [00:47, 860.56 examples/s]43992 examples [00:47, 874.34 examples/s]44080 examples [00:47, 866.47 examples/s]44169 examples [00:48, 870.48 examples/s]44257 examples [00:48, 848.48 examples/s]44343 examples [00:48, 839.18 examples/s]44441 examples [00:48, 876.60 examples/s]44538 examples [00:48, 901.95 examples/s]44629 examples [00:48, 877.73 examples/s]44728 examples [00:48, 907.01 examples/s]44829 examples [00:48, 934.43 examples/s]44926 examples [00:48, 944.68 examples/s]45027 examples [00:48, 962.36 examples/s]45124 examples [00:49, 963.15 examples/s]45223 examples [00:49, 970.17 examples/s]45323 examples [00:49, 978.89 examples/s]45422 examples [00:49, 980.44 examples/s]45521 examples [00:49, 957.00 examples/s]45617 examples [00:49, 938.91 examples/s]45712 examples [00:49, 929.31 examples/s]45806 examples [00:49, 929.47 examples/s]45900 examples [00:49, 922.56 examples/s]45998 examples [00:49, 937.33 examples/s]46094 examples [00:50, 942.75 examples/s]46192 examples [00:50, 951.94 examples/s]46290 examples [00:50, 958.82 examples/s]46389 examples [00:50, 967.46 examples/s]46486 examples [00:50, 955.81 examples/s]46582 examples [00:50, 922.62 examples/s]46677 examples [00:50, 928.16 examples/s]46778 examples [00:50, 950.33 examples/s]46881 examples [00:50, 970.33 examples/s]46979 examples [00:50, 961.95 examples/s]47076 examples [00:51, 954.72 examples/s]47176 examples [00:51, 966.00 examples/s]47276 examples [00:51, 973.13 examples/s]47375 examples [00:51, 976.82 examples/s]47474 examples [00:51, 978.09 examples/s]47572 examples [00:51, 975.32 examples/s]47671 examples [00:51, 978.22 examples/s]47770 examples [00:51, 979.90 examples/s]47869 examples [00:51, 965.00 examples/s]47966 examples [00:52, 947.05 examples/s]48061 examples [00:52, 905.38 examples/s]48153 examples [00:52, 875.88 examples/s]48252 examples [00:52, 905.38 examples/s]48344 examples [00:52, 891.04 examples/s]48438 examples [00:52, 903.85 examples/s]48532 examples [00:52, 914.09 examples/s]48632 examples [00:52, 936.57 examples/s]48731 examples [00:52, 950.55 examples/s]48827 examples [00:52, 951.97 examples/s]48923 examples [00:53, 944.40 examples/s]49018 examples [00:53, 938.55 examples/s]49112 examples [00:53, 937.77 examples/s]49206 examples [00:53, 918.22 examples/s]49298 examples [00:53, 890.57 examples/s]49395 examples [00:53, 912.75 examples/s]49487 examples [00:53, 903.27 examples/s]49583 examples [00:53, 917.88 examples/s]49682 examples [00:53, 935.48 examples/s]49780 examples [00:53, 946.37 examples/s]49878 examples [00:54, 955.14 examples/s]49974 examples [00:54, 952.41 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6797/50000 [00:00<00:00, 67964.23 examples/s] 36%|███▌      | 17992/50000 [00:00<00:00, 77044.73 examples/s] 58%|█████▊    | 29170/50000 [00:00<00:00, 84965.35 examples/s] 79%|███████▉  | 39411/50000 [00:00<00:00, 89389.92 examples/s] 99%|█████████▉| 49515/50000 [00:00<00:00, 92591.62 examples/s]                                                               0 examples [00:00, ? examples/s]77 examples [00:00, 763.95 examples/s]172 examples [00:00, 810.40 examples/s]270 examples [00:00, 854.14 examples/s]360 examples [00:00, 866.03 examples/s]459 examples [00:00, 897.95 examples/s]561 examples [00:00, 928.94 examples/s]664 examples [00:00, 955.21 examples/s]763 examples [00:00, 964.71 examples/s]862 examples [00:00, 969.61 examples/s]957 examples [00:01, 962.01 examples/s]1060 examples [00:01, 980.16 examples/s]1158 examples [00:01, 978.55 examples/s]1256 examples [00:01, 963.49 examples/s]1352 examples [00:01, 949.54 examples/s]1447 examples [00:01, 942.67 examples/s]1542 examples [00:01, 934.14 examples/s]1636 examples [00:01, 931.18 examples/s]1730 examples [00:01, 932.04 examples/s]1824 examples [00:01, 926.04 examples/s]1921 examples [00:02, 937.90 examples/s]2022 examples [00:02, 956.57 examples/s]2118 examples [00:02, 955.98 examples/s]2216 examples [00:02, 959.08 examples/s]2313 examples [00:02, 962.13 examples/s]2410 examples [00:02, 948.94 examples/s]2509 examples [00:02, 960.51 examples/s]2607 examples [00:02, 963.08 examples/s]2704 examples [00:02, 896.07 examples/s]2798 examples [00:02, 908.69 examples/s]2890 examples [00:03, 900.90 examples/s]2983 examples [00:03, 907.23 examples/s]3075 examples [00:03, 900.86 examples/s]3166 examples [00:03, 874.66 examples/s]3261 examples [00:03, 894.28 examples/s]3358 examples [00:03, 914.71 examples/s]3454 examples [00:03, 925.41 examples/s]3547 examples [00:03, 922.59 examples/s]3640 examples [00:03, 918.37 examples/s]3732 examples [00:04, 881.82 examples/s]3825 examples [00:04, 895.44 examples/s]3918 examples [00:04, 905.08 examples/s]4014 examples [00:04, 920.46 examples/s]4115 examples [00:04, 944.75 examples/s]4215 examples [00:04, 959.11 examples/s]4312 examples [00:04, 954.02 examples/s]4410 examples [00:04, 961.11 examples/s]4510 examples [00:04, 970.63 examples/s]4612 examples [00:04, 981.91 examples/s]4711 examples [00:05, 979.69 examples/s]4811 examples [00:05, 985.29 examples/s]4913 examples [00:05, 995.21 examples/s]5014 examples [00:05, 998.97 examples/s]5116 examples [00:05, 1004.44 examples/s]5217 examples [00:05, 999.50 examples/s] 5317 examples [00:05, 941.63 examples/s]5415 examples [00:05, 951.44 examples/s]5516 examples [00:05, 966.41 examples/s]5618 examples [00:05, 980.12 examples/s]5717 examples [00:06, 979.74 examples/s]5816 examples [00:06, 969.32 examples/s]5914 examples [00:06, 959.72 examples/s]6011 examples [00:06, 955.46 examples/s]6107 examples [00:06, 946.34 examples/s]6202 examples [00:06, 945.19 examples/s]6300 examples [00:06, 952.99 examples/s]6396 examples [00:06, 939.46 examples/s]6491 examples [00:06, 894.04 examples/s]6583 examples [00:06, 899.27 examples/s]6679 examples [00:07, 916.49 examples/s]6771 examples [00:07, 914.84 examples/s]6863 examples [00:07, 906.68 examples/s]6954 examples [00:07, 876.83 examples/s]7043 examples [00:07, 876.41 examples/s]7131 examples [00:07, 850.21 examples/s]7218 examples [00:07, 853.11 examples/s]7304 examples [00:07, 815.51 examples/s]7397 examples [00:07, 845.55 examples/s]7493 examples [00:08, 874.62 examples/s]7590 examples [00:08, 899.07 examples/s]7691 examples [00:08, 927.68 examples/s]7793 examples [00:08, 952.40 examples/s]7891 examples [00:08, 958.97 examples/s]7988 examples [00:08, 961.57 examples/s]8086 examples [00:08, 964.58 examples/s]8183 examples [00:08, 953.68 examples/s]8279 examples [00:08, 952.08 examples/s]8375 examples [00:08, 952.78 examples/s]8471 examples [00:09, 921.72 examples/s]8564 examples [00:09, 920.48 examples/s]8662 examples [00:09, 936.30 examples/s]8763 examples [00:09, 956.14 examples/s]8864 examples [00:09, 969.53 examples/s]8962 examples [00:09, 971.10 examples/s]9060 examples [00:09, 946.95 examples/s]9155 examples [00:09, 929.90 examples/s]9249 examples [00:09, 930.72 examples/s]9346 examples [00:09, 940.29 examples/s]9441 examples [00:10, 943.11 examples/s]9536 examples [00:10, 941.22 examples/s]9631 examples [00:10, 935.72 examples/s]9729 examples [00:10, 947.83 examples/s]9827 examples [00:10, 956.42 examples/s]9923 examples [00:10, 954.03 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteIX6T5C/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteIX6T5C/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f97e4f08950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f97e4f08950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f97e4f08950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f976b233a58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f976b22ea20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f97e4f08950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f97e4f08950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f97e4f08950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f97e3477160>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f97e3477048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f975d887ea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f975d887ea0> 

  function with postional parmater data_info <function split_train_valid at 0x7f975d887ea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f975d886048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f975d886048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f975d886048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=afab36db8f9a3033669a1a5de57f2da02cb069aa2f3ef43b1074e8fee009153d
  Stored in directory: /tmp/pip-ephem-wheel-cache-yoil3emn/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:00:33, 13.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:50:22, 18.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:02:29, 26.5kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:20:16, 37.7kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:25:33, 53.9kB/s].vector_cache/glove.6B.zip:   1%|          | 8.55M/862M [00:01<3:04:54, 76.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 12.9M/862M [00:01<2:08:52, 110kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.8M/862M [00:01<1:29:54, 157kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.3M/862M [00:01<1:02:48, 223kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.1M/862M [00:01<43:57, 318kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<30:46, 453kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.9M/862M [00:01<21:31, 644kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:01<15:18, 903kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.6M/862M [00:02<10:46, 1.28MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:02<07:33, 1.81MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.9M/862M [00:02<05:23, 2.52MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<03:49, 3.53MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<06:40, 2.02MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<06:34, 2.04MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<09:40, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<07:53, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:04<06:02, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<06:30, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<05:56, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:06<04:27, 2.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<06:11, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<07:01, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<05:35, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:08<04:03, 3.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<53:56, 245kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<39:05, 338kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:10<27:38, 477kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<22:23, 588kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<16:58, 775kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<12:08, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<11:36, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<10:46, 1.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<08:07, 1.61MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:14<05:54, 2.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<09:28, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<12:02, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:17<09:51, 1.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:17<07:20, 1.77MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<07:43, 1.68MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<09:43, 1.33MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<07:54, 1.64MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<05:45, 2.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<07:45, 1.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<09:40, 1.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<07:43, 1.67MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:21<05:56, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:21<04:18, 2.98MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<14:29, 884kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<14:13, 902kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<11:00, 1.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:23<08:09, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:23<05:52, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<13:41, 932kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<13:31, 943kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<10:24, 1.22MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<07:44, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<05:33, 2.28MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<19:08, 663kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<17:06, 741kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<13:03, 970kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<09:33, 1.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<06:51, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<14:10, 890kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<13:38, 924kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:27, 1.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<07:32, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:30, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<10:17, 1.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:03, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:05, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:23, 2.84MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<31:19, 398kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<25:44, 485kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<18:49, 662kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<13:32, 919kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<09:37, 1.29MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<5:50:22, 35.4kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<4:08:48, 49.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<2:54:55, 70.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<2:02:19, 101kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:30:03, 137kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:09:44, 177kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<50:39, 243kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<35:47, 344kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<25:06, 488kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<45:24, 270kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<35:18, 347kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<25:27, 481kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<17:58, 680kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<16:20, 746kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<14:57, 815kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<11:13, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<08:04, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:49, 1.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:22, 1.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:57, 1.52MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:46, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:29, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:44, 1.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:48, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:04, 2.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:07, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:42, 1.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:12, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:33, 2.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:35, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:46, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:57, 1.71MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<05:03, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:48, 1.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:56, 1.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:04, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<05:09, 2.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<07:55, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:41, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:55, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:02, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:41, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:34, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:39, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:53, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:23, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:37, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:08, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:30, 2.57MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:26, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<08:20, 1.39MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:38, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:50, 2.38MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:39, 1.50MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<08:28, 1.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:43, 1.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:54, 2.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:40, 1.49MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:29, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:37, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:49, 2.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:28, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:37, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:10, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<04:31, 2.50MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:09, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:06, 1.39MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:27, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:42, 2.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:14, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:07, 1.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:19, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:35, 2.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:42, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:43, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:10, 1.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:30, 2.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:06, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:59, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:21, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:39, 2.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<07:11, 1.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<08:02, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:16, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<04:34, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<06:19, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<07:24, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:49, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:18, 2.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:32, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:51, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:31, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:03, 2.67MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:48, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<07:42, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:06, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:27, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<07:19, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<08:02, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:14, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:33, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:07, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:50, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:12, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:32, 2.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<06:57, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<07:35, 1.40MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:00, 1.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:21, 2.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<07:03, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<07:47, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:11, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:29, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<06:55, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:22, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:49, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:12, 2.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<07:36, 1.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<07:51, 1.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:02, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:24, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<06:02, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<06:52, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:29, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<04:01, 2.55MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<06:36, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<07:00, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:31, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:00, 2.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<07:08, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<07:22, 1.38MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:46, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:12, 2.41MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<07:25, 1.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<08:04, 1.25MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:16, 1.61MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:33, 2.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<06:36, 1.52MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<06:50, 1.47MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:20, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<03:52, 2.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<09:52, 1.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<09:03, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:46, 1.47MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:57, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<05:58, 1.66MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<06:15, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:50, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:32, 2.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<09:24, 1.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<08:34, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<06:26, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:41, 2.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:06, 1.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<06:32, 1.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:04, 1.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<03:46, 2.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:57, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:52<05:42, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:34, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:18, 2.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:57, 1.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<07:47, 1.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:55, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:17, 2.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<06:04, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:56<06:01, 1.59MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:39, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:21, 2.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<1:10:50, 134kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<51:39, 184kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<36:36, 259kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<25:36, 368kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<25:03, 376kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<19:37, 480kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<14:09, 664kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<10:02, 934kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<10:08, 923kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<08:42, 1.07MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:25, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:35, 2.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<12:06, 767kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<10:26, 889kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<07:42, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:30, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:48, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:24, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:36, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<04:00, 2.28MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<10:58, 833kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<09:38, 948kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<07:14, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<05:09, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<22:56, 396kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<17:57, 505kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<12:57, 699kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<09:14, 978kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<08:53, 1.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<08:14, 1.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:13, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:31, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:39, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:13, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:55, 2.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<02:51, 3.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<09:08, 970kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<08:07, 1.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:06, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<04:21, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<13:32, 650kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<11:12, 785kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<08:15, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<05:50, 1.50MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<18:41, 467kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<14:46, 591kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<10:40, 816kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<07:36, 1.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<08:14, 1.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<07:26, 1.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:37, 1.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:19, 1.61MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:23, 1.59MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:07, 2.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:02, 2.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<04:54, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:01, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:55, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:07, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:31, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<03:34, 2.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<03:52, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:20, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:25, 2.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<03:46, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:16, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:20, 2.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:24, 3.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<15:15, 541kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<14:15, 579kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<10:44, 767kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<07:41, 1.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:18, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<06:38, 1.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:01, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<04:50, 1.67MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<04:59, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:49, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:47, 2.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<05:38, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:29, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:10, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:00, 2.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<06:29, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:41<06:01, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:31, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:19, 2.39MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<04:45, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<04:53, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:48, 2.08MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:56, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:40, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<04:37, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:23, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:05, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:18, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:22, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<03:37, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<05:25, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<04:33, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:20, 2.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<04:08, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<04:22, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:25, 2.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<03:37, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<04:00, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:10, 2.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:17, 3.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<14:29, 517kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<11:32, 650kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<08:22, 893kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<05:56, 1.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<08:06, 916kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<07:03, 1.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<05:17, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:53, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<04:25, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:17, 2.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:42, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<05:07, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<04:14, 1.72MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:05, 2.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:01, 1.79MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:06, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:09, 2.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:16, 3.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<26:30, 270kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<21:01, 340kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<15:21, 465kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<10:50, 656kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<09:23, 754kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:51, 902kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<05:47, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:13, 1.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:15, 1.12MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<04:59, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:36, 1.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:19, 1.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:13, 1.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:12, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:18, 2.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<11:06, 619kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<09:00, 763kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<06:36, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:44, 1.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:15, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:55, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:48, 2.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<07:56, 848kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<07:56, 849kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:04, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:22, 1.53MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:34, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:24, 1.51MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:22, 1.97MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:27, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:53, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:00, 1.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:56, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:45, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:45, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:54, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:07, 2.07MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:20, 1.93MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:35, 2.49MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<01:53, 3.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:38, 1.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:28, 1.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:23, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:11, 1.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:52, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:50, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:55, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:06, 2.97MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<06:47, 921kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<07:05, 882kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<05:30, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:57, 1.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:23, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:08, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:07, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:16, 2.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:23, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:10, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:11, 1.91MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:14, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:22, 1.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:36, 1.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:38, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:24, 1.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:27, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:40, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:51, 2.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:01, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:22, 2.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:38, 2.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:52, 1.51MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:13, 1.81MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:22, 2.44MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:10, 1.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:13, 1.79MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:28, 2.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<01:46, 3.21MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<08:17, 687kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<07:46, 734kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<05:56, 958kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:15, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<04:26, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<04:06, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:05, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:12, 2.52MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<07:54, 704kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<07:27, 747kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:43, 972kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<04:05, 1.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<04:15, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:58, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:00, 1.82MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<03:00, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:04, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:21, 2.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:33, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:38, 1.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:01, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:12, 2.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:56, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:59, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:19, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:30, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:34, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:54, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:08, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:47, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:52, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:14, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:24, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<03:34, 1.42MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:57, 1.72MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:09, 2.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:48, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:49, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:12, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:21, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:32, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<01:57, 2.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:24, 3.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<08:13, 593kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<07:34, 644kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<05:43, 851kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<04:05, 1.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:06, 1.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<03:42, 1.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:46, 1.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:58, 2.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<06:04, 780kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:06, 928kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:46, 1.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:24, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:59, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<03:12, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:20, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:49, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:46, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:06, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:30, 3.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<08:44, 518kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<06:55, 654kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<05:02, 896kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<04:14, 1.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:45, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:49, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:42, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:25, 1.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:47, 1.57MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:01, 2.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:32, 1.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:32, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:58, 2.19MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:05, 2.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:56, 1.45MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:26, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:47, 2.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:20, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:22, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:50, 2.26MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:58, 2.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:07, 1.94MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:39, 2.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:50, 2.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:00, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:33, 2.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:08, 3.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:36, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:30, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:55, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:59, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:05, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:38, 2.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:10, 3.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<13:18, 289kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<09:59, 385kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<07:07, 536kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<05:34, 678kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<04:34, 825kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:20, 1.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<02:21, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<04:19, 857kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<04:18, 860kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:17, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:22, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:34, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:27, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:51, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:20, 2.68MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:37, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:03, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:24, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:45, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:08, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:07, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:38, 2.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:42, 2.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:47, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:22, 2.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:31, 2.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<02:14, 1.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:52, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:22, 2.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:48, 1.82MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:51, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:25, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:01, 3.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<03:48, 848kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<03:15, 990kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:24, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:11, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:42, 1.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:07, 1.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:31, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:52, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:51, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:25, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:29, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:05, 1.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:44, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:16, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:39, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:40, 1.75MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:18, 2.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:23, 2.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:57, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:37, 1.76MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:11, 2.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:32, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:35, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:12, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:51, 3.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<20:56, 131kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<15:36, 176kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<11:08, 245kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<07:46, 348kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<06:03, 441kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<04:43, 566kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<03:24, 780kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:46, 937kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<02:24, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:47, 1.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:39, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<02:02, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:38, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:11, 2.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:28, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:28, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:07, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:47, 3.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:49, 851kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:51, 837kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:10, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:33, 1.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:39, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:33, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:11, 1.94MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:11, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:40, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:20, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:59, 2.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:14, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:15, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:58, 2.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:01, 2.07MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:04, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:50, 2.49MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:55, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:24, 1.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:09, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:50, 2.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:06, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:06, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:51, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:54, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:20, 1.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:06, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:48, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:02, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:03, 1.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:48, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:34, 3.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<03:39, 486kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:51, 620kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<02:03, 852kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:41, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:30, 1.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:07, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:48, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:59, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:59, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:45, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:46, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:49, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:38, 2.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:41, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:00, 1.50MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:49, 1.80MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:36, 2.43MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:47, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:47, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:36, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:38, 2.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:55, 1.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:33, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:42, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:43, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:33, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:23, 3.15MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:16, 957kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:19, 928kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:01, 1.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:43, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:48, 1.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:45, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:34, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:23, 2.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<03:51, 282kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<02:53, 376kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<02:02, 524kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:32, 665kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:15, 811kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:54, 1.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:45, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:41, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:31, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:29, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:30, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:23, 2.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:23, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:33, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:19, 2.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:24, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:25, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:18, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:12, 3.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:51, 781kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:51, 789kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:39, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:26, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:27, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:25, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:17, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:18, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:13, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:13, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:14, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:10, 2.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:15, 1.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:08, 2.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:11, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:10, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:05, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 3.43MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:09, 339kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:06, 446kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:03, 620kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<87:58:29,  1.26it/s]  0%|          | 674/400000 [00:00<61:29:01,  1.80it/s]  0%|          | 1407/400000 [00:00<42:57:50,  2.58it/s]  1%|          | 2093/400000 [00:01<30:01:40,  3.68it/s]  1%|          | 2831/400000 [00:01<20:59:06,  5.26it/s]  1%|          | 3575/400000 [00:01<14:39:59,  7.51it/s]  1%|          | 4239/400000 [00:01<10:15:15, 10.72it/s]  1%|          | 4971/400000 [00:01<7:10:09, 15.31it/s]   1%|▏         | 5690/400000 [00:01<5:00:50, 21.85it/s]  2%|▏         | 6427/400000 [00:01<3:30:27, 31.17it/s]  2%|▏         | 7172/400000 [00:01<2:27:18, 44.45it/s]  2%|▏         | 7880/400000 [00:01<1:43:13, 63.31it/s]  2%|▏         | 8569/400000 [00:02<1:12:26, 90.06it/s]  2%|▏         | 9288/400000 [00:02<50:53, 127.98it/s]   3%|▎         | 10031/400000 [00:02<35:48, 181.48it/s]  3%|▎         | 10777/400000 [00:02<25:16, 256.59it/s]  3%|▎         | 11495/400000 [00:02<17:56, 360.79it/s]  3%|▎         | 12205/400000 [00:02<12:50, 503.50it/s]  3%|▎         | 12898/400000 [00:02<09:15, 696.52it/s]  3%|▎         | 13581/400000 [00:02<06:45, 953.15it/s]  4%|▎         | 14304/400000 [00:02<04:59, 1288.77it/s]  4%|▎         | 14998/400000 [00:02<03:46, 1701.17it/s]  4%|▍         | 15686/400000 [00:03<02:55, 2185.73it/s]  4%|▍         | 16363/400000 [00:03<02:20, 2728.12it/s]  4%|▍         | 17055/400000 [00:03<01:54, 3333.51it/s]  4%|▍         | 17735/400000 [00:03<01:37, 3934.65it/s]  5%|▍         | 18427/400000 [00:03<01:24, 4519.23it/s]  5%|▍         | 19173/400000 [00:03<01:14, 5123.90it/s]  5%|▍         | 19887/400000 [00:03<01:07, 5596.85it/s]  5%|▌         | 20617/400000 [00:03<01:03, 6017.95it/s]  5%|▌         | 21339/400000 [00:03<00:59, 6330.09it/s]  6%|▌         | 22054/400000 [00:03<00:58, 6482.50it/s]  6%|▌         | 22761/400000 [00:04<00:56, 6627.41it/s]  6%|▌         | 23475/400000 [00:04<00:55, 6771.49it/s]  6%|▌         | 24205/400000 [00:04<00:54, 6918.92it/s]  6%|▌         | 24919/400000 [00:04<00:55, 6751.29it/s]  6%|▋         | 25611/400000 [00:04<00:56, 6649.95it/s]  7%|▋         | 26292/400000 [00:04<00:55, 6696.49it/s]  7%|▋         | 27021/400000 [00:04<00:54, 6862.45it/s]  7%|▋         | 27757/400000 [00:04<00:53, 7003.81it/s]  7%|▋         | 28480/400000 [00:04<00:52, 7068.87it/s]  7%|▋         | 29191/400000 [00:04<00:55, 6675.99it/s]  7%|▋         | 29922/400000 [00:05<00:53, 6854.12it/s]  8%|▊         | 30651/400000 [00:05<00:52, 6976.10it/s]  8%|▊         | 31369/400000 [00:05<00:52, 7035.36it/s]  8%|▊         | 32077/400000 [00:05<00:53, 6889.55it/s]  8%|▊         | 32793/400000 [00:05<00:52, 6966.48it/s]  8%|▊         | 33501/400000 [00:05<00:52, 6999.57it/s]  9%|▊         | 34246/400000 [00:05<00:51, 7128.25it/s]  9%|▊         | 34991/400000 [00:05<00:50, 7219.73it/s]  9%|▉         | 35715/400000 [00:05<00:50, 7166.25it/s]  9%|▉         | 36463/400000 [00:05<00:50, 7256.56it/s]  9%|▉         | 37216/400000 [00:06<00:49, 7334.07it/s]  9%|▉         | 37969/400000 [00:06<00:48, 7389.14it/s] 10%|▉         | 38709/400000 [00:06<00:49, 7314.11it/s] 10%|▉         | 39442/400000 [00:06<00:50, 7202.34it/s] 10%|█         | 40177/400000 [00:06<00:49, 7245.89it/s] 10%|█         | 40926/400000 [00:06<00:49, 7315.35it/s] 10%|█         | 41673/400000 [00:06<00:48, 7359.84it/s] 11%|█         | 42410/400000 [00:06<00:48, 7361.61it/s] 11%|█         | 43149/400000 [00:06<00:48, 7367.57it/s] 11%|█         | 43887/400000 [00:07<00:48, 7356.30it/s] 11%|█         | 44623/400000 [00:07<00:49, 7227.56it/s] 11%|█▏        | 45371/400000 [00:07<00:48, 7300.61it/s] 12%|█▏        | 46119/400000 [00:07<00:48, 7351.94it/s] 12%|█▏        | 46855/400000 [00:07<00:48, 7223.71it/s] 12%|█▏        | 47596/400000 [00:07<00:48, 7277.74it/s] 12%|█▏        | 48325/400000 [00:07<00:48, 7205.42it/s] 12%|█▏        | 49051/400000 [00:07<00:48, 7219.75it/s] 12%|█▏        | 49799/400000 [00:07<00:48, 7294.07it/s] 13%|█▎        | 50529/400000 [00:07<00:48, 7225.57it/s] 13%|█▎        | 51253/400000 [00:08<00:48, 7205.23it/s] 13%|█▎        | 51974/400000 [00:08<00:49, 6999.49it/s] 13%|█▎        | 52717/400000 [00:08<00:48, 7121.63it/s] 13%|█▎        | 53452/400000 [00:08<00:48, 7173.97it/s] 14%|█▎        | 54171/400000 [00:08<00:48, 7121.88it/s] 14%|█▎        | 54911/400000 [00:08<00:47, 7201.30it/s] 14%|█▍        | 55651/400000 [00:08<00:47, 7259.29it/s] 14%|█▍        | 56387/400000 [00:08<00:47, 7288.34it/s] 14%|█▍        | 57117/400000 [00:08<00:47, 7217.78it/s] 14%|█▍        | 57840/400000 [00:08<00:47, 7219.90it/s] 15%|█▍        | 58568/400000 [00:09<00:47, 7235.39it/s] 15%|█▍        | 59295/400000 [00:09<00:47, 7243.89it/s] 15%|█▌        | 60020/400000 [00:09<00:48, 7053.38it/s] 15%|█▌        | 60752/400000 [00:09<00:47, 7129.94it/s] 15%|█▌        | 61467/400000 [00:09<00:47, 7077.56it/s] 16%|█▌        | 62176/400000 [00:09<00:48, 6940.51it/s] 16%|█▌        | 62872/400000 [00:09<00:49, 6754.21it/s] 16%|█▌        | 63550/400000 [00:09<00:50, 6690.12it/s] 16%|█▌        | 64221/400000 [00:09<00:50, 6651.02it/s] 16%|█▌        | 64888/400000 [00:09<00:50, 6645.38it/s] 16%|█▋        | 65615/400000 [00:10<00:49, 6818.98it/s] 17%|█▋        | 66345/400000 [00:10<00:47, 6954.49it/s] 17%|█▋        | 67074/400000 [00:10<00:47, 7050.30it/s] 17%|█▋        | 67816/400000 [00:10<00:46, 7155.36it/s] 17%|█▋        | 68534/400000 [00:10<00:48, 6856.09it/s] 17%|█▋        | 69224/400000 [00:10<00:48, 6808.82it/s] 17%|█▋        | 69940/400000 [00:10<00:48, 6856.42it/s] 18%|█▊        | 70644/400000 [00:10<00:47, 6908.83it/s] 18%|█▊        | 71383/400000 [00:10<00:46, 7045.74it/s] 18%|█▊        | 72113/400000 [00:11<00:46, 7118.51it/s] 18%|█▊        | 72851/400000 [00:11<00:45, 7192.05it/s] 18%|█▊        | 73572/400000 [00:11<00:46, 7081.31it/s] 19%|█▊        | 74304/400000 [00:11<00:45, 7149.56it/s] 19%|█▉        | 75025/400000 [00:11<00:45, 7166.93it/s] 19%|█▉        | 75743/400000 [00:11<00:45, 7059.16it/s] 19%|█▉        | 76450/400000 [00:11<00:46, 6908.83it/s] 19%|█▉        | 77143/400000 [00:11<00:47, 6814.25it/s] 19%|█▉        | 77832/400000 [00:11<00:47, 6836.46it/s] 20%|█▉        | 78564/400000 [00:11<00:46, 6973.24it/s] 20%|█▉        | 79263/400000 [00:12<00:46, 6929.69it/s] 20%|█▉        | 79957/400000 [00:12<00:46, 6906.15it/s] 20%|██        | 80649/400000 [00:12<00:47, 6782.62it/s] 20%|██        | 81329/400000 [00:12<00:47, 6751.76it/s] 21%|██        | 82005/400000 [00:12<00:47, 6663.23it/s] 21%|██        | 82705/400000 [00:12<00:46, 6759.24it/s] 21%|██        | 83418/400000 [00:12<00:46, 6864.01it/s] 21%|██        | 84152/400000 [00:12<00:45, 6997.80it/s] 21%|██        | 84877/400000 [00:12<00:44, 7071.52it/s] 21%|██▏       | 85599/400000 [00:12<00:44, 7114.21it/s] 22%|██▏       | 86312/400000 [00:13<00:44, 7022.07it/s] 22%|██▏       | 87023/400000 [00:13<00:44, 7045.96it/s] 22%|██▏       | 87732/400000 [00:13<00:44, 7057.15it/s] 22%|██▏       | 88461/400000 [00:13<00:43, 7122.83it/s] 22%|██▏       | 89174/400000 [00:13<00:43, 7094.45it/s] 22%|██▏       | 89893/400000 [00:13<00:43, 7121.16it/s] 23%|██▎       | 90606/400000 [00:13<00:44, 7010.99it/s] 23%|██▎       | 91343/400000 [00:13<00:43, 7112.35it/s] 23%|██▎       | 92055/400000 [00:13<00:43, 6999.95it/s] 23%|██▎       | 92791/400000 [00:13<00:43, 7103.73it/s] 23%|██▎       | 93503/400000 [00:14<00:43, 7078.95it/s] 24%|██▎       | 94215/400000 [00:14<00:43, 7090.40it/s] 24%|██▎       | 94925/400000 [00:14<00:43, 7040.69it/s] 24%|██▍       | 95630/400000 [00:14<00:43, 7002.70it/s] 24%|██▍       | 96339/400000 [00:14<00:43, 7025.97it/s] 24%|██▍       | 97042/400000 [00:14<00:43, 6965.00it/s] 24%|██▍       | 97779/400000 [00:14<00:42, 7080.78it/s] 25%|██▍       | 98488/400000 [00:14<00:42, 7075.96it/s] 25%|██▍       | 99233/400000 [00:14<00:41, 7181.52it/s] 25%|██▍       | 99975/400000 [00:14<00:41, 7250.31it/s] 25%|██▌       | 100701/400000 [00:15<00:41, 7243.71it/s] 25%|██▌       | 101441/400000 [00:15<00:40, 7287.84it/s] 26%|██▌       | 102171/400000 [00:15<00:41, 7217.43it/s] 26%|██▌       | 102910/400000 [00:15<00:40, 7266.70it/s] 26%|██▌       | 103659/400000 [00:15<00:40, 7331.65it/s] 26%|██▌       | 104393/400000 [00:15<00:40, 7219.65it/s] 26%|██▋       | 105116/400000 [00:15<00:42, 7007.70it/s] 26%|██▋       | 105819/400000 [00:15<00:43, 6818.56it/s] 27%|██▋       | 106504/400000 [00:15<00:43, 6735.24it/s] 27%|██▋       | 107180/400000 [00:16<00:45, 6478.21it/s] 27%|██▋       | 107832/400000 [00:16<00:45, 6458.39it/s] 27%|██▋       | 108583/400000 [00:16<00:43, 6739.86it/s] 27%|██▋       | 109322/400000 [00:16<00:42, 6919.35it/s] 28%|██▊       | 110070/400000 [00:16<00:40, 7077.22it/s] 28%|██▊       | 110782/400000 [00:16<00:42, 6811.24it/s] 28%|██▊       | 111469/400000 [00:16<00:42, 6766.35it/s] 28%|██▊       | 112194/400000 [00:16<00:41, 6903.46it/s] 28%|██▊       | 112888/400000 [00:16<00:41, 6900.06it/s] 28%|██▊       | 113581/400000 [00:16<00:42, 6719.55it/s] 29%|██▊       | 114256/400000 [00:17<00:42, 6724.47it/s] 29%|██▊       | 114987/400000 [00:17<00:41, 6889.64it/s] 29%|██▉       | 115739/400000 [00:17<00:40, 7064.63it/s] 29%|██▉       | 116468/400000 [00:17<00:39, 7128.09it/s] 29%|██▉       | 117183/400000 [00:17<00:40, 6973.67it/s] 29%|██▉       | 117897/400000 [00:17<00:40, 7022.14it/s] 30%|██▉       | 118601/400000 [00:17<00:40, 6954.73it/s] 30%|██▉       | 119345/400000 [00:17<00:39, 7089.84it/s] 30%|███       | 120095/400000 [00:17<00:38, 7207.77it/s] 30%|███       | 120843/400000 [00:17<00:38, 7285.58it/s] 30%|███       | 121584/400000 [00:18<00:38, 7322.20it/s] 31%|███       | 122318/400000 [00:18<00:38, 7260.40it/s] 31%|███       | 123066/400000 [00:18<00:37, 7322.09it/s] 31%|███       | 123814/400000 [00:18<00:37, 7366.69it/s] 31%|███       | 124552/400000 [00:18<00:37, 7356.65it/s] 31%|███▏      | 125289/400000 [00:18<00:37, 7339.43it/s] 32%|███▏      | 126024/400000 [00:18<00:38, 7190.23it/s] 32%|███▏      | 126744/400000 [00:18<00:38, 7187.91it/s] 32%|███▏      | 127477/400000 [00:18<00:37, 7228.06it/s] 32%|███▏      | 128217/400000 [00:18<00:37, 7277.18it/s] 32%|███▏      | 128954/400000 [00:19<00:37, 7303.18it/s] 32%|███▏      | 129685/400000 [00:19<00:37, 7222.01it/s] 33%|███▎      | 130433/400000 [00:19<00:36, 7294.97it/s] 33%|███▎      | 131163/400000 [00:19<00:37, 7140.70it/s] 33%|███▎      | 131892/400000 [00:19<00:37, 7182.28it/s] 33%|███▎      | 132631/400000 [00:19<00:36, 7241.36it/s] 33%|███▎      | 133356/400000 [00:19<00:37, 7154.89it/s] 34%|███▎      | 134073/400000 [00:19<00:37, 7147.44it/s] 34%|███▎      | 134804/400000 [00:19<00:36, 7193.66it/s] 34%|███▍      | 135545/400000 [00:19<00:36, 7255.15it/s] 34%|███▍      | 136281/400000 [00:20<00:36, 7284.13it/s] 34%|███▍      | 137010/400000 [00:20<00:36, 7221.75it/s] 34%|███▍      | 137753/400000 [00:20<00:36, 7282.87it/s] 35%|███▍      | 138495/400000 [00:20<00:35, 7320.65it/s] 35%|███▍      | 139228/400000 [00:20<00:36, 7224.72it/s] 35%|███▍      | 139951/400000 [00:20<00:37, 7014.82it/s] 35%|███▌      | 140672/400000 [00:20<00:36, 7070.35it/s] 35%|███▌      | 141381/400000 [00:20<00:37, 6824.95it/s] 36%|███▌      | 142116/400000 [00:20<00:36, 6973.73it/s] 36%|███▌      | 142817/400000 [00:21<00:37, 6861.70it/s] 36%|███▌      | 143563/400000 [00:21<00:36, 7030.26it/s] 36%|███▌      | 144279/400000 [00:21<00:36, 7067.94it/s] 36%|███▋      | 145020/400000 [00:21<00:35, 7165.30it/s] 36%|███▋      | 145759/400000 [00:21<00:35, 7229.71it/s] 37%|███▋      | 146484/400000 [00:21<00:35, 7170.30it/s] 37%|███▋      | 147203/400000 [00:21<00:35, 7151.74it/s] 37%|███▋      | 147919/400000 [00:21<00:35, 7010.41it/s] 37%|███▋      | 148622/400000 [00:21<00:36, 6867.89it/s] 37%|███▋      | 149355/400000 [00:21<00:35, 6998.23it/s] 38%|███▊      | 150103/400000 [00:22<00:35, 7134.18it/s] 38%|███▊      | 150843/400000 [00:22<00:34, 7211.50it/s] 38%|███▊      | 151566/400000 [00:22<00:34, 7172.20it/s] 38%|███▊      | 152285/400000 [00:22<00:35, 7042.58it/s] 38%|███▊      | 153034/400000 [00:22<00:34, 7170.36it/s] 38%|███▊      | 153753/400000 [00:22<00:35, 7012.41it/s] 39%|███▊      | 154457/400000 [00:22<00:35, 7012.20it/s] 39%|███▉      | 155186/400000 [00:22<00:34, 7091.48it/s] 39%|███▉      | 155937/400000 [00:22<00:33, 7211.95it/s] 39%|███▉      | 156673/400000 [00:22<00:33, 7254.10it/s] 39%|███▉      | 157416/400000 [00:23<00:33, 7303.29it/s] 40%|███▉      | 158149/400000 [00:23<00:33, 7309.86it/s] 40%|███▉      | 158881/400000 [00:23<00:33, 7269.85it/s] 40%|███▉      | 159633/400000 [00:23<00:32, 7341.39it/s] 40%|████      | 160371/400000 [00:23<00:32, 7352.85it/s] 40%|████      | 161107/400000 [00:23<00:34, 7025.77it/s] 40%|████      | 161813/400000 [00:23<00:34, 6919.46it/s] 41%|████      | 162508/400000 [00:23<00:35, 6772.61it/s] 41%|████      | 163197/400000 [00:23<00:34, 6807.21it/s] 41%|████      | 163945/400000 [00:23<00:33, 6995.77it/s] 41%|████      | 164688/400000 [00:24<00:33, 7119.79it/s] 41%|████▏     | 165417/400000 [00:24<00:32, 7169.96it/s] 42%|████▏     | 166143/400000 [00:24<00:32, 7195.28it/s] 42%|████▏     | 166880/400000 [00:24<00:32, 7246.43it/s] 42%|████▏     | 167606/400000 [00:24<00:33, 6980.16it/s] 42%|████▏     | 168307/400000 [00:24<00:33, 6820.52it/s] 42%|████▏     | 168992/400000 [00:24<00:34, 6756.08it/s] 42%|████▏     | 169670/400000 [00:24<00:34, 6656.78it/s] 43%|████▎     | 170369/400000 [00:24<00:34, 6752.38it/s] 43%|████▎     | 171114/400000 [00:25<00:32, 6945.95it/s] 43%|████▎     | 171855/400000 [00:25<00:32, 7077.54it/s] 43%|████▎     | 172586/400000 [00:25<00:31, 7142.79it/s] 43%|████▎     | 173325/400000 [00:25<00:31, 7214.00it/s] 44%|████▎     | 174060/400000 [00:25<00:31, 7253.44it/s] 44%|████▎     | 174787/400000 [00:25<00:31, 7239.86it/s] 44%|████▍     | 175512/400000 [00:25<00:31, 7142.08it/s] 44%|████▍     | 176228/400000 [00:25<00:32, 6954.99it/s] 44%|████▍     | 176926/400000 [00:25<00:32, 6947.84it/s] 44%|████▍     | 177662/400000 [00:25<00:31, 7064.21it/s] 45%|████▍     | 178398/400000 [00:26<00:30, 7148.48it/s] 45%|████▍     | 179132/400000 [00:26<00:30, 7192.81it/s] 45%|████▍     | 179867/400000 [00:26<00:30, 7237.59it/s] 45%|████▌     | 180604/400000 [00:26<00:30, 7275.34it/s] 45%|████▌     | 181333/400000 [00:26<00:30, 7184.73it/s] 46%|████▌     | 182053/400000 [00:26<00:31, 6912.63it/s] 46%|████▌     | 182747/400000 [00:26<00:31, 6849.75it/s] 46%|████▌     | 183434/400000 [00:26<00:31, 6817.44it/s] 46%|████▌     | 184118/400000 [00:26<00:32, 6647.12it/s] 46%|████▌     | 184846/400000 [00:26<00:31, 6824.32it/s] 46%|████▋     | 185574/400000 [00:27<00:30, 6952.63it/s] 47%|████▋     | 186306/400000 [00:27<00:30, 7057.02it/s] 47%|████▋     | 187041/400000 [00:27<00:29, 7141.88it/s] 47%|████▋     | 187757/400000 [00:27<00:30, 7036.21it/s] 47%|████▋     | 188473/400000 [00:27<00:29, 7071.44it/s] 47%|████▋     | 189204/400000 [00:27<00:29, 7139.77it/s] 47%|████▋     | 189928/400000 [00:27<00:29, 7168.44it/s] 48%|████▊     | 190646/400000 [00:27<00:29, 7051.27it/s] 48%|████▊     | 191390/400000 [00:27<00:29, 7162.78it/s] 48%|████▊     | 192108/400000 [00:27<00:29, 7093.57it/s] 48%|████▊     | 192855/400000 [00:28<00:28, 7201.00it/s] 48%|████▊     | 193592/400000 [00:28<00:28, 7249.06it/s] 49%|████▊     | 194319/400000 [00:28<00:28, 7254.63it/s] 49%|████▉     | 195061/400000 [00:28<00:28, 7302.92it/s] 49%|████▉     | 195792/400000 [00:28<00:28, 7167.51it/s] 49%|████▉     | 196536/400000 [00:28<00:28, 7246.95it/s] 49%|████▉     | 197274/400000 [00:28<00:27, 7285.72it/s] 50%|████▉     | 198009/400000 [00:28<00:27, 7303.15it/s] 50%|████▉     | 198740/400000 [00:28<00:28, 7079.24it/s] 50%|████▉     | 199450/400000 [00:29<00:28, 6935.91it/s] 50%|█████     | 200146/400000 [00:29<00:29, 6878.89it/s] 50%|█████     | 200836/400000 [00:29<00:29, 6798.79it/s] 50%|█████     | 201518/400000 [00:29<00:29, 6684.37it/s] 51%|█████     | 202230/400000 [00:29<00:29, 6807.43it/s] 51%|█████     | 202944/400000 [00:29<00:28, 6902.00it/s] 51%|█████     | 203683/400000 [00:29<00:27, 7040.04it/s] 51%|█████     | 204402/400000 [00:29<00:27, 7081.84it/s] 51%|█████▏    | 205124/400000 [00:29<00:27, 7122.03it/s] 51%|█████▏    | 205858/400000 [00:29<00:27, 7186.01it/s] 52%|█████▏    | 206578/400000 [00:30<00:26, 7174.25it/s] 52%|█████▏    | 207307/400000 [00:30<00:26, 7205.80it/s] 52%|█████▏    | 208053/400000 [00:30<00:26, 7278.20it/s] 52%|█████▏    | 208782/400000 [00:30<00:26, 7203.94it/s] 52%|█████▏    | 209514/400000 [00:30<00:26, 7237.40it/s] 53%|█████▎    | 210248/400000 [00:30<00:26, 7257.20it/s] 53%|█████▎    | 210998/400000 [00:30<00:25, 7325.73it/s] 53%|█████▎    | 211731/400000 [00:30<00:26, 7063.40it/s] 53%|█████▎    | 212440/400000 [00:30<00:27, 6881.40it/s] 53%|█████▎    | 213176/400000 [00:30<00:26, 7017.00it/s] 53%|█████▎    | 213926/400000 [00:31<00:26, 7153.34it/s] 54%|█████▎    | 214668/400000 [00:31<00:25, 7231.03it/s] 54%|█████▍    | 215393/400000 [00:31<00:25, 7106.95it/s] 54%|█████▍    | 216106/400000 [00:31<00:26, 6912.36it/s] 54%|█████▍    | 216800/400000 [00:31<00:27, 6733.50it/s] 54%|█████▍    | 217477/400000 [00:31<00:27, 6697.55it/s] 55%|█████▍    | 218149/400000 [00:31<00:27, 6669.57it/s] 55%|█████▍    | 218818/400000 [00:31<00:27, 6624.12it/s] 55%|█████▍    | 219482/400000 [00:31<00:27, 6494.84it/s] 55%|█████▌    | 220133/400000 [00:31<00:27, 6495.49it/s] 55%|█████▌    | 220795/400000 [00:32<00:27, 6531.03it/s] 55%|█████▌    | 221449/400000 [00:32<00:27, 6490.02it/s] 56%|█████▌    | 222112/400000 [00:32<00:27, 6530.85it/s] 56%|█████▌    | 222766/400000 [00:32<00:27, 6490.68it/s] 56%|█████▌    | 223416/400000 [00:32<00:27, 6478.55it/s] 56%|█████▌    | 224085/400000 [00:32<00:26, 6539.69it/s] 56%|█████▌    | 224755/400000 [00:32<00:26, 6584.69it/s] 56%|█████▋    | 225507/400000 [00:32<00:25, 6838.06it/s] 57%|█████▋    | 226210/400000 [00:32<00:25, 6893.63it/s] 57%|█████▋    | 226902/400000 [00:32<00:25, 6819.84it/s] 57%|█████▋    | 227586/400000 [00:33<00:25, 6767.53it/s] 57%|█████▋    | 228298/400000 [00:33<00:24, 6868.15it/s] 57%|█████▋    | 229020/400000 [00:33<00:24, 6969.78it/s] 57%|█████▋    | 229748/400000 [00:33<00:24, 7059.08it/s] 58%|█████▊    | 230483/400000 [00:33<00:23, 7143.00it/s] 58%|█████▊    | 231218/400000 [00:33<00:23, 7201.82it/s] 58%|█████▊    | 231959/400000 [00:33<00:23, 7261.19it/s] 58%|█████▊    | 232691/400000 [00:33<00:22, 7278.22it/s] 58%|█████▊    | 233420/400000 [00:33<00:23, 7191.89it/s] 59%|█████▊    | 234160/400000 [00:34<00:22, 7252.34it/s] 59%|█████▊    | 234910/400000 [00:34<00:22, 7322.68it/s] 59%|█████▉    | 235653/400000 [00:34<00:22, 7352.84it/s] 59%|█████▉    | 236389/400000 [00:34<00:22, 7246.30it/s] 59%|█████▉    | 237115/400000 [00:34<00:22, 7178.39it/s] 59%|█████▉    | 237834/400000 [00:34<00:22, 7115.97it/s] 60%|█████▉    | 238581/400000 [00:34<00:22, 7216.56it/s] 60%|█████▉    | 239321/400000 [00:34<00:22, 7270.29it/s] 60%|██████    | 240059/400000 [00:34<00:21, 7301.28it/s] 60%|██████    | 240790/400000 [00:34<00:21, 7253.72it/s] 60%|██████    | 241516/400000 [00:35<00:22, 7015.38it/s] 61%|██████    | 242220/400000 [00:35<00:22, 6928.22it/s] 61%|██████    | 242915/400000 [00:35<00:22, 6844.56it/s] 61%|██████    | 243636/400000 [00:35<00:22, 6948.12it/s] 61%|██████    | 244333/400000 [00:35<00:22, 6920.94it/s] 61%|██████▏   | 245070/400000 [00:35<00:21, 7049.61it/s] 61%|██████▏   | 245819/400000 [00:35<00:21, 7175.14it/s] 62%|██████▏   | 246558/400000 [00:35<00:21, 7236.10it/s] 62%|██████▏   | 247310/400000 [00:35<00:20, 7318.52it/s] 62%|██████▏   | 248043/400000 [00:35<00:21, 7150.13it/s] 62%|██████▏   | 248760/400000 [00:36<00:21, 6971.21it/s] 62%|██████▏   | 249460/400000 [00:36<00:22, 6766.39it/s] 63%|██████▎   | 250175/400000 [00:36<00:21, 6875.99it/s] 63%|██████▎   | 250920/400000 [00:36<00:21, 7038.15it/s] 63%|██████▎   | 251627/400000 [00:36<00:22, 6556.24it/s] 63%|██████▎   | 252320/400000 [00:36<00:22, 6663.86it/s] 63%|██████▎   | 253065/400000 [00:36<00:21, 6881.37it/s] 63%|██████▎   | 253777/400000 [00:36<00:21, 6948.95it/s] 64%|██████▎   | 254477/400000 [00:36<00:21, 6816.83it/s] 64%|██████▍   | 255163/400000 [00:37<00:21, 6681.52it/s] 64%|██████▍   | 255835/400000 [00:37<00:21, 6692.67it/s] 64%|██████▍   | 256574/400000 [00:37<00:20, 6886.83it/s] 64%|██████▍   | 257311/400000 [00:37<00:20, 7023.53it/s] 65%|██████▍   | 258040/400000 [00:37<00:19, 7099.96it/s] 65%|██████▍   | 258753/400000 [00:37<00:20, 6967.24it/s] 65%|██████▍   | 259498/400000 [00:37<00:19, 7103.27it/s] 65%|██████▌   | 260225/400000 [00:37<00:19, 7152.31it/s] 65%|██████▌   | 260953/400000 [00:37<00:19, 7188.91it/s] 65%|██████▌   | 261687/400000 [00:37<00:19, 7232.65it/s] 66%|██████▌   | 262412/400000 [00:38<00:19, 7235.61it/s] 66%|██████▌   | 263137/400000 [00:38<00:19, 7128.78it/s] 66%|██████▌   | 263851/400000 [00:38<00:19, 6869.59it/s] 66%|██████▌   | 264541/400000 [00:38<00:19, 6836.65it/s] 66%|██████▋   | 265227/400000 [00:38<00:20, 6717.77it/s] 66%|██████▋   | 265902/400000 [00:38<00:19, 6726.84it/s] 67%|██████▋   | 266659/400000 [00:38<00:19, 6957.27it/s] 67%|██████▋   | 267377/400000 [00:38<00:18, 7022.14it/s] 67%|██████▋   | 268110/400000 [00:38<00:18, 7111.54it/s] 67%|██████▋   | 268823/400000 [00:38<00:18, 7101.14it/s] 67%|██████▋   | 269535/400000 [00:39<00:18, 7055.67it/s] 68%|██████▊   | 270251/400000 [00:39<00:18, 7085.16it/s] 68%|██████▊   | 270985/400000 [00:39<00:18, 7159.32it/s] 68%|██████▊   | 271724/400000 [00:39<00:17, 7224.66it/s] 68%|██████▊   | 272465/400000 [00:39<00:17, 7278.15it/s] 68%|██████▊   | 273194/400000 [00:39<00:17, 7247.12it/s] 68%|██████▊   | 273920/400000 [00:39<00:17, 7182.68it/s] 69%|██████▊   | 274639/400000 [00:39<00:17, 6967.98it/s] 69%|██████▉   | 275338/400000 [00:39<00:18, 6867.60it/s] 69%|██████▉   | 276027/400000 [00:39<00:18, 6718.17it/s] 69%|██████▉   | 276730/400000 [00:40<00:18, 6807.33it/s] 69%|██████▉   | 277438/400000 [00:40<00:17, 6886.69it/s] 70%|██████▉   | 278179/400000 [00:40<00:17, 7035.60it/s] 70%|██████▉   | 278928/400000 [00:40<00:16, 7163.66it/s] 70%|██████▉   | 279651/400000 [00:40<00:16, 7181.07it/s] 70%|███████   | 280385/400000 [00:40<00:16, 7226.13it/s] 70%|███████   | 281136/400000 [00:40<00:16, 7307.08it/s] 70%|███████   | 281877/400000 [00:40<00:16, 7336.42it/s] 71%|███████   | 282612/400000 [00:40<00:16, 7045.63it/s] 71%|███████   | 283320/400000 [00:40<00:16, 6869.53it/s] 71%|███████   | 284010/400000 [00:41<00:17, 6800.11it/s] 71%|███████   | 284693/400000 [00:41<00:17, 6714.93it/s] 71%|███████▏  | 285369/400000 [00:41<00:17, 6726.54it/s] 72%|███████▏  | 286123/400000 [00:41<00:16, 6949.06it/s] 72%|███████▏  | 286821/400000 [00:41<00:16, 6909.50it/s] 72%|███████▏  | 287561/400000 [00:41<00:15, 7047.79it/s] 72%|███████▏  | 288305/400000 [00:41<00:15, 7158.38it/s] 72%|███████▏  | 289035/400000 [00:41<00:15, 7199.67it/s] 72%|███████▏  | 289781/400000 [00:41<00:15, 7274.63it/s] 73%|███████▎  | 290518/400000 [00:42<00:14, 7301.56it/s] 73%|███████▎  | 291249/400000 [00:42<00:14, 7260.97it/s] 73%|███████▎  | 291993/400000 [00:42<00:14, 7311.60it/s] 73%|███████▎  | 292740/400000 [00:42<00:14, 7356.32it/s] 73%|███████▎  | 293477/400000 [00:42<00:14, 7208.18it/s] 74%|███████▎  | 294199/400000 [00:42<00:14, 7172.30it/s] 74%|███████▎  | 294946/400000 [00:42<00:14, 7258.86it/s] 74%|███████▍  | 295685/400000 [00:42<00:14, 7295.26it/s] 74%|███████▍  | 296419/400000 [00:42<00:14, 7308.60it/s] 74%|███████▍  | 297155/400000 [00:42<00:14, 7322.66it/s] 74%|███████▍  | 297888/400000 [00:43<00:14, 7255.64it/s] 75%|███████▍  | 298614/400000 [00:43<00:13, 7246.60it/s] 75%|███████▍  | 299364/400000 [00:43<00:13, 7318.99it/s] 75%|███████▌  | 300101/400000 [00:43<00:13, 7331.85it/s] 75%|███████▌  | 300844/400000 [00:43<00:13, 7358.86it/s] 75%|███████▌  | 301581/400000 [00:43<00:13, 7286.17it/s] 76%|███████▌  | 302321/400000 [00:43<00:13, 7318.87it/s] 76%|███████▌  | 303054/400000 [00:43<00:13, 7150.31it/s] 76%|███████▌  | 303801/400000 [00:43<00:13, 7240.81it/s] 76%|███████▌  | 304527/400000 [00:43<00:13, 7172.05it/s] 76%|███████▋  | 305246/400000 [00:44<00:13, 7013.93it/s] 76%|███████▋  | 305980/400000 [00:44<00:13, 7106.57it/s] 77%|███████▋  | 306727/400000 [00:44<00:12, 7210.82it/s] 77%|███████▋  | 307475/400000 [00:44<00:12, 7289.06it/s] 77%|███████▋  | 308205/400000 [00:44<00:12, 7256.70it/s] 77%|███████▋  | 308932/400000 [00:44<00:12, 7243.11it/s] 77%|███████▋  | 309657/400000 [00:44<00:12, 7193.02it/s] 78%|███████▊  | 310386/400000 [00:44<00:12, 7220.02it/s] 78%|███████▊  | 311109/400000 [00:44<00:12, 7220.11it/s] 78%|███████▊  | 311846/400000 [00:44<00:12, 7264.00it/s] 78%|███████▊  | 312580/400000 [00:45<00:12, 7283.93it/s] 78%|███████▊  | 313309/400000 [00:45<00:11, 7241.16it/s] 79%|███████▊  | 314044/400000 [00:45<00:11, 7273.01it/s] 79%|███████▊  | 314776/400000 [00:45<00:11, 7282.19it/s] 79%|███████▉  | 315511/400000 [00:45<00:11, 7300.41it/s] 79%|███████▉  | 316242/400000 [00:45<00:11, 7187.97it/s] 79%|███████▉  | 316981/400000 [00:45<00:11, 7245.10it/s] 79%|███████▉  | 317712/400000 [00:45<00:11, 7263.08it/s] 80%|███████▉  | 318459/400000 [00:45<00:11, 7323.39it/s] 80%|███████▉  | 319195/400000 [00:45<00:11, 7334.23it/s] 80%|███████▉  | 319929/400000 [00:46<00:11, 7201.07it/s] 80%|████████  | 320650/400000 [00:46<00:11, 6883.88it/s] 80%|████████  | 321342/400000 [00:46<00:12, 6526.60it/s] 81%|████████  | 322010/400000 [00:46<00:11, 6571.27it/s] 81%|████████  | 322730/400000 [00:46<00:11, 6747.92it/s] 81%|████████  | 323423/400000 [00:46<00:11, 6800.36it/s] 81%|████████  | 324171/400000 [00:46<00:10, 6989.01it/s] 81%|████████  | 324909/400000 [00:46<00:10, 7101.01it/s] 81%|████████▏ | 325631/400000 [00:46<00:10, 7134.70it/s] 82%|████████▏ | 326347/400000 [00:47<00:10, 7098.69it/s] 82%|████████▏ | 327059/400000 [00:47<00:10, 6815.15it/s] 82%|████████▏ | 327752/400000 [00:47<00:10, 6847.96it/s] 82%|████████▏ | 328473/400000 [00:47<00:10, 6950.91it/s] 82%|████████▏ | 329213/400000 [00:47<00:10, 7077.77it/s] 82%|████████▏ | 329940/400000 [00:47<00:09, 7132.95it/s] 83%|████████▎ | 330676/400000 [00:47<00:09, 7199.46it/s] 83%|████████▎ | 331410/400000 [00:47<00:09, 7238.60it/s] 83%|████████▎ | 332159/400000 [00:47<00:09, 7309.55it/s] 83%|████████▎ | 332905/400000 [00:47<00:09, 7351.64it/s] 83%|████████▎ | 333653/400000 [00:48<00:08, 7388.65it/s] 84%|████████▎ | 334393/400000 [00:48<00:08, 7373.99it/s] 84%|████████▍ | 335134/400000 [00:48<00:08, 7384.05it/s] 84%|████████▍ | 335873/400000 [00:48<00:08, 7320.03it/s] 84%|████████▍ | 336612/400000 [00:48<00:08, 7338.94it/s] 84%|████████▍ | 337347/400000 [00:48<00:08, 7214.41it/s] 85%|████████▍ | 338070/400000 [00:48<00:08, 6956.39it/s] 85%|████████▍ | 338813/400000 [00:48<00:08, 7090.46it/s] 85%|████████▍ | 339525/400000 [00:48<00:08, 7089.38it/s] 85%|████████▌ | 340236/400000 [00:48<00:08, 6888.49it/s] 85%|████████▌ | 340981/400000 [00:49<00:08, 7046.09it/s] 85%|████████▌ | 341706/400000 [00:49<00:08, 7102.65it/s] 86%|████████▌ | 342419/400000 [00:49<00:08, 7064.95it/s] 86%|████████▌ | 343127/400000 [00:49<00:08, 6902.26it/s] 86%|████████▌ | 343835/400000 [00:49<00:08, 6953.79it/s] 86%|████████▌ | 344562/400000 [00:49<00:07, 7043.20it/s] 86%|████████▋ | 345278/400000 [00:49<00:07, 7075.76it/s] 87%|████████▋ | 346004/400000 [00:49<00:07, 7128.54it/s] 87%|████████▋ | 346751/400000 [00:49<00:07, 7225.78it/s] 87%|████████▋ | 347493/400000 [00:49<00:07, 7280.45it/s] 87%|████████▋ | 348222/400000 [00:50<00:07, 7244.47it/s] 87%|████████▋ | 348947/400000 [00:50<00:07, 7186.33it/s] 87%|████████▋ | 349667/400000 [00:50<00:07, 7065.24it/s] 88%|████████▊ | 350407/400000 [00:50<00:06, 7160.39it/s] 88%|████████▊ | 351146/400000 [00:50<00:06, 7226.82it/s] 88%|████████▊ | 351870/400000 [00:50<00:06, 7163.18it/s] 88%|████████▊ | 352604/400000 [00:50<00:06, 7212.91it/s] 88%|████████▊ | 353340/400000 [00:50<00:06, 7255.66it/s] 89%|████████▊ | 354092/400000 [00:50<00:06, 7331.43it/s] 89%|████████▊ | 354826/400000 [00:50<00:06, 7273.19it/s] 89%|████████▉ | 355564/400000 [00:51<00:06, 7303.50it/s] 89%|████████▉ | 356295/400000 [00:51<00:05, 7294.66it/s] 89%|████████▉ | 357026/400000 [00:51<00:05, 7298.72it/s] 89%|████████▉ | 357759/400000 [00:51<00:05, 7307.46it/s] 90%|████████▉ | 358490/400000 [00:51<00:05, 7287.11it/s] 90%|████████▉ | 359242/400000 [00:51<00:05, 7355.39it/s] 90%|████████▉ | 359978/400000 [00:51<00:05, 7331.10it/s] 90%|█████████ | 360715/400000 [00:51<00:05, 7342.02it/s] 90%|█████████ | 361468/400000 [00:51<00:05, 7396.58it/s] 91%|█████████ | 362211/400000 [00:51<00:05, 7405.23it/s] 91%|█████████ | 362952/400000 [00:52<00:05, 7182.00it/s] 91%|█████████ | 363673/400000 [00:52<00:05, 7190.34it/s] 91%|█████████ | 364416/400000 [00:52<00:04, 7259.28it/s] 91%|█████████▏| 365157/400000 [00:52<00:04, 7302.37it/s] 91%|█████████▏| 365888/400000 [00:52<00:04, 7257.84it/s] 92%|█████████▏| 366637/400000 [00:52<00:04, 7324.52it/s] 92%|█████████▏| 367370/400000 [00:52<00:04, 7298.65it/s] 92%|█████████▏| 368112/400000 [00:52<00:04, 7334.28it/s] 92%|█████████▏| 368852/400000 [00:52<00:04, 7352.86it/s] 92%|█████████▏| 369600/400000 [00:52<00:04, 7389.87it/s] 93%|█████████▎| 370340/400000 [00:53<00:04, 7213.04it/s] 93%|█████████▎| 371064/400000 [00:53<00:04, 7219.78it/s] 93%|█████████▎| 371798/400000 [00:53<00:03, 7253.84it/s] 93%|█████████▎| 372524/400000 [00:53<00:03, 7044.38it/s] 93%|█████████▎| 373231/400000 [00:53<00:03, 6921.04it/s] 93%|█████████▎| 373925/400000 [00:53<00:03, 6812.58it/s] 94%|█████████▎| 374608/400000 [00:53<00:03, 6797.16it/s] 94%|█████████▍| 375351/400000 [00:53<00:03, 6973.58it/s] 94%|█████████▍| 376092/400000 [00:53<00:03, 7097.85it/s] 94%|█████████▍| 376837/400000 [00:54<00:03, 7199.77it/s] 94%|█████████▍| 377583/400000 [00:54<00:03, 7274.65it/s] 95%|█████████▍| 378312/400000 [00:54<00:03, 7175.83it/s] 95%|█████████▍| 379031/400000 [00:54<00:02, 7175.32it/s] 95%|█████████▍| 379766/400000 [00:54<00:02, 7226.78it/s] 95%|█████████▌| 380490/400000 [00:54<00:02, 7080.18it/s] 95%|█████████▌| 381220/400000 [00:54<00:02, 7142.57it/s] 95%|█████████▌| 381954/400000 [00:54<00:02, 7198.81it/s] 96%|█████████▌| 382680/400000 [00:54<00:02, 7212.94it/s] 96%|█████████▌| 383402/400000 [00:54<00:02, 7073.61it/s] 96%|█████████▌| 384145/400000 [00:55<00:02, 7174.69it/s] 96%|█████████▌| 384884/400000 [00:55<00:02, 7235.57it/s] 96%|█████████▋| 385609/400000 [00:55<00:01, 7236.09it/s] 97%|█████████▋| 386353/400000 [00:55<00:01, 7293.58it/s] 97%|█████████▋| 387083/400000 [00:55<00:01, 7270.14it/s] 97%|█████████▋| 387811/400000 [00:55<00:01, 7259.84it/s] 97%|█████████▋| 388556/400000 [00:55<00:01, 7314.53it/s] 97%|█████████▋| 389290/400000 [00:55<00:01, 7319.45it/s] 98%|█████████▊| 390038/400000 [00:55<00:01, 7366.49it/s] 98%|█████████▊| 390775/400000 [00:55<00:01, 7366.95it/s] 98%|█████████▊| 391518/400000 [00:56<00:01, 7385.35it/s] 98%|█████████▊| 392257/400000 [00:56<00:01, 7267.41it/s] 98%|█████████▊| 392985/400000 [00:56<00:01, 6984.02it/s] 98%|█████████▊| 393687/400000 [00:56<00:00, 6894.42it/s] 99%|█████████▊| 394379/400000 [00:56<00:00, 6698.41it/s] 99%|█████████▉| 395122/400000 [00:56<00:00, 6900.70it/s] 99%|█████████▉| 395845/400000 [00:56<00:00, 6990.30it/s] 99%|█████████▉| 396573/400000 [00:56<00:00, 7072.47it/s] 99%|█████████▉| 397328/400000 [00:56<00:00, 7208.87it/s]100%|█████████▉| 398068/400000 [00:56<00:00, 7263.59it/s]100%|█████████▉| 398813/400000 [00:57<00:00, 7316.79it/s]100%|█████████▉| 399546/400000 [00:57<00:00, 7283.17it/s]100%|█████████▉| 399999/400000 [00:57<00:00, 6986.91it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f975d8e8c50>, <torchtext.data.dataset.TabularDataset object at 0x7f975d8e8da0>, <torchtext.vocab.Vocab object at 0x7f975d8e8cc0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.35 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 24.12 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.65 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.65 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.99 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.99 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.36 file/s]2020-07-18 00:19:13.602944: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-18 00:19:13.607816: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-07-18 00:19:13.608521: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556ba8672d90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-18 00:19:13.608746: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3547136/9912422 [00:00<00:00, 35217733.45it/s]9920512it [00:00, 35013457.84it/s]                             
0it [00:00, ?it/s]32768it [00:00, 663116.99it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160620.70it/s]1654784it [00:00, 10847673.84it/s]                         
0it [00:00, ?it/s]8192it [00:00, 232009.90it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
