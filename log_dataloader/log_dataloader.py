
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fce122f90d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fce122f90d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fce7d6431e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fce7d6431e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fce9b98ae18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fce9b98ae18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fce2a971620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fce2a971620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fce2a971620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:28, 111765.16it/s] 75%|███████▍  | 7413760/9912422 [00:00<00:15, 159561.16it/s]9920512it [00:00, 36783760.87it/s]                           
0it [00:00, ?it/s]32768it [00:00, 563459.14it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 465557.03it/s]1654784it [00:00, 11715293.66it/s]                         
0it [00:00, ?it/s]8192it [00:00, 199049.57it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fce119b0978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fce119a4c18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fce2a971268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fce2a971268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fce2a971268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:32,  1.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:32,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:31,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:31,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:30,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:29,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:29,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:02,  2.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:02,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:02,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:02,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:01,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:01,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:00,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:43,  3.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:43,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:42,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:42,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:42,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:42,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:41,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:29,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:29,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:28,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:20,  6.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:20,  6.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:20,  6.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:20,  6.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:20,  6.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:20,  6.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:20,  6.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:14,  8.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:14,  8.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:14,  8.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:14,  8.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:14,  8.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:14,  8.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:14,  8.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 11.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 28.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 28.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 28.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 33.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 33.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 33.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 33.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 33.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 33.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 33.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 37.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 37.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 37.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 37.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 37.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 37.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 37.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 40.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 40.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 40.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 40.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 40.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 40.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 40.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 43.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 43.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 43.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 43.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 43.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 43.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 43.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 46.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 46.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 46.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 46.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 46.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 46.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 46.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 48.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 49.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 49.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 49.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 49.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 49.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 49.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 49.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 50.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 50.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 50.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 50.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 50.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 50.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 50.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 51.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 51.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 51.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 51.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 51.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 51.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 51.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 51.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 51.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 51.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 51.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 51.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 51.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 51.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 51.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 51.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 51.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 51.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 51.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 51.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 51.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 52.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 52.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 52.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 52.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 52.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 52.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 52.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 52.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 52.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 52.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 52.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 52.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 52.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 52.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 50.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 50.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 50.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 50.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 50.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 50.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 50.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 50.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 52.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 52.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 52.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 52.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 52.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 52.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 52.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 52.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.62s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 52.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 52.48 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.70s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 52.48 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.70s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.70s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 28.44 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.70s/ url]
0 examples [00:00, ? examples/s]2020-07-07 18:09:16.601157: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-07 18:09:16.615212: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-07 18:09:16.615394: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55679719ee40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-07 18:09:16.615412: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
30 examples [00:00, 296.85 examples/s]125 examples [00:00, 373.61 examples/s]217 examples [00:00, 454.59 examples/s]315 examples [00:00, 540.81 examples/s]409 examples [00:00, 619.31 examples/s]505 examples [00:00, 692.54 examples/s]603 examples [00:00, 758.29 examples/s]689 examples [00:00, 782.05 examples/s]783 examples [00:00, 821.83 examples/s]885 examples [00:01, 871.96 examples/s]982 examples [00:01, 897.33 examples/s]1081 examples [00:01, 921.08 examples/s]1176 examples [00:01, 910.44 examples/s]1270 examples [00:01, 916.88 examples/s]1369 examples [00:01, 936.14 examples/s]1465 examples [00:01, 941.48 examples/s]1563 examples [00:01, 951.44 examples/s]1659 examples [00:01, 943.73 examples/s]1754 examples [00:01, 941.74 examples/s]1849 examples [00:02, 928.21 examples/s]1943 examples [00:02, 922.45 examples/s]2036 examples [00:02, 916.27 examples/s]2128 examples [00:02, 909.17 examples/s]2220 examples [00:02, 889.29 examples/s]2316 examples [00:02, 909.18 examples/s]2411 examples [00:02, 919.32 examples/s]2509 examples [00:02, 934.96 examples/s]2603 examples [00:02, 930.74 examples/s]2697 examples [00:02, 926.24 examples/s]2792 examples [00:03, 932.58 examples/s]2887 examples [00:03, 936.33 examples/s]2981 examples [00:03, 933.05 examples/s]3075 examples [00:03, 921.15 examples/s]3177 examples [00:03, 948.17 examples/s]3273 examples [00:03, 943.40 examples/s]3368 examples [00:03, 939.83 examples/s]3463 examples [00:03, 934.52 examples/s]3557 examples [00:03, 932.21 examples/s]3654 examples [00:03, 942.57 examples/s]3749 examples [00:04, 943.44 examples/s]3844 examples [00:04, 941.77 examples/s]3939 examples [00:04, 939.57 examples/s]4033 examples [00:04, 923.92 examples/s]4126 examples [00:04, 896.74 examples/s]4222 examples [00:04, 911.76 examples/s]4323 examples [00:04, 937.27 examples/s]4425 examples [00:04, 958.37 examples/s]4522 examples [00:04, 958.87 examples/s]4624 examples [00:04, 975.44 examples/s]4726 examples [00:05, 987.17 examples/s]4827 examples [00:05, 993.17 examples/s]4930 examples [00:05, 1002.28 examples/s]5031 examples [00:05, 994.48 examples/s] 5131 examples [00:05, 979.32 examples/s]5230 examples [00:05, 917.30 examples/s]5329 examples [00:05, 935.93 examples/s]5430 examples [00:05, 955.82 examples/s]5527 examples [00:05, 958.30 examples/s]5624 examples [00:06, 961.27 examples/s]5721 examples [00:06, 954.81 examples/s]5818 examples [00:06, 959.15 examples/s]5917 examples [00:06, 966.83 examples/s]6014 examples [00:06, 958.96 examples/s]6112 examples [00:06, 963.04 examples/s]6210 examples [00:06, 967.82 examples/s]6307 examples [00:06, 965.16 examples/s]6404 examples [00:06, 945.61 examples/s]6499 examples [00:06, 929.82 examples/s]6596 examples [00:07, 940.97 examples/s]6691 examples [00:07, 916.60 examples/s]6783 examples [00:07, 893.48 examples/s]6873 examples [00:07, 890.98 examples/s]6963 examples [00:07, 893.38 examples/s]7057 examples [00:07, 904.57 examples/s]7156 examples [00:07, 926.46 examples/s]7255 examples [00:07, 944.44 examples/s]7351 examples [00:07, 949.00 examples/s]7447 examples [00:07, 950.68 examples/s]7546 examples [00:08, 945.99 examples/s]7641 examples [00:08, 902.46 examples/s]7732 examples [00:08, 891.56 examples/s]7824 examples [00:08, 898.25 examples/s]7915 examples [00:08, 891.47 examples/s]8009 examples [00:08, 903.02 examples/s]8100 examples [00:08, 886.98 examples/s]8189 examples [00:08, 881.96 examples/s]8283 examples [00:08, 897.48 examples/s]8383 examples [00:09, 923.96 examples/s]8479 examples [00:09, 934.03 examples/s]8577 examples [00:09, 944.89 examples/s]8672 examples [00:09, 943.68 examples/s]8767 examples [00:09, 928.81 examples/s]8861 examples [00:09, 917.65 examples/s]8962 examples [00:09, 941.57 examples/s]9059 examples [00:09, 948.65 examples/s]9156 examples [00:09, 953.84 examples/s]9252 examples [00:09, 951.39 examples/s]9349 examples [00:10, 955.31 examples/s]9448 examples [00:10, 963.79 examples/s]9548 examples [00:10, 971.62 examples/s]9648 examples [00:10, 979.19 examples/s]9746 examples [00:10, 966.00 examples/s]9843 examples [00:10, 942.62 examples/s]9938 examples [00:10, 941.97 examples/s]10033 examples [00:10, 897.38 examples/s]10136 examples [00:10, 930.38 examples/s]10233 examples [00:10, 941.78 examples/s]10331 examples [00:11, 952.80 examples/s]10427 examples [00:11, 920.62 examples/s]10531 examples [00:11, 951.76 examples/s]10631 examples [00:11, 965.38 examples/s]10729 examples [00:11, 953.71 examples/s]10825 examples [00:11, 948.23 examples/s]10923 examples [00:11, 956.29 examples/s]11026 examples [00:11, 975.28 examples/s]11129 examples [00:11, 989.24 examples/s]11229 examples [00:12, 952.31 examples/s]11325 examples [00:12, 947.26 examples/s]11428 examples [00:12, 969.31 examples/s]11526 examples [00:12, 955.43 examples/s]11622 examples [00:12, 948.33 examples/s]11718 examples [00:12, 917.84 examples/s]11814 examples [00:12, 927.72 examples/s]11914 examples [00:12, 946.39 examples/s]12013 examples [00:12, 958.66 examples/s]12112 examples [00:12, 967.06 examples/s]12209 examples [00:13, 954.97 examples/s]12305 examples [00:13, 941.34 examples/s]12403 examples [00:13, 951.82 examples/s]12506 examples [00:13, 971.57 examples/s]12604 examples [00:13, 953.96 examples/s]12700 examples [00:13, 928.50 examples/s]12794 examples [00:13, 907.31 examples/s]12892 examples [00:13, 927.16 examples/s]12994 examples [00:13, 951.67 examples/s]13095 examples [00:13, 965.92 examples/s]13192 examples [00:14, 958.67 examples/s]13290 examples [00:14, 964.27 examples/s]13387 examples [00:14, 949.50 examples/s]13483 examples [00:14, 938.34 examples/s]13577 examples [00:14, 913.89 examples/s]13669 examples [00:14, 908.74 examples/s]13761 examples [00:14, 894.38 examples/s]13851 examples [00:14, 866.43 examples/s]13939 examples [00:14, 869.37 examples/s]14027 examples [00:15, 863.06 examples/s]14116 examples [00:15, 870.96 examples/s]14204 examples [00:15, 848.39 examples/s]14300 examples [00:15, 877.41 examples/s]14394 examples [00:15, 895.13 examples/s]14484 examples [00:15, 878.71 examples/s]14573 examples [00:15, 872.23 examples/s]14667 examples [00:15, 890.78 examples/s]14757 examples [00:15, 893.08 examples/s]14855 examples [00:15, 914.99 examples/s]14947 examples [00:16, 882.80 examples/s]15036 examples [00:16, 851.63 examples/s]15129 examples [00:16, 871.61 examples/s]15226 examples [00:16, 896.58 examples/s]15321 examples [00:16, 911.36 examples/s]15413 examples [00:16, 911.76 examples/s]15507 examples [00:16, 918.08 examples/s]15600 examples [00:16, 903.12 examples/s]15693 examples [00:16, 910.17 examples/s]15790 examples [00:16, 927.28 examples/s]15883 examples [00:17, 913.59 examples/s]15981 examples [00:17, 931.64 examples/s]16078 examples [00:17, 942.68 examples/s]16174 examples [00:17, 946.78 examples/s]16269 examples [00:17, 945.87 examples/s]16364 examples [00:17, 939.66 examples/s]16459 examples [00:17, 933.52 examples/s]16553 examples [00:17, 901.03 examples/s]16644 examples [00:17, 895.93 examples/s]16734 examples [00:18, 891.59 examples/s]16824 examples [00:18, 877.62 examples/s]16913 examples [00:18, 880.97 examples/s]17006 examples [00:18, 894.63 examples/s]17096 examples [00:18, 878.19 examples/s]17196 examples [00:18, 908.96 examples/s]17288 examples [00:18, 888.82 examples/s]17378 examples [00:18, 890.79 examples/s]17473 examples [00:18, 906.70 examples/s]17567 examples [00:18, 914.26 examples/s]17670 examples [00:19, 945.46 examples/s]17765 examples [00:19, 931.64 examples/s]17863 examples [00:19, 943.97 examples/s]17963 examples [00:19, 959.25 examples/s]18064 examples [00:19, 971.19 examples/s]18162 examples [00:19, 939.00 examples/s]18257 examples [00:19, 913.08 examples/s]18349 examples [00:19, 890.01 examples/s]18445 examples [00:19, 908.37 examples/s]18537 examples [00:19, 887.67 examples/s]18627 examples [00:20, 872.01 examples/s]18720 examples [00:20, 886.58 examples/s]18819 examples [00:20, 912.83 examples/s]18918 examples [00:20, 931.82 examples/s]19016 examples [00:20, 945.25 examples/s]19112 examples [00:20, 947.06 examples/s]19207 examples [00:20, 944.04 examples/s]19302 examples [00:20, 937.50 examples/s]19396 examples [00:20, 902.83 examples/s]19490 examples [00:21, 911.21 examples/s]19584 examples [00:21, 907.95 examples/s]19677 examples [00:21, 913.69 examples/s]19774 examples [00:21, 928.30 examples/s]19868 examples [00:21, 930.91 examples/s]19963 examples [00:21, 933.55 examples/s]20057 examples [00:21, 871.22 examples/s]20154 examples [00:21, 895.79 examples/s]20245 examples [00:21, 895.05 examples/s]20338 examples [00:21, 904.61 examples/s]20431 examples [00:22, 911.80 examples/s]20523 examples [00:22, 902.21 examples/s]20617 examples [00:22, 911.80 examples/s]20711 examples [00:22, 917.79 examples/s]20804 examples [00:22, 919.75 examples/s]20901 examples [00:22, 931.49 examples/s]20995 examples [00:22, 915.01 examples/s]21088 examples [00:22, 918.48 examples/s]21190 examples [00:22, 946.24 examples/s]21289 examples [00:22, 958.47 examples/s]21386 examples [00:23, 950.70 examples/s]21482 examples [00:23, 919.14 examples/s]21575 examples [00:23, 880.69 examples/s]21668 examples [00:23, 894.33 examples/s]21758 examples [00:23, 890.30 examples/s]21851 examples [00:23, 899.50 examples/s]21942 examples [00:23, 894.95 examples/s]22032 examples [00:23, 886.18 examples/s]22121 examples [00:23, 859.95 examples/s]22208 examples [00:24, 853.91 examples/s]22301 examples [00:24, 873.81 examples/s]22389 examples [00:24, 874.92 examples/s]22477 examples [00:24, 870.85 examples/s]22569 examples [00:24, 882.63 examples/s]22662 examples [00:24, 896.10 examples/s]22758 examples [00:24, 911.72 examples/s]22850 examples [00:24, 895.79 examples/s]22941 examples [00:24, 897.75 examples/s]23035 examples [00:24, 907.68 examples/s]23126 examples [00:25, 907.05 examples/s]23224 examples [00:25, 925.57 examples/s]23317 examples [00:25, 919.46 examples/s]23410 examples [00:25, 914.53 examples/s]23502 examples [00:25, 911.30 examples/s]23596 examples [00:25, 917.41 examples/s]23692 examples [00:25, 928.81 examples/s]23786 examples [00:25, 931.25 examples/s]23882 examples [00:25, 939.37 examples/s]23976 examples [00:25, 928.28 examples/s]24071 examples [00:26, 933.31 examples/s]24165 examples [00:26, 929.55 examples/s]24258 examples [00:26, 922.57 examples/s]24354 examples [00:26, 931.74 examples/s]24448 examples [00:26, 898.87 examples/s]24539 examples [00:26, 892.01 examples/s]24629 examples [00:26, 881.08 examples/s]24718 examples [00:26, 876.44 examples/s]24807 examples [00:26, 880.29 examples/s]24896 examples [00:26, 877.19 examples/s]24987 examples [00:27, 883.45 examples/s]25078 examples [00:27, 888.52 examples/s]25167 examples [00:27, 878.76 examples/s]25255 examples [00:27, 863.11 examples/s]25343 examples [00:27, 866.00 examples/s]25437 examples [00:27, 885.24 examples/s]25527 examples [00:27, 888.54 examples/s]25616 examples [00:27, 872.64 examples/s]25706 examples [00:27, 878.70 examples/s]25796 examples [00:28, 883.23 examples/s]25886 examples [00:28, 885.27 examples/s]25980 examples [00:28, 898.62 examples/s]26070 examples [00:28, 868.62 examples/s]26158 examples [00:28, 863.67 examples/s]26251 examples [00:28, 881.81 examples/s]26340 examples [00:28, 881.14 examples/s]26434 examples [00:28, 896.84 examples/s]26529 examples [00:28, 911.88 examples/s]26625 examples [00:28, 923.07 examples/s]26718 examples [00:29, 915.58 examples/s]26810 examples [00:29, 913.39 examples/s]26906 examples [00:29, 925.17 examples/s]26999 examples [00:29, 873.71 examples/s]27088 examples [00:29, 863.83 examples/s]27175 examples [00:29, 817.98 examples/s]27258 examples [00:29, 785.59 examples/s]27344 examples [00:29, 806.38 examples/s]27427 examples [00:29, 811.47 examples/s]27509 examples [00:29, 808.08 examples/s]27595 examples [00:30, 822.34 examples/s]27687 examples [00:30, 848.45 examples/s]27783 examples [00:30, 877.58 examples/s]27875 examples [00:30, 887.56 examples/s]27966 examples [00:30, 891.79 examples/s]28056 examples [00:30, 869.92 examples/s]28146 examples [00:30, 876.24 examples/s]28235 examples [00:30, 878.76 examples/s]28324 examples [00:30, 858.98 examples/s]28414 examples [00:31, 868.24 examples/s]28504 examples [00:31, 875.24 examples/s]28592 examples [00:31, 871.77 examples/s]28680 examples [00:31, 847.71 examples/s]28766 examples [00:31, 851.11 examples/s]28857 examples [00:31, 867.59 examples/s]28946 examples [00:31, 870.98 examples/s]29035 examples [00:31, 875.86 examples/s]29123 examples [00:31, 872.29 examples/s]29211 examples [00:31, 868.00 examples/s]29306 examples [00:32, 888.67 examples/s]29399 examples [00:32, 898.35 examples/s]29489 examples [00:32, 888.64 examples/s]29585 examples [00:32, 907.92 examples/s]29676 examples [00:32, 893.29 examples/s]29770 examples [00:32, 905.08 examples/s]29865 examples [00:32, 917.60 examples/s]29961 examples [00:32, 928.29 examples/s]30054 examples [00:32, 842.85 examples/s]30144 examples [00:32, 859.09 examples/s]30236 examples [00:33, 874.17 examples/s]30332 examples [00:33, 897.10 examples/s]30429 examples [00:33, 915.81 examples/s]30525 examples [00:33, 928.02 examples/s]30619 examples [00:33, 920.17 examples/s]30712 examples [00:33, 907.91 examples/s]30805 examples [00:33, 913.60 examples/s]30902 examples [00:33, 927.51 examples/s]30995 examples [00:33, 914.70 examples/s]31087 examples [00:34, 899.97 examples/s]31178 examples [00:34, 893.49 examples/s]31268 examples [00:34, 889.71 examples/s]31360 examples [00:34, 896.10 examples/s]31451 examples [00:34, 896.96 examples/s]31541 examples [00:34, 863.54 examples/s]31639 examples [00:34, 894.84 examples/s]31732 examples [00:34, 905.10 examples/s]31830 examples [00:34, 925.93 examples/s]31926 examples [00:34, 932.88 examples/s]32021 examples [00:35, 935.31 examples/s]32115 examples [00:35, 927.70 examples/s]32208 examples [00:35, 925.79 examples/s]32301 examples [00:35, 912.08 examples/s]32393 examples [00:35, 869.86 examples/s]32481 examples [00:35, 859.97 examples/s]32571 examples [00:35, 870.84 examples/s]32665 examples [00:35, 889.30 examples/s]32757 examples [00:35, 896.72 examples/s]32847 examples [00:35, 879.96 examples/s]32936 examples [00:36, 871.89 examples/s]33028 examples [00:36, 883.98 examples/s]33122 examples [00:36, 897.50 examples/s]33219 examples [00:36, 917.75 examples/s]33312 examples [00:36, 917.85 examples/s]33405 examples [00:36, 920.51 examples/s]33498 examples [00:36, 915.35 examples/s]33590 examples [00:36, 908.24 examples/s]33687 examples [00:36, 924.74 examples/s]33780 examples [00:36, 922.30 examples/s]33873 examples [00:37, 920.02 examples/s]33968 examples [00:37, 927.14 examples/s]34062 examples [00:37, 929.33 examples/s]34155 examples [00:37, 911.82 examples/s]34247 examples [00:37, 884.91 examples/s]34338 examples [00:37, 887.36 examples/s]34427 examples [00:37, 883.99 examples/s]34516 examples [00:37, 884.16 examples/s]34612 examples [00:37, 903.26 examples/s]34703 examples [00:38, 896.66 examples/s]34795 examples [00:38, 900.89 examples/s]34891 examples [00:38, 916.37 examples/s]34988 examples [00:38, 931.24 examples/s]35082 examples [00:38, 925.15 examples/s]35175 examples [00:38, 911.44 examples/s]35267 examples [00:38, 912.12 examples/s]35359 examples [00:38, 910.00 examples/s]35451 examples [00:38, 898.07 examples/s]35541 examples [00:38, 892.94 examples/s]35631 examples [00:39, 888.35 examples/s]35730 examples [00:39, 916.01 examples/s]35825 examples [00:39, 925.24 examples/s]35920 examples [00:39, 929.99 examples/s]36014 examples [00:39, 931.92 examples/s]36108 examples [00:39, 898.38 examples/s]36207 examples [00:39, 922.06 examples/s]36306 examples [00:39, 939.82 examples/s]36404 examples [00:39, 949.16 examples/s]36509 examples [00:39, 975.09 examples/s]36607 examples [00:40, 958.62 examples/s]36704 examples [00:40, 944.71 examples/s]36800 examples [00:40, 948.55 examples/s]36896 examples [00:40, 949.96 examples/s]36992 examples [00:40, 933.70 examples/s]37086 examples [00:40, 912.20 examples/s]37178 examples [00:40, 898.21 examples/s]37269 examples [00:40, 880.62 examples/s]37358 examples [00:40, 875.50 examples/s]37449 examples [00:41, 884.65 examples/s]37538 examples [00:41, 881.50 examples/s]37631 examples [00:41, 895.23 examples/s]37725 examples [00:41, 905.94 examples/s]37822 examples [00:41, 924.23 examples/s]37915 examples [00:41, 917.61 examples/s]38007 examples [00:41, 901.04 examples/s]38102 examples [00:41, 914.43 examples/s]38194 examples [00:41, 915.35 examples/s]38287 examples [00:41, 918.27 examples/s]38380 examples [00:42, 919.28 examples/s]38472 examples [00:42, 904.96 examples/s]38569 examples [00:42, 921.54 examples/s]38662 examples [00:42, 922.42 examples/s]38755 examples [00:42, 898.63 examples/s]38846 examples [00:42, 888.72 examples/s]38936 examples [00:42, 881.65 examples/s]39032 examples [00:42, 903.14 examples/s]39123 examples [00:42, 896.32 examples/s]39214 examples [00:42, 899.94 examples/s]39305 examples [00:43, 891.35 examples/s]39395 examples [00:43, 891.32 examples/s]39485 examples [00:43, 890.32 examples/s]39575 examples [00:43, 887.53 examples/s]39670 examples [00:43, 904.69 examples/s]39761 examples [00:43, 904.11 examples/s]39852 examples [00:43, 895.28 examples/s]39951 examples [00:43, 919.00 examples/s]40044 examples [00:43, 870.42 examples/s]40140 examples [00:43, 894.94 examples/s]40231 examples [00:44, 892.78 examples/s]40321 examples [00:44, 880.90 examples/s]40410 examples [00:44, 881.21 examples/s]40499 examples [00:44, 883.07 examples/s]40588 examples [00:44, 871.78 examples/s]40681 examples [00:44, 887.10 examples/s]40773 examples [00:44, 894.38 examples/s]40869 examples [00:44, 885.54 examples/s]40962 examples [00:44, 897.86 examples/s]41052 examples [00:45, 894.95 examples/s]41144 examples [00:45, 900.47 examples/s]41241 examples [00:45, 919.24 examples/s]41335 examples [00:45, 924.73 examples/s]41428 examples [00:45, 915.59 examples/s]41522 examples [00:45, 922.12 examples/s]41615 examples [00:45, 880.06 examples/s]41705 examples [00:45, 884.34 examples/s]41803 examples [00:45, 910.23 examples/s]41895 examples [00:45, 909.79 examples/s]41997 examples [00:46, 940.10 examples/s]42092 examples [00:46, 920.24 examples/s]42185 examples [00:46, 904.26 examples/s]42284 examples [00:46, 927.02 examples/s]42378 examples [00:46, 923.34 examples/s]42471 examples [00:46, 916.62 examples/s]42563 examples [00:46, 892.09 examples/s]42659 examples [00:46, 910.40 examples/s]42760 examples [00:46, 937.22 examples/s]42855 examples [00:46, 932.57 examples/s]42950 examples [00:47, 937.35 examples/s]43044 examples [00:47, 925.64 examples/s]43138 examples [00:47, 929.82 examples/s]43233 examples [00:47, 933.39 examples/s]43327 examples [00:47, 912.80 examples/s]43424 examples [00:47, 927.42 examples/s]43517 examples [00:47, 925.66 examples/s]43618 examples [00:47, 947.05 examples/s]43715 examples [00:47, 951.37 examples/s]43812 examples [00:47, 956.54 examples/s]43908 examples [00:48, 953.73 examples/s]44004 examples [00:48, 938.73 examples/s]44099 examples [00:48, 941.43 examples/s]44195 examples [00:48, 945.55 examples/s]44291 examples [00:48, 948.72 examples/s]44391 examples [00:48, 963.35 examples/s]44488 examples [00:48, 959.95 examples/s]44585 examples [00:48, 957.41 examples/s]44681 examples [00:48, 953.14 examples/s]44777 examples [00:49, 947.38 examples/s]44872 examples [00:49, 940.19 examples/s]44967 examples [00:49, 915.31 examples/s]45059 examples [00:49, 913.93 examples/s]45151 examples [00:49, 912.48 examples/s]45245 examples [00:49, 918.41 examples/s]45338 examples [00:49, 919.99 examples/s]45431 examples [00:49, 916.37 examples/s]45528 examples [00:49, 929.14 examples/s]45626 examples [00:49, 941.32 examples/s]45722 examples [00:50, 946.55 examples/s]45819 examples [00:50, 952.34 examples/s]45915 examples [00:50, 943.20 examples/s]46015 examples [00:50, 959.26 examples/s]46112 examples [00:50, 959.86 examples/s]46209 examples [00:50, 942.16 examples/s]46308 examples [00:50, 953.90 examples/s]46404 examples [00:50, 937.21 examples/s]46498 examples [00:50, 920.43 examples/s]46591 examples [00:50, 917.90 examples/s]46683 examples [00:51, 911.55 examples/s]46777 examples [00:51, 918.39 examples/s]46869 examples [00:51, 917.62 examples/s]46968 examples [00:51, 936.48 examples/s]47067 examples [00:51, 950.35 examples/s]47167 examples [00:51, 963.65 examples/s]47272 examples [00:51, 985.24 examples/s]47371 examples [00:51, 969.80 examples/s]47469 examples [00:51, 970.21 examples/s]47570 examples [00:51, 978.35 examples/s]47668 examples [00:52, 964.22 examples/s]47769 examples [00:52, 976.75 examples/s]47867 examples [00:52, 967.14 examples/s]47969 examples [00:52, 980.73 examples/s]48068 examples [00:52, 967.51 examples/s]48165 examples [00:52, 939.44 examples/s]48260 examples [00:52, 935.54 examples/s]48354 examples [00:52, 923.99 examples/s]48449 examples [00:52, 930.76 examples/s]48545 examples [00:53, 936.27 examples/s]48639 examples [00:53, 913.41 examples/s]48732 examples [00:53, 916.73 examples/s]48824 examples [00:53, 908.87 examples/s]48915 examples [00:53, 899.83 examples/s]49011 examples [00:53, 916.82 examples/s]49109 examples [00:53, 932.79 examples/s]49207 examples [00:53, 945.89 examples/s]49305 examples [00:53, 955.20 examples/s]49404 examples [00:53, 963.11 examples/s]49501 examples [00:54, 964.37 examples/s]49598 examples [00:54, 960.16 examples/s]49698 examples [00:54, 969.17 examples/s]49795 examples [00:54, 936.69 examples/s]49889 examples [00:54, 932.02 examples/s]49983 examples [00:54, 925.34 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 6977/50000 [00:00<00:00, 69766.07 examples/s] 38%|███▊      | 18756/50000 [00:00<00:00, 79487.94 examples/s] 61%|██████    | 30290/50000 [00:00<00:00, 87657.81 examples/s] 87%|████████▋ | 43424/50000 [00:00<00:00, 96815.77 examples/s]                                                               0 examples [00:00, ? examples/s]80 examples [00:00, 797.14 examples/s]176 examples [00:00, 838.31 examples/s]273 examples [00:00, 872.07 examples/s]377 examples [00:00, 916.01 examples/s]481 examples [00:00, 948.70 examples/s]583 examples [00:00, 968.01 examples/s]688 examples [00:00, 990.36 examples/s]782 examples [00:00, 972.15 examples/s]878 examples [00:00, 967.38 examples/s]973 examples [00:01, 960.25 examples/s]1073 examples [00:01, 969.83 examples/s]1169 examples [00:01, 962.98 examples/s]1265 examples [00:01, 946.94 examples/s]1361 examples [00:01, 948.20 examples/s]1456 examples [00:01, 933.62 examples/s]1551 examples [00:01, 937.93 examples/s]1645 examples [00:01, 933.30 examples/s]1739 examples [00:01, 905.00 examples/s]1839 examples [00:01, 930.02 examples/s]1936 examples [00:02, 941.11 examples/s]2037 examples [00:02, 958.05 examples/s]2134 examples [00:02, 955.70 examples/s]2230 examples [00:02, 929.38 examples/s]2324 examples [00:02, 920.27 examples/s]2417 examples [00:02, 920.83 examples/s]2519 examples [00:02, 947.22 examples/s]2615 examples [00:02, 943.14 examples/s]2710 examples [00:02, 925.26 examples/s]2808 examples [00:02, 939.74 examples/s]2903 examples [00:03, 899.40 examples/s]2998 examples [00:03, 911.61 examples/s]3092 examples [00:03, 917.67 examples/s]3185 examples [00:03, 881.62 examples/s]3281 examples [00:03, 900.76 examples/s]3378 examples [00:03, 919.77 examples/s]3474 examples [00:03, 928.15 examples/s]3568 examples [00:03, 912.84 examples/s]3660 examples [00:03, 895.79 examples/s]3757 examples [00:04, 914.27 examples/s]3855 examples [00:04, 932.09 examples/s]3953 examples [00:04, 944.44 examples/s]4048 examples [00:04, 917.28 examples/s]4144 examples [00:04, 927.20 examples/s]4241 examples [00:04, 936.81 examples/s]4338 examples [00:04, 944.38 examples/s]4434 examples [00:04, 947.42 examples/s]4530 examples [00:04, 949.14 examples/s]4625 examples [00:04, 947.60 examples/s]4721 examples [00:05, 950.72 examples/s]4823 examples [00:05, 969.69 examples/s]4924 examples [00:05, 981.11 examples/s]5028 examples [00:05, 997.70 examples/s]5128 examples [00:05, 984.90 examples/s]5228 examples [00:05, 989.02 examples/s]5328 examples [00:05, 985.35 examples/s]5427 examples [00:05, 980.22 examples/s]5526 examples [00:05, 974.81 examples/s]5624 examples [00:05, 959.76 examples/s]5727 examples [00:06, 979.62 examples/s]5826 examples [00:06, 943.38 examples/s]5921 examples [00:06, 939.78 examples/s]6021 examples [00:06, 955.85 examples/s]6117 examples [00:06, 943.46 examples/s]6212 examples [00:06, 910.13 examples/s]6309 examples [00:06, 926.08 examples/s]6406 examples [00:06, 938.33 examples/s]6501 examples [00:06, 919.57 examples/s]6594 examples [00:06, 916.47 examples/s]6686 examples [00:07, 915.87 examples/s]6779 examples [00:07, 918.20 examples/s]6873 examples [00:07, 924.18 examples/s]6969 examples [00:07, 933.76 examples/s]7063 examples [00:07, 934.31 examples/s]7157 examples [00:07, 934.65 examples/s]7255 examples [00:07, 946.79 examples/s]7354 examples [00:07, 958.02 examples/s]7451 examples [00:07, 960.41 examples/s]7548 examples [00:08, 935.43 examples/s]7642 examples [00:08, 923.90 examples/s]7740 examples [00:08, 937.77 examples/s]7843 examples [00:08, 963.15 examples/s]7940 examples [00:08, 964.09 examples/s]8037 examples [00:08, 943.53 examples/s]8134 examples [00:08, 948.40 examples/s]8232 examples [00:08, 957.24 examples/s]8331 examples [00:08, 965.98 examples/s]8428 examples [00:08, 957.43 examples/s]8524 examples [00:09, 955.08 examples/s]8622 examples [00:09, 960.80 examples/s]8719 examples [00:09, 942.01 examples/s]8814 examples [00:09, 943.25 examples/s]8913 examples [00:09, 954.97 examples/s]9013 examples [00:09, 965.63 examples/s]9115 examples [00:09, 979.48 examples/s]9214 examples [00:09, 975.90 examples/s]9313 examples [00:09, 977.33 examples/s]9412 examples [00:09, 977.89 examples/s]9510 examples [00:10, 953.33 examples/s]9606 examples [00:10, 950.42 examples/s]9705 examples [00:10, 961.48 examples/s]9807 examples [00:10, 976.96 examples/s]9905 examples [00:10, 965.31 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJ4A6OU/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJ4A6OU/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fce2a971620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fce2a971620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fce2a971620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fce27b96b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fcdb4bf7ac8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fce2a971620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fce2a971620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fce2a971620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fce27b966d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fce119a43c8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fcda2405378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fcda2405378> 

  function with postional parmater data_info <function split_train_valid at 0x7fcda2405378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fcda2405488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fcda2405488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fcda2405488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=809736beb2ec03971946ff15f050d191a2ef77e8bc4e17052fec95e3b4778c96
  Stored in directory: /tmp/pip-ephem-wheel-cache-v6p0dq3t/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<29:23:19, 8.15kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<20:48:17, 11.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<14:36:56, 16.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<10:14:12, 23.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.49M/862M [00:01<7:08:52, 33.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.17M/862M [00:01<4:58:19, 47.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.8M/862M [00:01<3:27:32, 68.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:01<2:24:22, 97.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:02<1:40:28, 139kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 31.6M/862M [00:02<1:09:58, 198kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.0M/862M [00:02<48:54, 282kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 40.4M/862M [00:02<34:05, 402kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<23:54, 571kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.8M/862M [00:02<16:42, 812kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:03<12:11, 1.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:05<10:24, 1.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<11:06, 1.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<08:43, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:05<06:17, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<09:33, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<08:13, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<06:17, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.3M/862M [00:07<04:35, 2.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<08:57, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<08:02, 1.65MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:09<06:00, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<04:21, 3.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<31:09, 424kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<23:11, 570kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<16:38, 793kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.8M/862M [00:11<11:49, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<15:04, 872kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<12:08, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:13<08:48, 1.49MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:13<06:20, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<21:08, 619kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<16:17, 803kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:15<11:43, 1.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<10:58, 1.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<10:28, 1.24MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<07:56, 1.64MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:17<05:49, 2.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<07:27, 1.74MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:19<06:20, 2.04MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<04:51, 2.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:19<03:32, 3.63MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<26:45, 481kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<20:02, 642kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:21<14:20, 896kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<13:02, 982kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<10:26, 1.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:23<07:33, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<08:18, 1.53MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<08:24, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:25<06:26, 1.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:41, 2.71MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:33, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:17, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:24, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:44, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:15, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:42, 2.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:01, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:28, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:08, 3.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:49, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:29, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:09, 2.41MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:37, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:32, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:07, 2.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:44, 3.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:45, 1.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:23, 1.67MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:25, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:43, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:10, 1.71MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:33, 2.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<04:02, 3.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:24, 1.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:19, 1.46MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:03, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<04:26, 2.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:15:29, 161kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<55:16, 219kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<39:15, 308kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<29:19, 411kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<21:44, 554kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<15:29, 776kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<13:37, 880kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<10:44, 1.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<07:45, 1.54MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:15, 1.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:11, 1.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:14, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:30, 2.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:53, 1.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:07, 1.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:58, 1.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:56, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<07:16, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:40, 2.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:51, 2.00MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:18, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:00, 2.92MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<05:30, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:12, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:50, 2.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:30, 3.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<13:09, 879kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<10:23, 1.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<07:31, 1.53MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:57, 1.45MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:56, 1.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:08, 1.87MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:25, 2.59MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<11:19, 1.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:56, 1.28MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:35, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:44, 2.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<19:39, 578kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<14:44, 771kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<10:38, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<07:34, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<19:49, 570kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<14:50, 761kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<10:40, 1.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<07:35, 1.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<45:12, 248kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<32:46, 343kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<23:08, 484kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<18:48, 594kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<14:18, 780kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<10:16, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:47, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<09:08, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:53, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:59, 2.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:22, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:19, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:39, 2.36MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<05:49, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:11, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:52, 2.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:17, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:57, 1.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:38, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<03:23, 3.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:31, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:22, 1.70MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:40, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:48, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:14, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:49, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<03:30, 3.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<08:43, 1.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<07:12, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:18, 2.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:10, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:27, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:59, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:36, 2.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<10:31, 1.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<08:25, 1.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<06:06, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:44, 1.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:49, 1.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:18, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:23, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:49, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:37, 2.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:57, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:38, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:24, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:11, 3.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<12:01, 854kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<09:28, 1.08MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<06:49, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:10, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:05, 1.44MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:24, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:53, 2.60MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<08:50, 1.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<07:12, 1.41MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:15, 1.92MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:02, 1.67MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:15, 1.92MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:55, 2.56MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:05, 1.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:35, 1.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:24, 2.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:41, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:19, 2.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:16, 3.03MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<04:35, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:13, 2.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:09, 3.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:29, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:03, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:01, 2.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:23, 2.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:05, 2.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:07, 3.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:23, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<03:52, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:16, 2.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<02:26, 3.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:12, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:38, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<03:51, 2.48MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<02:50, 3.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<05:00, 1.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<05:27, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:03, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:46, 2.52MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<02:47, 3.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<07:45, 1.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:23, 1.48MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:50, 1.95MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:29, 2.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<10:37, 882kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<09:19, 1.00MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:55, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:01, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<06:25, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<06:29, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:57, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:38, 2.54MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:13, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:37, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:25, 2.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:35, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<05:04, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:01, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<02:54, 3.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<12:26, 732kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<09:30, 956kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:53, 1.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:54, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<22:13, 406kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<17:24, 519kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<12:34, 717kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<08:56, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<09:06, 984kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<07:17, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:16, 1.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<05:46, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<05:51, 1.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:29, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<03:13, 2.74MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<09:01, 977kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<07:13, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:16, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<05:44, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<05:47, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:29, 1.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:14, 2.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<1:04:25, 135kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<45:56, 189kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<32:16, 268kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<24:31, 351kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<18:54, 456kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<13:39, 630kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<09:37, 889kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<1:07:49, 126kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<48:18, 177kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<33:55, 251kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<25:40, 330kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<19:40, 431kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<14:10, 597kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<11:14, 749kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<08:43, 964kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:18, 1.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:20, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<06:11, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<04:46, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:25, 2.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<1:01:36, 134kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<43:56, 188kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<30:52, 267kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<23:26, 350kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<17:14, 476kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<12:12, 670kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<10:27, 779kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<08:57, 909kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<06:36, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:42, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<08:29, 951kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<06:45, 1.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:55, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:18, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:24, 1.48MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<04:11, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:01, 2.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<31:03, 255kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<22:31, 352kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<15:54, 497kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<12:57, 607kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<10:39, 738kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<07:50, 1.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<06:43, 1.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:29, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:01, 1.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<04:34, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:42, 1.64MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:36, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:37, 2.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:27, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:36, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:23, 2.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:08, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:40, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:43, 2.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:41, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:21, 2.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<02:32, 2.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:31, 2.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:00, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:09, 2.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<02:18, 3.21MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<04:59, 1.48MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:16, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:09, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:17, 3.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<28:45, 254kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<20:41, 353kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<14:52, 491kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<10:29, 693kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<10:05, 718kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<07:51, 922kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:39, 1.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<04:02, 1.78MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<12:02, 596kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<09:58, 719kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<07:16, 985kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:17, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<05:16, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<04:26, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:15, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:21, 2.98MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<19:50, 355kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<14:36, 481kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<10:19, 679kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<07:18, 954kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<14:45, 472kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<11:05, 628kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:53, 879kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<07:01, 982kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<06:22, 1.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:44, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:24, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<05:05, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:14<04:15, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:08, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:45, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:59, 1.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:05, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:16, 2.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:45, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<03:18, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:28, 2.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:16, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<02:58, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:14, 2.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:06, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:29, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<02:44, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<01:59, 3.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<06:22, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<05:06, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:42, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<04:05, 1.57MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<04:09, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:10, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:18, 2.77MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<04:57, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<04:07, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:00, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:34, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<03:08, 2.00MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<02:19, 2.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:06, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<03:23, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<02:38, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<01:55, 3.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:23, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:42, 1.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:42, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:19, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:57, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:12, 2.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:57, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:16, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:35, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:46, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:32, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<01:53, 3.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:42, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:02, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:22, 2.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<01:45, 3.32MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:20, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:56, 1.97MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:12, 2.62MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:52, 1.99MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:36, 2.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<01:57, 2.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:42, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:28, 2.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:51, 3.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:37, 2.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:19, 2.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<01:46, 3.14MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<01:18, 4.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<11:41, 473kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<08:44, 632kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<06:14, 882kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<05:37, 970kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:29, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:14, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:32, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:57, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:12, 2.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:36, 3.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<14:30, 367kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<10:41, 497kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<07:35, 698kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<06:30, 807kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<05:05, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:40, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:47, 1.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:04, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:16, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:48, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:59, 1.71MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:20, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:41, 3.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<4:51:15, 17.3kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<3:24:08, 24.7kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<2:22:17, 35.3kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<1:40:03, 49.8kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<1:10:59, 70.1kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<49:46, 99.8kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<34:41, 142kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<26:16, 187kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<18:51, 260kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<13:15, 368kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<10:21, 468kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<08:13, 589kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<05:59, 807kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<04:55, 971kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<03:55, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:50, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:04, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:05, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:21, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:41, 2.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<04:45, 975kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<03:48, 1.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:45, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<02:59, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<02:33, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:53, 2.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:22, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<02:34, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:01, 2.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:07, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<01:55, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:27, 3.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:01, 2.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:18, 1.89MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<01:49, 2.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:18, 3.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<4:09:31, 17.2kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<2:54:50, 24.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<2:01:45, 35.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<1:25:29, 49.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<1:00:38, 69.7kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<42:30, 99.1kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<29:28, 141kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<25:20, 164kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<18:03, 230kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<12:42, 325kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [04:31<08:51, 462kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<13:55, 294kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<10:09, 402kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<07:09, 567kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<05:54, 681kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<04:32, 884kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:15, 1.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<03:11, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<03:02, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:18, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:40, 2.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:34, 1.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:12, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:37, 2.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:01, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<01:48, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:20, 2.82MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:48, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:35, 2.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:13, 3.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<00:53, 4.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<08:17, 444kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<06:10, 595kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:22, 832kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<03:53, 928kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<03:27, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<02:34, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:50, 1.93MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:36, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:11, 1.61MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:36, 2.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:55, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:38, 2.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:13, 2.81MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<00:53, 3.79MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<26:03, 131kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<18:54, 180kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<13:21, 254kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<09:44, 342kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<07:30, 444kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<05:24, 614kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<03:46, 867kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<26:03, 125kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<18:32, 176kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<12:56, 250kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<09:42, 329kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<07:26, 430kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<05:19, 598kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<03:42, 846kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<06:31, 479kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<04:52, 641kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:27, 897kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<03:06, 984kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<02:47, 1.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<02:06, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:28, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<30:12, 99.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<21:24, 140kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<14:55, 198kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<11:00, 266kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<08:17, 353kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<05:55, 491kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<04:32, 630kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<03:27, 823kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:28, 1.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:21, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:12, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:40, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:35, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<01:23, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:01, 2.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:20, 1.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:11, 2.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<00:53, 2.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:13, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:22, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:04, 2.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<00:46, 3.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:50, 1.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:32, 1.63MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:07, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:21, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:26, 1.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:07, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:09, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:00, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<00:45, 3.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:04, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:13, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<00:57, 2.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:41, 3.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:21, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:10, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:52, 2.54MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:06, 1.95MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:00, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:44, 2.90MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:32, 3.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<15:58, 131kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<11:36, 181kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<08:10, 255kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<05:39, 363kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<04:45, 428kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<03:31, 575kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<02:29, 805kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<02:10, 906kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:42, 1.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:13, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:17, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:05, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:47, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:58, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:52, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:38, 2.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:51, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:46, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:34, 3.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:47, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:53, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:41, 2.41MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:30, 3.20MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:47, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<00:43, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:32, 2.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:43, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:49, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:38, 2.36MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:27, 3.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<10:56, 135kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<07:46, 190kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<05:22, 269kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<04:00, 353kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<02:55, 480kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<02:02, 675kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:42, 784kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:28, 914kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:04, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:44, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:33, 822kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:12, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:51, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:52, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:50, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:38, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:26, 2.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:54, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:44, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:32, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:37, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:38, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:30, 2.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:29, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:26, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:19, 2.94MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:03<00:26, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:03<00:29, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:22, 2.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:16, 3.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:33, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:27, 1.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:19, 2.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:24, 1.94MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:07<00:26, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:20, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:20, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:18, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:13, 3.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:18, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:16, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:12, 3.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:18, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:14, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:09, 3.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:56, 559kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:41, 737kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:28, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:24, 1.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:19, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:13, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:14, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:14, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:10, 2.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:07, 2.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:10, 1.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:06, 2.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:07, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:04, 2.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:05, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:02, 3.20MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:01, 2.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 2.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.83MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.34MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<75:49:08,  1.47it/s]  0%|          | 858/400000 [00:00<52:57:48,  2.09it/s]  0%|          | 1746/400000 [00:00<36:59:44,  2.99it/s]  1%|          | 2606/400000 [00:00<25:50:41,  4.27it/s]  1%|          | 3503/400000 [00:01<18:03:15,  6.10it/s]  1%|          | 4358/400000 [00:01<12:36:52,  8.71it/s]  1%|▏         | 5220/400000 [00:01<8:48:53, 12.44it/s]   2%|▏         | 6027/400000 [00:01<6:09:42, 17.76it/s]  2%|▏         | 6864/400000 [00:01<4:18:28, 25.35it/s]  2%|▏         | 7712/400000 [00:01<3:00:46, 36.17it/s]  2%|▏         | 8568/400000 [00:01<2:06:29, 51.57it/s]  2%|▏         | 9467/400000 [00:01<1:28:33, 73.50it/s]  3%|▎         | 10314/400000 [00:01<1:02:05, 104.60it/s]  3%|▎         | 11160/400000 [00:01<43:36, 148.62it/s]    3%|▎         | 12063/400000 [00:02<30:40, 210.83it/s]  3%|▎         | 12921/400000 [00:02<21:40, 297.69it/s]  3%|▎         | 13758/400000 [00:02<15:22, 418.88it/s]  4%|▎         | 14665/400000 [00:02<10:56, 586.78it/s]  4%|▍         | 15525/400000 [00:02<07:52, 814.43it/s]  4%|▍         | 16459/400000 [00:02<05:41, 1121.52it/s]  4%|▍         | 17339/400000 [00:02<04:12, 1516.55it/s]  5%|▍         | 18210/400000 [00:02<03:09, 2012.88it/s]  5%|▍         | 19122/400000 [00:02<02:24, 2627.04it/s]  5%|▌         | 20002/400000 [00:03<01:54, 3313.86it/s]  5%|▌         | 20877/400000 [00:03<01:33, 4072.45it/s]  5%|▌         | 21749/400000 [00:03<01:18, 4828.15it/s]  6%|▌         | 22615/400000 [00:03<01:08, 5539.34it/s]  6%|▌         | 23475/400000 [00:03<01:02, 6068.34it/s]  6%|▌         | 24311/400000 [00:03<00:56, 6601.68it/s]  6%|▋         | 25157/400000 [00:03<00:53, 7067.33it/s]  6%|▋         | 25995/400000 [00:03<00:50, 7408.58it/s]  7%|▋         | 26832/400000 [00:03<00:48, 7663.61it/s]  7%|▋         | 27671/400000 [00:03<00:47, 7840.61it/s]  7%|▋         | 28505/400000 [00:04<00:47, 7839.09it/s]  7%|▋         | 29328/400000 [00:04<00:46, 7951.79it/s]  8%|▊         | 30195/400000 [00:04<00:45, 8153.66it/s]  8%|▊         | 31109/400000 [00:04<00:43, 8425.19it/s]  8%|▊         | 32009/400000 [00:04<00:42, 8589.52it/s]  8%|▊         | 32894/400000 [00:04<00:42, 8665.68it/s]  8%|▊         | 33822/400000 [00:04<00:41, 8840.61it/s]  9%|▊         | 34713/400000 [00:04<00:41, 8775.05it/s]  9%|▉         | 35610/400000 [00:04<00:41, 8832.45it/s]  9%|▉         | 36520/400000 [00:04<00:40, 8909.74it/s]  9%|▉         | 37414/400000 [00:05<00:41, 8742.31it/s] 10%|▉         | 38302/400000 [00:05<00:41, 8781.56it/s] 10%|▉         | 39220/400000 [00:05<00:40, 8897.12it/s] 10%|█         | 40158/400000 [00:05<00:39, 9034.23it/s] 10%|█         | 41064/400000 [00:05<00:40, 8923.42it/s] 10%|█         | 41958/400000 [00:05<00:40, 8889.08it/s] 11%|█         | 42865/400000 [00:05<00:39, 8940.28it/s] 11%|█         | 43762/400000 [00:05<00:39, 8948.12it/s] 11%|█         | 44671/400000 [00:05<00:39, 8989.99it/s] 11%|█▏        | 45622/400000 [00:05<00:38, 9138.14it/s] 12%|█▏        | 46537/400000 [00:06<00:38, 9095.17it/s] 12%|█▏        | 47448/400000 [00:06<00:39, 8833.83it/s] 12%|█▏        | 48334/400000 [00:06<00:39, 8814.33it/s] 12%|█▏        | 49234/400000 [00:06<00:39, 8868.90it/s] 13%|█▎        | 50138/400000 [00:06<00:39, 8918.56it/s] 13%|█▎        | 51031/400000 [00:06<00:39, 8816.88it/s] 13%|█▎        | 51914/400000 [00:06<00:39, 8768.41it/s] 13%|█▎        | 52794/400000 [00:06<00:39, 8775.93it/s] 13%|█▎        | 53740/400000 [00:06<00:38, 8968.27it/s] 14%|█▎        | 54668/400000 [00:06<00:38, 9058.37it/s] 14%|█▍        | 55576/400000 [00:07<00:39, 8686.40it/s] 14%|█▍        | 56449/400000 [00:07<00:40, 8475.43it/s] 14%|█▍        | 57330/400000 [00:07<00:39, 8570.61it/s] 15%|█▍        | 58196/400000 [00:07<00:39, 8595.25it/s] 15%|█▍        | 59058/400000 [00:07<00:39, 8577.79it/s] 15%|█▍        | 59918/400000 [00:07<00:39, 8584.21it/s] 15%|█▌        | 60785/400000 [00:07<00:39, 8607.40it/s] 15%|█▌        | 61712/400000 [00:07<00:38, 8795.73it/s] 16%|█▌        | 62594/400000 [00:07<00:38, 8722.83it/s] 16%|█▌        | 63488/400000 [00:07<00:38, 8786.44it/s] 16%|█▌        | 64370/400000 [00:08<00:38, 8795.75it/s] 16%|█▋        | 65251/400000 [00:08<00:38, 8739.19it/s] 17%|█▋        | 66183/400000 [00:08<00:37, 8905.27it/s] 17%|█▋        | 67103/400000 [00:08<00:37, 8990.08it/s] 17%|█▋        | 68006/400000 [00:08<00:36, 9001.56it/s] 17%|█▋        | 68907/400000 [00:08<00:37, 8791.70it/s] 17%|█▋        | 69788/400000 [00:08<00:37, 8710.74it/s] 18%|█▊        | 70720/400000 [00:08<00:37, 8882.71it/s] 18%|█▊        | 71656/400000 [00:08<00:36, 9019.59it/s] 18%|█▊        | 72576/400000 [00:09<00:36, 9071.51it/s] 18%|█▊        | 73485/400000 [00:09<00:36, 8891.42it/s] 19%|█▊        | 74407/400000 [00:09<00:36, 8984.93it/s] 19%|█▉        | 75307/400000 [00:09<00:36, 8967.96it/s] 19%|█▉        | 76205/400000 [00:09<00:36, 8935.55it/s] 19%|█▉        | 77124/400000 [00:09<00:35, 9008.63it/s] 20%|█▉        | 78026/400000 [00:09<00:36, 8805.47it/s] 20%|█▉        | 78909/400000 [00:09<00:36, 8805.06it/s] 20%|█▉        | 79791/400000 [00:09<00:36, 8785.27it/s] 20%|██        | 80671/400000 [00:09<00:37, 8630.05it/s] 20%|██        | 81536/400000 [00:10<00:37, 8550.00it/s] 21%|██        | 82392/400000 [00:10<00:37, 8363.11it/s] 21%|██        | 83230/400000 [00:10<00:39, 8119.57it/s] 21%|██        | 84121/400000 [00:10<00:37, 8340.02it/s] 21%|██▏       | 85028/400000 [00:10<00:36, 8546.25it/s] 21%|██▏       | 85923/400000 [00:10<00:36, 8662.59it/s] 22%|██▏       | 86793/400000 [00:10<00:37, 8452.42it/s] 22%|██▏       | 87652/400000 [00:10<00:36, 8492.20it/s] 22%|██▏       | 88504/400000 [00:10<00:36, 8473.40it/s] 22%|██▏       | 89367/400000 [00:10<00:36, 8519.38it/s] 23%|██▎       | 90264/400000 [00:11<00:35, 8647.14it/s] 23%|██▎       | 91130/400000 [00:11<00:36, 8353.66it/s] 23%|██▎       | 92018/400000 [00:11<00:36, 8503.39it/s] 23%|██▎       | 92956/400000 [00:11<00:35, 8746.29it/s] 23%|██▎       | 93856/400000 [00:11<00:34, 8820.74it/s] 24%|██▎       | 94782/400000 [00:11<00:34, 8947.08it/s] 24%|██▍       | 95680/400000 [00:11<00:34, 8756.88it/s] 24%|██▍       | 96559/400000 [00:11<00:34, 8734.86it/s] 24%|██▍       | 97435/400000 [00:11<00:34, 8668.45it/s] 25%|██▍       | 98330/400000 [00:11<00:34, 8749.67it/s] 25%|██▍       | 99222/400000 [00:12<00:34, 8797.47it/s] 25%|██▌       | 100103/400000 [00:12<00:34, 8606.18it/s] 25%|██▌       | 100966/400000 [00:12<00:34, 8579.40it/s] 25%|██▌       | 101875/400000 [00:12<00:34, 8726.11it/s] 26%|██▌       | 102802/400000 [00:12<00:33, 8880.95it/s] 26%|██▌       | 103692/400000 [00:12<00:33, 8858.21it/s] 26%|██▌       | 104579/400000 [00:12<00:33, 8697.71it/s] 26%|██▋       | 105451/400000 [00:12<00:34, 8571.47it/s] 27%|██▋       | 106323/400000 [00:12<00:34, 8614.91it/s] 27%|██▋       | 107223/400000 [00:13<00:33, 8726.51it/s] 27%|██▋       | 108097/400000 [00:13<00:34, 8394.62it/s] 27%|██▋       | 108940/400000 [00:13<00:34, 8332.15it/s] 27%|██▋       | 109778/400000 [00:13<00:34, 8345.95it/s] 28%|██▊       | 110615/400000 [00:13<00:36, 8021.08it/s] 28%|██▊       | 111437/400000 [00:13<00:35, 8079.49it/s] 28%|██▊       | 112248/400000 [00:13<00:35, 8051.86it/s] 28%|██▊       | 113065/400000 [00:13<00:35, 8085.72it/s] 28%|██▊       | 113883/400000 [00:13<00:35, 8113.46it/s] 29%|██▊       | 114738/400000 [00:13<00:34, 8237.96it/s] 29%|██▉       | 115604/400000 [00:14<00:34, 8358.06it/s] 29%|██▉       | 116465/400000 [00:14<00:33, 8431.44it/s] 29%|██▉       | 117310/400000 [00:14<00:33, 8332.89it/s] 30%|██▉       | 118200/400000 [00:14<00:33, 8493.98it/s] 30%|██▉       | 119055/400000 [00:14<00:33, 8508.24it/s] 30%|██▉       | 119959/400000 [00:14<00:32, 8660.64it/s] 30%|███       | 120827/400000 [00:14<00:32, 8664.57it/s] 30%|███       | 121732/400000 [00:14<00:31, 8776.20it/s] 31%|███       | 122617/400000 [00:14<00:31, 8797.75it/s] 31%|███       | 123498/400000 [00:14<00:31, 8750.19it/s] 31%|███       | 124374/400000 [00:15<00:32, 8504.00it/s] 31%|███▏      | 125227/400000 [00:15<00:32, 8330.38it/s] 32%|███▏      | 126127/400000 [00:15<00:32, 8519.43it/s] 32%|███▏      | 127001/400000 [00:15<00:31, 8582.53it/s] 32%|███▏      | 127906/400000 [00:15<00:31, 8715.38it/s] 32%|███▏      | 128861/400000 [00:15<00:30, 8947.86it/s] 32%|███▏      | 129759/400000 [00:15<00:30, 8917.63it/s] 33%|███▎      | 130661/400000 [00:15<00:30, 8946.86it/s] 33%|███▎      | 131568/400000 [00:15<00:29, 8982.33it/s] 33%|███▎      | 132468/400000 [00:15<00:31, 8560.94it/s] 33%|███▎      | 133329/400000 [00:16<00:31, 8527.53it/s] 34%|███▎      | 134186/400000 [00:16<00:31, 8401.05it/s] 34%|███▍      | 135029/400000 [00:16<00:32, 8259.62it/s] 34%|███▍      | 135869/400000 [00:16<00:31, 8299.63it/s] 34%|███▍      | 136702/400000 [00:16<00:31, 8306.78it/s] 34%|███▍      | 137534/400000 [00:16<00:32, 8195.90it/s] 35%|███▍      | 138355/400000 [00:16<00:32, 8143.51it/s] 35%|███▍      | 139231/400000 [00:16<00:31, 8318.89it/s] 35%|███▌      | 140085/400000 [00:16<00:31, 8382.35it/s] 35%|███▌      | 140946/400000 [00:17<00:30, 8446.20it/s] 35%|███▌      | 141792/400000 [00:17<00:31, 8329.26it/s] 36%|███▌      | 142638/400000 [00:17<00:30, 8366.76it/s] 36%|███▌      | 143566/400000 [00:17<00:29, 8620.69it/s] 36%|███▌      | 144498/400000 [00:17<00:28, 8816.92it/s] 36%|███▋      | 145424/400000 [00:17<00:28, 8944.64it/s] 37%|███▋      | 146367/400000 [00:17<00:27, 9084.77it/s] 37%|███▋      | 147278/400000 [00:17<00:28, 8958.30it/s] 37%|███▋      | 148176/400000 [00:17<00:28, 8911.57it/s] 37%|███▋      | 149069/400000 [00:17<00:28, 8802.48it/s] 37%|███▋      | 149951/400000 [00:18<00:28, 8758.48it/s] 38%|███▊      | 150856/400000 [00:18<00:28, 8842.09it/s] 38%|███▊      | 151742/400000 [00:18<00:29, 8492.04it/s] 38%|███▊      | 152595/400000 [00:18<00:29, 8359.16it/s] 38%|███▊      | 153494/400000 [00:18<00:28, 8536.52it/s] 39%|███▊      | 154411/400000 [00:18<00:28, 8716.10it/s] 39%|███▉      | 155293/400000 [00:18<00:27, 8745.25it/s] 39%|███▉      | 156171/400000 [00:18<00:27, 8753.93it/s] 39%|███▉      | 157068/400000 [00:18<00:27, 8816.44it/s] 39%|███▉      | 157951/400000 [00:18<00:27, 8780.33it/s] 40%|███▉      | 158830/400000 [00:19<00:27, 8709.04it/s] 40%|███▉      | 159702/400000 [00:19<00:28, 8563.24it/s] 40%|████      | 160566/400000 [00:19<00:27, 8585.10it/s] 40%|████      | 161472/400000 [00:19<00:27, 8720.58it/s] 41%|████      | 162371/400000 [00:19<00:27, 8799.68it/s] 41%|████      | 163284/400000 [00:19<00:26, 8896.19it/s] 41%|████      | 164175/400000 [00:19<00:26, 8775.43it/s] 41%|████▏     | 165054/400000 [00:19<00:27, 8550.94it/s] 41%|████▏     | 165912/400000 [00:19<00:28, 8233.74it/s] 42%|████▏     | 166824/400000 [00:19<00:27, 8479.17it/s] 42%|████▏     | 167677/400000 [00:20<00:27, 8408.77it/s] 42%|████▏     | 168522/400000 [00:20<00:27, 8408.86it/s] 42%|████▏     | 169377/400000 [00:20<00:27, 8448.49it/s] 43%|████▎     | 170287/400000 [00:20<00:26, 8633.80it/s] 43%|████▎     | 171162/400000 [00:20<00:26, 8666.16it/s] 43%|████▎     | 172031/400000 [00:20<00:26, 8612.14it/s] 43%|████▎     | 172894/400000 [00:20<00:26, 8569.65it/s] 43%|████▎     | 173752/400000 [00:20<00:26, 8435.43it/s] 44%|████▎     | 174682/400000 [00:20<00:25, 8676.20it/s] 44%|████▍     | 175622/400000 [00:20<00:25, 8880.47it/s] 44%|████▍     | 176568/400000 [00:21<00:24, 9046.47it/s] 44%|████▍     | 177476/400000 [00:21<00:25, 8884.63it/s] 45%|████▍     | 178368/400000 [00:21<00:25, 8778.80it/s] 45%|████▍     | 179263/400000 [00:21<00:25, 8828.73it/s] 45%|████▌     | 180148/400000 [00:21<00:25, 8741.73it/s] 45%|████▌     | 181070/400000 [00:21<00:24, 8877.36it/s] 46%|████▌     | 182001/400000 [00:21<00:24, 9002.66it/s] 46%|████▌     | 182903/400000 [00:21<00:24, 8899.01it/s] 46%|████▌     | 183822/400000 [00:21<00:24, 8983.39it/s] 46%|████▌     | 184722/400000 [00:22<00:24, 8955.91it/s] 46%|████▋     | 185619/400000 [00:22<00:24, 8579.83it/s] 47%|████▋     | 186481/400000 [00:22<00:25, 8438.27it/s] 47%|████▋     | 187329/400000 [00:22<00:25, 8414.28it/s] 47%|████▋     | 188173/400000 [00:22<00:25, 8358.43it/s] 47%|████▋     | 189011/400000 [00:22<00:25, 8336.04it/s] 47%|████▋     | 189916/400000 [00:22<00:24, 8536.24it/s] 48%|████▊     | 190858/400000 [00:22<00:23, 8783.37it/s] 48%|████▊     | 191746/400000 [00:22<00:23, 8811.60it/s] 48%|████▊     | 192630/400000 [00:22<00:23, 8814.81it/s] 48%|████▊     | 193514/400000 [00:23<00:24, 8588.98it/s] 49%|████▊     | 194376/400000 [00:23<00:24, 8424.38it/s] 49%|████▉     | 195228/400000 [00:23<00:24, 8452.23it/s] 49%|████▉     | 196075/400000 [00:23<00:24, 8418.77it/s] 49%|████▉     | 196919/400000 [00:23<00:24, 8326.92it/s] 49%|████▉     | 197852/400000 [00:23<00:23, 8602.73it/s] 50%|████▉     | 198718/400000 [00:23<00:23, 8617.90it/s] 50%|████▉     | 199654/400000 [00:23<00:22, 8826.53it/s] 50%|█████     | 200540/400000 [00:23<00:22, 8717.01it/s] 50%|█████     | 201450/400000 [00:23<00:22, 8826.83it/s] 51%|█████     | 202360/400000 [00:24<00:22, 8903.61it/s] 51%|█████     | 203252/400000 [00:24<00:22, 8903.40it/s] 51%|█████     | 204165/400000 [00:24<00:21, 8968.20it/s] 51%|█████▏    | 205063/400000 [00:24<00:22, 8839.10it/s] 51%|█████▏    | 205989/400000 [00:24<00:21, 8958.68it/s] 52%|█████▏    | 206886/400000 [00:24<00:21, 8810.69it/s] 52%|█████▏    | 207769/400000 [00:24<00:22, 8695.12it/s] 52%|█████▏    | 208640/400000 [00:24<00:22, 8681.99it/s] 52%|█████▏    | 209510/400000 [00:24<00:22, 8644.04it/s] 53%|█████▎    | 210443/400000 [00:24<00:21, 8836.96it/s] 53%|█████▎    | 211329/400000 [00:25<00:21, 8821.04it/s] 53%|█████▎    | 212213/400000 [00:25<00:21, 8817.79it/s] 53%|█████▎    | 213110/400000 [00:25<00:21, 8860.69it/s] 53%|█████▎    | 213997/400000 [00:25<00:21, 8783.78it/s] 54%|█████▎    | 214938/400000 [00:25<00:20, 8962.17it/s] 54%|█████▍    | 215855/400000 [00:25<00:20, 9021.49it/s] 54%|█████▍    | 216765/400000 [00:25<00:20, 9043.30it/s] 54%|█████▍    | 217713/400000 [00:25<00:19, 9168.29it/s] 55%|█████▍    | 218631/400000 [00:25<00:20, 9062.03it/s] 55%|█████▍    | 219542/400000 [00:25<00:19, 9072.86it/s] 55%|█████▌    | 220450/400000 [00:26<00:19, 9044.02it/s] 55%|█████▌    | 221355/400000 [00:26<00:20, 8911.57it/s] 56%|█████▌    | 222247/400000 [00:26<00:20, 8774.49it/s] 56%|█████▌    | 223126/400000 [00:26<00:20, 8582.82it/s] 56%|█████▌    | 224012/400000 [00:26<00:20, 8662.12it/s] 56%|█████▌    | 224937/400000 [00:26<00:19, 8828.18it/s] 56%|█████▋    | 225854/400000 [00:26<00:19, 8925.71it/s] 57%|█████▋    | 226750/400000 [00:26<00:19, 8935.45it/s] 57%|█████▋    | 227645/400000 [00:26<00:19, 8766.73it/s] 57%|█████▋    | 228538/400000 [00:27<00:19, 8813.33it/s] 57%|█████▋    | 229434/400000 [00:27<00:19, 8855.70it/s] 58%|█████▊    | 230331/400000 [00:27<00:19, 8888.69it/s] 58%|█████▊    | 231243/400000 [00:27<00:18, 8956.65it/s] 58%|█████▊    | 232140/400000 [00:27<00:19, 8778.85it/s] 58%|█████▊    | 233035/400000 [00:27<00:18, 8828.35it/s] 58%|█████▊    | 233981/400000 [00:27<00:18, 9006.09it/s] 59%|█████▊    | 234904/400000 [00:27<00:18, 9069.76it/s] 59%|█████▉    | 235813/400000 [00:27<00:18, 8879.44it/s] 59%|█████▉    | 236703/400000 [00:27<00:18, 8719.64it/s] 59%|█████▉    | 237620/400000 [00:28<00:18, 8848.81it/s] 60%|█████▉    | 238508/400000 [00:28<00:18, 8855.28it/s] 60%|█████▉    | 239400/400000 [00:28<00:18, 8873.73it/s] 60%|██████    | 240289/400000 [00:28<00:18, 8782.88it/s] 60%|██████    | 241169/400000 [00:28<00:18, 8427.31it/s] 61%|██████    | 242024/400000 [00:28<00:18, 8461.61it/s] 61%|██████    | 242877/400000 [00:28<00:18, 8481.66it/s] 61%|██████    | 243744/400000 [00:28<00:18, 8535.43it/s] 61%|██████    | 244648/400000 [00:28<00:17, 8680.52it/s] 61%|██████▏   | 245518/400000 [00:28<00:17, 8601.53it/s] 62%|██████▏   | 246422/400000 [00:29<00:17, 8726.60it/s] 62%|██████▏   | 247296/400000 [00:29<00:17, 8642.85it/s] 62%|██████▏   | 248210/400000 [00:29<00:17, 8785.45it/s] 62%|██████▏   | 249100/400000 [00:29<00:17, 8817.74it/s] 62%|██████▏   | 249983/400000 [00:29<00:17, 8575.17it/s] 63%|██████▎   | 250851/400000 [00:29<00:17, 8605.42it/s] 63%|██████▎   | 251714/400000 [00:29<00:17, 8485.40it/s] 63%|██████▎   | 252564/400000 [00:29<00:17, 8466.14it/s] 63%|██████▎   | 253435/400000 [00:29<00:17, 8536.58it/s] 64%|██████▎   | 254290/400000 [00:29<00:17, 8507.97it/s] 64%|██████▍   | 255182/400000 [00:30<00:16, 8622.58it/s] 64%|██████▍   | 256078/400000 [00:30<00:16, 8720.68it/s] 64%|██████▍   | 256995/400000 [00:30<00:16, 8847.87it/s] 64%|██████▍   | 257881/400000 [00:30<00:16, 8742.12it/s] 65%|██████▍   | 258757/400000 [00:30<00:16, 8628.48it/s] 65%|██████▍   | 259707/400000 [00:30<00:15, 8871.05it/s] 65%|██████▌   | 260650/400000 [00:30<00:15, 9029.25it/s] 65%|██████▌   | 261557/400000 [00:30<00:15, 9040.70it/s] 66%|██████▌   | 262463/400000 [00:30<00:15, 9019.86it/s] 66%|██████▌   | 263367/400000 [00:31<00:15, 8661.29it/s] 66%|██████▌   | 264239/400000 [00:31<00:15, 8677.87it/s] 66%|██████▋   | 265148/400000 [00:31<00:15, 8795.31it/s] 67%|██████▋   | 266071/400000 [00:31<00:15, 8919.71it/s] 67%|██████▋   | 266988/400000 [00:31<00:14, 8991.14it/s] 67%|██████▋   | 267889/400000 [00:31<00:14, 8821.28it/s] 67%|██████▋   | 268773/400000 [00:31<00:14, 8773.38it/s] 67%|██████▋   | 269680/400000 [00:31<00:14, 8858.48it/s] 68%|██████▊   | 270576/400000 [00:31<00:14, 8886.57it/s] 68%|██████▊   | 271466/400000 [00:31<00:15, 8534.10it/s] 68%|██████▊   | 272337/400000 [00:32<00:14, 8583.27it/s] 68%|██████▊   | 273257/400000 [00:32<00:14, 8759.30it/s] 69%|██████▊   | 274198/400000 [00:32<00:14, 8943.17it/s] 69%|██████▉   | 275103/400000 [00:32<00:13, 8974.87it/s] 69%|██████▉   | 276003/400000 [00:32<00:13, 8861.00it/s] 69%|██████▉   | 276891/400000 [00:32<00:13, 8830.88it/s] 69%|██████▉   | 277789/400000 [00:32<00:13, 8873.03it/s] 70%|██████▉   | 278705/400000 [00:32<00:13, 8955.82it/s] 70%|██████▉   | 279608/400000 [00:32<00:13, 8974.06it/s] 70%|███████   | 280506/400000 [00:32<00:13, 8688.67it/s] 70%|███████   | 281378/400000 [00:33<00:13, 8631.62it/s] 71%|███████   | 282286/400000 [00:33<00:13, 8761.06it/s] 71%|███████   | 283215/400000 [00:33<00:13, 8913.01it/s] 71%|███████   | 284109/400000 [00:33<00:13, 8740.84it/s] 71%|███████   | 284986/400000 [00:33<00:13, 8734.76it/s] 71%|███████▏  | 285861/400000 [00:33<00:13, 8708.46it/s] 72%|███████▏  | 286733/400000 [00:33<00:13, 8647.24it/s] 72%|███████▏  | 287599/400000 [00:33<00:13, 8548.00it/s] 72%|███████▏  | 288455/400000 [00:33<00:13, 8522.22it/s] 72%|███████▏  | 289308/400000 [00:33<00:13, 8382.52it/s] 73%|███████▎  | 290148/400000 [00:34<00:13, 8309.51it/s] 73%|███████▎  | 290980/400000 [00:34<00:13, 7985.46it/s] 73%|███████▎  | 291856/400000 [00:34<00:13, 8201.42it/s] 73%|███████▎  | 292749/400000 [00:34<00:12, 8406.21it/s] 73%|███████▎  | 293594/400000 [00:34<00:12, 8400.55it/s] 74%|███████▎  | 294506/400000 [00:34<00:12, 8602.54it/s] 74%|███████▍  | 295423/400000 [00:34<00:11, 8764.43it/s] 74%|███████▍  | 296303/400000 [00:34<00:11, 8676.97it/s] 74%|███████▍  | 297173/400000 [00:34<00:11, 8577.58it/s] 75%|███████▍  | 298033/400000 [00:34<00:11, 8576.36it/s] 75%|███████▍  | 298892/400000 [00:35<00:11, 8567.41it/s] 75%|███████▍  | 299767/400000 [00:35<00:11, 8619.15it/s] 75%|███████▌  | 300661/400000 [00:35<00:11, 8711.91it/s] 75%|███████▌  | 301533/400000 [00:35<00:11, 8702.73it/s] 76%|███████▌  | 302424/400000 [00:35<00:11, 8762.61it/s] 76%|███████▌  | 303301/400000 [00:35<00:11, 8763.44it/s] 76%|███████▌  | 304195/400000 [00:35<00:10, 8815.33it/s] 76%|███████▋  | 305077/400000 [00:35<00:11, 8521.39it/s] 76%|███████▋  | 305985/400000 [00:35<00:10, 8677.12it/s] 77%|███████▋  | 306868/400000 [00:36<00:10, 8720.41it/s] 77%|███████▋  | 307742/400000 [00:36<00:10, 8535.25it/s] 77%|███████▋  | 308598/400000 [00:36<00:10, 8400.25it/s] 77%|███████▋  | 309440/400000 [00:36<00:11, 8096.36it/s] 78%|███████▊  | 310254/400000 [00:36<00:11, 8079.89it/s] 78%|███████▊  | 311068/400000 [00:36<00:10, 8095.18it/s] 78%|███████▊  | 311899/400000 [00:36<00:10, 8156.79it/s] 78%|███████▊  | 312781/400000 [00:36<00:10, 8342.88it/s] 78%|███████▊  | 313698/400000 [00:36<00:10, 8573.42it/s] 79%|███████▊  | 314650/400000 [00:36<00:09, 8836.87it/s] 79%|███████▉  | 315538/400000 [00:37<00:09, 8816.20it/s] 79%|███████▉  | 316447/400000 [00:37<00:09, 8895.39it/s] 79%|███████▉  | 317391/400000 [00:37<00:09, 9051.16it/s] 80%|███████▉  | 318299/400000 [00:37<00:09, 8845.45it/s] 80%|███████▉  | 319187/400000 [00:37<00:09, 8712.90it/s] 80%|████████  | 320061/400000 [00:37<00:09, 8395.09it/s] 80%|████████  | 320905/400000 [00:37<00:09, 8292.47it/s] 80%|████████  | 321828/400000 [00:37<00:09, 8551.15it/s] 81%|████████  | 322688/400000 [00:37<00:09, 8518.65it/s] 81%|████████  | 323613/400000 [00:37<00:08, 8724.83it/s] 81%|████████  | 324489/400000 [00:38<00:08, 8582.28it/s] 81%|████████▏ | 325393/400000 [00:38<00:08, 8713.23it/s] 82%|████████▏ | 326275/400000 [00:38<00:08, 8744.13it/s] 82%|████████▏ | 327159/400000 [00:38<00:08, 8748.12it/s] 82%|████████▏ | 328036/400000 [00:38<00:08, 8543.05it/s] 82%|████████▏ | 328893/400000 [00:38<00:08, 8487.31it/s] 82%|████████▏ | 329780/400000 [00:38<00:08, 8598.34it/s] 83%|████████▎ | 330642/400000 [00:38<00:08, 8575.39it/s] 83%|████████▎ | 331501/400000 [00:38<00:08, 8355.52it/s] 83%|████████▎ | 332339/400000 [00:39<00:08, 8165.43it/s] 83%|████████▎ | 333158/400000 [00:39<00:08, 8061.55it/s] 83%|████████▎ | 333995/400000 [00:39<00:08, 8149.84it/s] 84%|████████▎ | 334834/400000 [00:39<00:07, 8218.04it/s] 84%|████████▍ | 335671/400000 [00:39<00:07, 8260.95it/s] 84%|████████▍ | 336499/400000 [00:39<00:07, 8188.61it/s] 84%|████████▍ | 337319/400000 [00:39<00:07, 8139.41it/s] 85%|████████▍ | 338143/400000 [00:39<00:07, 8167.28it/s] 85%|████████▍ | 339028/400000 [00:39<00:07, 8359.78it/s] 85%|████████▍ | 339922/400000 [00:39<00:07, 8524.75it/s] 85%|████████▌ | 340777/400000 [00:40<00:07, 8399.86it/s] 85%|████████▌ | 341619/400000 [00:40<00:06, 8400.20it/s] 86%|████████▌ | 342527/400000 [00:40<00:06, 8591.54it/s] 86%|████████▌ | 343402/400000 [00:40<00:06, 8636.01it/s] 86%|████████▌ | 344291/400000 [00:40<00:06, 8708.66it/s] 86%|████████▋ | 345163/400000 [00:40<00:06, 8655.95it/s] 87%|████████▋ | 346030/400000 [00:40<00:06, 8528.78it/s] 87%|████████▋ | 346930/400000 [00:40<00:06, 8663.83it/s] 87%|████████▋ | 347798/400000 [00:40<00:06, 8627.69it/s] 87%|████████▋ | 348668/400000 [00:40<00:05, 8647.56it/s] 87%|████████▋ | 349567/400000 [00:41<00:05, 8745.81it/s] 88%|████████▊ | 350443/400000 [00:41<00:05, 8563.50it/s] 88%|████████▊ | 351301/400000 [00:41<00:05, 8474.08it/s] 88%|████████▊ | 352150/400000 [00:41<00:05, 8398.08it/s] 88%|████████▊ | 353005/400000 [00:41<00:05, 8442.13it/s] 88%|████████▊ | 353858/400000 [00:41<00:05, 8468.21it/s] 89%|████████▊ | 354706/400000 [00:41<00:05, 8363.57it/s] 89%|████████▉ | 355544/400000 [00:41<00:05, 8333.31it/s] 89%|████████▉ | 356378/400000 [00:41<00:05, 8188.44it/s] 89%|████████▉ | 357206/400000 [00:41<00:05, 8214.51it/s] 90%|████████▉ | 358041/400000 [00:42<00:05, 8250.85it/s] 90%|████████▉ | 358867/400000 [00:42<00:05, 8014.54it/s] 90%|████████▉ | 359726/400000 [00:42<00:04, 8175.94it/s] 90%|█████████ | 360572/400000 [00:42<00:04, 8256.54it/s] 90%|█████████ | 361438/400000 [00:42<00:04, 8372.32it/s] 91%|█████████ | 362283/400000 [00:42<00:04, 8394.22it/s] 91%|█████████ | 363128/400000 [00:42<00:04, 8408.96it/s] 91%|█████████ | 363970/400000 [00:42<00:04, 8253.35it/s] 91%|█████████ | 364797/400000 [00:42<00:04, 7986.07it/s] 91%|█████████▏| 365645/400000 [00:42<00:04, 8126.99it/s] 92%|█████████▏| 366525/400000 [00:43<00:04, 8316.81it/s] 92%|█████████▏| 367374/400000 [00:43<00:03, 8366.13it/s] 92%|█████████▏| 368301/400000 [00:43<00:03, 8615.94it/s] 92%|█████████▏| 369166/400000 [00:43<00:03, 8534.11it/s] 93%|█████████▎| 370022/400000 [00:43<00:03, 8243.53it/s] 93%|█████████▎| 370941/400000 [00:43<00:03, 8505.45it/s] 93%|█████████▎| 371797/400000 [00:43<00:03, 8451.59it/s] 93%|█████████▎| 372646/400000 [00:43<00:03, 8373.78it/s] 93%|█████████▎| 373486/400000 [00:43<00:03, 8238.57it/s] 94%|█████████▎| 374313/400000 [00:44<00:03, 8134.39it/s] 94%|█████████▍| 375129/400000 [00:44<00:03, 7965.27it/s] 94%|█████████▍| 375944/400000 [00:44<00:02, 8019.45it/s] 94%|█████████▍| 376839/400000 [00:44<00:02, 8276.62it/s] 94%|█████████▍| 377754/400000 [00:44<00:02, 8518.49it/s] 95%|█████████▍| 378638/400000 [00:44<00:02, 8610.11it/s] 95%|█████████▍| 379503/400000 [00:44<00:02, 8532.90it/s] 95%|█████████▌| 380359/400000 [00:44<00:02, 8456.87it/s] 95%|█████████▌| 381266/400000 [00:44<00:02, 8630.41it/s] 96%|█████████▌| 382178/400000 [00:44<00:02, 8769.39it/s] 96%|█████████▌| 383088/400000 [00:45<00:01, 8865.59it/s] 96%|█████████▌| 383992/400000 [00:45<00:01, 8915.79it/s] 96%|█████████▌| 384885/400000 [00:45<00:01, 8546.56it/s] 96%|█████████▋| 385744/400000 [00:45<00:01, 8523.27it/s] 97%|█████████▋| 386701/400000 [00:45<00:01, 8809.99it/s] 97%|█████████▋| 387600/400000 [00:45<00:01, 8862.16it/s] 97%|█████████▋| 388498/400000 [00:45<00:01, 8894.07it/s] 97%|█████████▋| 389390/400000 [00:45<00:01, 8693.39it/s] 98%|█████████▊| 390262/400000 [00:45<00:01, 8398.88it/s] 98%|█████████▊| 391106/400000 [00:45<00:01, 8183.81it/s] 98%|█████████▊| 391949/400000 [00:46<00:00, 8255.47it/s] 98%|█████████▊| 392785/400000 [00:46<00:00, 8286.53it/s] 98%|█████████▊| 393616/400000 [00:46<00:00, 8275.39it/s] 99%|█████████▊| 394446/400000 [00:46<00:00, 8182.95it/s] 99%|█████████▉| 395266/400000 [00:46<00:00, 8163.83it/s] 99%|█████████▉| 396094/400000 [00:46<00:00, 8196.57it/s] 99%|█████████▉| 396977/400000 [00:46<00:00, 8374.84it/s] 99%|█████████▉| 397856/400000 [00:46<00:00, 8493.35it/s]100%|█████████▉| 398707/400000 [00:46<00:00, 8186.68it/s]100%|█████████▉| 399530/400000 [00:47<00:00, 7880.70it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8497.65it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fcda2420cc0>, <torchtext.data.dataset.TabularDataset object at 0x7fcda2420e10>, <torchtext.vocab.Vocab object at 0x7fcda2420d30>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.07 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.68 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 23.09 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 23.09 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.74 file/s]2020-07-07 18:18:31.275086: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-07 18:18:31.279358: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-07 18:18:31.279550: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559de5dfa810 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-07 18:18:31.279566: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3538944/9912422 [00:00<00:00, 34198876.98it/s]9920512it [00:00, 34269889.57it/s]                             
0it [00:00, ?it/s]32768it [00:00, 575291.87it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 476136.26it/s]1654784it [00:00, 11553125.37it/s]                         
0it [00:00, ?it/s]8192it [00:00, 185484.68it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
