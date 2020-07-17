
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fadce867a60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fadce867a60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fae394d2510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fae394d2510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fae58721ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fae58721ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fade6805950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fade6805950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fade6805950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▉       | 2883584/9912422 [00:00<00:00, 28588581.01it/s]9920512it [00:00, 32425330.44it/s]                             
0it [00:00, ?it/s]32768it [00:00, 633172.49it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 137570.20it/s]1654784it [00:00, 10248233.16it/s]                         
0it [00:00, ?it/s]8192it [00:00, 225289.08it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fade4d515c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fade4d76860>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fade6805598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fade6805598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fade6805598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:43,  3.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:43,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:42,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:42,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:42,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:42,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:41,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:41,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:41,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:29,  5.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:29,  5.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:29,  5.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:28,  5.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:28,  5.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:28,  5.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:28,  5.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:28,  5.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:19,  7.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:19,  7.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:19,  7.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:19,  7.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:19,  7.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:19,  7.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:19,  7.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:13,  9.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:13,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:13,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:13,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:13,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:09, 13.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:09, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:09, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:06, 18.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:06, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:06, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:06, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:06, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:06, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:06, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:05, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:05, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:05, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:05, 18.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:00<00:04, 24.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:04, 24.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:04, 24.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:04, 24.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:00<00:04, 24.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:00<00:04, 24.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:00<00:04, 24.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:00<00:04, 24.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 24.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 24.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 30.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 38.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 46.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 54.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 61.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 61.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 61.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 61.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 61.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 61.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 61.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 61.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 67.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 72.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 72.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 77.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 80.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 80.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 83.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 83.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 83.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 83.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 83.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 83.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 83.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 83.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 83.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 84.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 84.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.10s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.66 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.25s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 84.66 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.25s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.25s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.14 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.25s/ url]
0 examples [00:00, ? examples/s]2020-07-17 00:08:31.341983: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-17 00:08:31.355069: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-17 00:08:31.355324: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d002460850 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-17 00:08:31.355342: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  4.22 examples/s]103 examples [00:00,  6.02 examples/s]207 examples [00:00,  8.58 examples/s]313 examples [00:00, 12.21 examples/s]416 examples [00:00, 17.36 examples/s]520 examples [00:00, 24.61 examples/s]621 examples [00:00, 34.80 examples/s]721 examples [00:00, 48.98 examples/s]815 examples [00:01, 68.28 examples/s]909 examples [00:01, 94.60 examples/s]1015 examples [00:01, 130.14 examples/s]1126 examples [00:01, 177.01 examples/s]1236 examples [00:01, 236.45 examples/s]1339 examples [00:01, 307.18 examples/s]1442 examples [00:01, 388.99 examples/s]1549 examples [00:01, 480.40 examples/s]1653 examples [00:01, 572.40 examples/s]1757 examples [00:01, 652.82 examples/s]1862 examples [00:02, 735.68 examples/s]1965 examples [00:02, 803.26 examples/s]2068 examples [00:02, 853.55 examples/s]2170 examples [00:02, 892.20 examples/s]2274 examples [00:02, 931.17 examples/s]2379 examples [00:02, 961.65 examples/s]2482 examples [00:02, 976.66 examples/s]2587 examples [00:02, 995.58 examples/s]2690 examples [00:02, 994.32 examples/s]2792 examples [00:02, 970.09 examples/s]2892 examples [00:03, 978.24 examples/s]2995 examples [00:03, 992.68 examples/s]3097 examples [00:03, 1000.48 examples/s]3198 examples [00:03, 1001.45 examples/s]3299 examples [00:03, 988.85 examples/s] 3399 examples [00:03, 976.58 examples/s]3497 examples [00:03, 968.11 examples/s]3600 examples [00:03, 983.89 examples/s]3699 examples [00:03, 980.24 examples/s]3801 examples [00:04, 989.79 examples/s]3905 examples [00:04, 1003.19 examples/s]4007 examples [00:04, 1007.81 examples/s]4110 examples [00:04, 1013.27 examples/s]4212 examples [00:04, 1008.21 examples/s]4316 examples [00:04, 1015.15 examples/s]4418 examples [00:04, 1012.10 examples/s]4520 examples [00:04, 1005.18 examples/s]4621 examples [00:04, 1005.95 examples/s]4722 examples [00:04, 1001.95 examples/s]4824 examples [00:05, 1006.59 examples/s]4925 examples [00:05, 981.34 examples/s] 5024 examples [00:05, 958.80 examples/s]5121 examples [00:05, 885.59 examples/s]5220 examples [00:05, 913.35 examples/s]5317 examples [00:05, 927.76 examples/s]5418 examples [00:05, 950.94 examples/s]5518 examples [00:05, 963.57 examples/s]5615 examples [00:05, 951.07 examples/s]5713 examples [00:05, 959.31 examples/s]5814 examples [00:06, 972.02 examples/s]5917 examples [00:06, 987.47 examples/s]6018 examples [00:06, 991.52 examples/s]6118 examples [00:06, 984.32 examples/s]6218 examples [00:06, 988.48 examples/s]6320 examples [00:06, 995.42 examples/s]6425 examples [00:06, 1005.85 examples/s]6526 examples [00:06, 994.39 examples/s] 6626 examples [00:06, 928.50 examples/s]6729 examples [00:07, 955.12 examples/s]6834 examples [00:07, 981.58 examples/s]6934 examples [00:07, 985.07 examples/s]7034 examples [00:07, 946.57 examples/s]7134 examples [00:07, 960.33 examples/s]7238 examples [00:07, 982.14 examples/s]7342 examples [00:07, 996.72 examples/s]7443 examples [00:07, 980.42 examples/s]7547 examples [00:07, 994.80 examples/s]7647 examples [00:07, 995.99 examples/s]7749 examples [00:08, 1001.69 examples/s]7850 examples [00:08, 995.26 examples/s] 7950 examples [00:08, 982.63 examples/s]8053 examples [00:08, 995.57 examples/s]8154 examples [00:08, 997.07 examples/s]8261 examples [00:08, 1016.85 examples/s]8363 examples [00:08, 1016.11 examples/s]8469 examples [00:08, 1027.51 examples/s]8572 examples [00:08, 1026.81 examples/s]8675 examples [00:08, 1004.54 examples/s]8778 examples [00:09, 1010.41 examples/s]8885 examples [00:09, 1025.48 examples/s]8988 examples [00:09, 982.50 examples/s] 9087 examples [00:09, 974.08 examples/s]9185 examples [00:09, 975.02 examples/s]9286 examples [00:09, 984.53 examples/s]9388 examples [00:09, 993.38 examples/s]9489 examples [00:09, 998.19 examples/s]9594 examples [00:09, 1012.80 examples/s]9696 examples [00:09, 984.29 examples/s] 9795 examples [00:10, 972.83 examples/s]9897 examples [00:10, 985.12 examples/s]9998 examples [00:10, 990.48 examples/s]10098 examples [00:10, 941.98 examples/s]10203 examples [00:10, 969.43 examples/s]10311 examples [00:10, 998.43 examples/s]10417 examples [00:10, 1015.43 examples/s]10520 examples [00:10, 994.99 examples/s] 10625 examples [00:10, 1008.82 examples/s]10730 examples [00:11, 1019.13 examples/s]10833 examples [00:11, 998.89 examples/s] 10934 examples [00:11, 996.05 examples/s]11035 examples [00:11, 997.94 examples/s]11136 examples [00:11, 998.82 examples/s]11236 examples [00:11, 987.46 examples/s]11335 examples [00:11, 973.47 examples/s]11433 examples [00:11, 972.31 examples/s]11532 examples [00:11, 974.73 examples/s]11630 examples [00:11, 948.95 examples/s]11732 examples [00:12, 967.16 examples/s]11839 examples [00:12, 993.89 examples/s]11943 examples [00:12, 1005.80 examples/s]12045 examples [00:12, 1008.43 examples/s]12147 examples [00:12, 985.90 examples/s] 12250 examples [00:12, 998.64 examples/s]12353 examples [00:12, 1007.25 examples/s]12456 examples [00:12, 1011.88 examples/s]12558 examples [00:12, 1006.59 examples/s]12659 examples [00:12, 993.14 examples/s] 12760 examples [00:13, 997.44 examples/s]12864 examples [00:13, 1009.38 examples/s]12968 examples [00:13, 1017.83 examples/s]13072 examples [00:13, 1024.28 examples/s]13175 examples [00:13, 1018.02 examples/s]13277 examples [00:13, 988.33 examples/s] 13382 examples [00:13, 1004.24 examples/s]13486 examples [00:13, 1011.74 examples/s]13592 examples [00:13, 1024.45 examples/s]13695 examples [00:13, 1013.67 examples/s]13797 examples [00:14, 1014.68 examples/s]13901 examples [00:14, 1018.87 examples/s]14004 examples [00:14, 1020.02 examples/s]14107 examples [00:14, 1021.10 examples/s]14210 examples [00:14, 1017.36 examples/s]14312 examples [00:14, 990.81 examples/s] 14412 examples [00:14, 968.73 examples/s]14519 examples [00:14, 996.38 examples/s]14624 examples [00:14, 1009.65 examples/s]14728 examples [00:15, 1018.56 examples/s]14831 examples [00:15, 989.14 examples/s] 14931 examples [00:15, 987.04 examples/s]15034 examples [00:15, 998.94 examples/s]15138 examples [00:15, 1009.61 examples/s]15240 examples [00:15, 1003.99 examples/s]15341 examples [00:15, 997.73 examples/s] 15444 examples [00:15, 1004.31 examples/s]15545 examples [00:15, 1001.66 examples/s]15646 examples [00:15, 990.90 examples/s] 15746 examples [00:16, 975.05 examples/s]15849 examples [00:16, 989.96 examples/s]15954 examples [00:16, 1006.58 examples/s]16055 examples [00:16, 1001.67 examples/s]16156 examples [00:16, 984.63 examples/s] 16255 examples [00:16, 984.98 examples/s]16364 examples [00:16, 1012.01 examples/s]16470 examples [00:16, 1025.21 examples/s]16573 examples [00:16, 1013.38 examples/s]16675 examples [00:16, 981.35 examples/s] 16774 examples [00:17, 970.33 examples/s]16872 examples [00:17, 953.51 examples/s]16975 examples [00:17, 972.60 examples/s]17077 examples [00:17, 983.53 examples/s]17176 examples [00:17, 981.90 examples/s]17275 examples [00:17, 972.01 examples/s]17379 examples [00:17, 989.28 examples/s]17484 examples [00:17, 1004.82 examples/s]17586 examples [00:17, 1008.53 examples/s]17688 examples [00:17, 1010.38 examples/s]17790 examples [00:18, 1001.93 examples/s]17896 examples [00:18, 1018.36 examples/s]17999 examples [00:18, 1019.72 examples/s]18105 examples [00:18, 1028.73 examples/s]18208 examples [00:18, 1019.84 examples/s]18311 examples [00:18, 1009.56 examples/s]18418 examples [00:18, 1025.53 examples/s]18524 examples [00:18, 1034.81 examples/s]18628 examples [00:18, 1031.95 examples/s]18732 examples [00:19, 1020.40 examples/s]18835 examples [00:19, 1006.62 examples/s]18936 examples [00:19, 976.38 examples/s] 19040 examples [00:19, 993.06 examples/s]19148 examples [00:19, 1015.17 examples/s]19256 examples [00:19, 1032.58 examples/s]19360 examples [00:19, 1017.07 examples/s]19462 examples [00:19, 1015.46 examples/s]19568 examples [00:19, 1026.05 examples/s]19671 examples [00:19, 1025.91 examples/s]19775 examples [00:20, 1029.64 examples/s]19879 examples [00:20, 1010.08 examples/s]19982 examples [00:20, 1013.58 examples/s]20084 examples [00:20, 959.29 examples/s] 20188 examples [00:20, 982.05 examples/s]20293 examples [00:20, 1000.37 examples/s]20398 examples [00:20, 1012.46 examples/s]20506 examples [00:20, 1029.93 examples/s]20612 examples [00:20, 1036.83 examples/s]20716 examples [00:20, 1037.74 examples/s]20820 examples [00:21, 1018.35 examples/s]20923 examples [00:21, 1017.63 examples/s]21030 examples [00:21, 1030.60 examples/s]21136 examples [00:21, 1036.89 examples/s]21240 examples [00:21, 1000.50 examples/s]21342 examples [00:21, 1005.19 examples/s]21446 examples [00:21, 1014.22 examples/s]21551 examples [00:21, 1021.90 examples/s]21656 examples [00:21, 1028.37 examples/s]21759 examples [00:21, 1018.66 examples/s]21861 examples [00:22, 1018.69 examples/s]21963 examples [00:22, 993.67 examples/s] 22063 examples [00:22, 909.49 examples/s]22162 examples [00:22, 931.80 examples/s]22265 examples [00:22, 958.65 examples/s]22369 examples [00:22, 981.19 examples/s]22477 examples [00:22, 1007.53 examples/s]22582 examples [00:22, 1018.73 examples/s]22689 examples [00:22, 1032.41 examples/s]22793 examples [00:23, 1030.23 examples/s]22897 examples [00:23, 1030.13 examples/s]23001 examples [00:23, 1031.46 examples/s]23105 examples [00:23, 1026.21 examples/s]23213 examples [00:23, 1039.64 examples/s]23318 examples [00:23, 1007.43 examples/s]23420 examples [00:23, 982.54 examples/s] 23519 examples [00:23, 983.51 examples/s]23618 examples [00:23, 978.05 examples/s]23716 examples [00:23, 977.09 examples/s]23814 examples [00:24, 976.28 examples/s]23917 examples [00:24, 991.76 examples/s]24018 examples [00:24, 995.38 examples/s]24121 examples [00:24, 1005.39 examples/s]24223 examples [00:24, 1008.59 examples/s]24327 examples [00:24, 1014.15 examples/s]24429 examples [00:24, 982.18 examples/s] 24529 examples [00:24, 986.41 examples/s]24628 examples [00:24, 987.03 examples/s]24727 examples [00:24, 963.74 examples/s]24827 examples [00:25, 972.09 examples/s]24925 examples [00:25, 973.85 examples/s]25023 examples [00:25, 973.71 examples/s]25123 examples [00:25, 979.02 examples/s]25226 examples [00:25, 992.30 examples/s]25326 examples [00:25, 988.87 examples/s]25425 examples [00:25, 987.03 examples/s]25524 examples [00:25, 983.13 examples/s]25626 examples [00:25, 993.15 examples/s]25726 examples [00:26, 993.24 examples/s]25826 examples [00:26, 950.87 examples/s]25922 examples [00:26, 928.72 examples/s]26021 examples [00:26, 945.79 examples/s]26121 examples [00:26, 959.37 examples/s]26223 examples [00:26, 974.99 examples/s]26326 examples [00:26, 987.68 examples/s]26428 examples [00:26, 995.13 examples/s]26532 examples [00:26, 1008.01 examples/s]26633 examples [00:26, 989.77 examples/s] 26734 examples [00:27, 993.68 examples/s]26837 examples [00:27, 1002.14 examples/s]26938 examples [00:27, 976.23 examples/s] 27041 examples [00:27, 989.62 examples/s]27141 examples [00:27, 984.76 examples/s]27240 examples [00:27, 955.55 examples/s]27340 examples [00:27, 967.86 examples/s]27441 examples [00:27, 979.13 examples/s]27542 examples [00:27, 985.85 examples/s]27646 examples [00:27, 999.51 examples/s]27747 examples [00:28, 1001.55 examples/s]27851 examples [00:28, 1011.04 examples/s]27954 examples [00:28, 1014.80 examples/s]28059 examples [00:28, 1023.28 examples/s]28163 examples [00:28, 1026.30 examples/s]28266 examples [00:28, 1025.99 examples/s]28369 examples [00:28, 1021.22 examples/s]28472 examples [00:28, 1019.19 examples/s]28574 examples [00:28, 1013.21 examples/s]28676 examples [00:28, 1009.87 examples/s]28778 examples [00:29, 997.32 examples/s] 28882 examples [00:29, 1005.99 examples/s]28983 examples [00:29, 1001.64 examples/s]29084 examples [00:29, 955.71 examples/s] 29184 examples [00:29, 966.53 examples/s]29282 examples [00:29, 966.95 examples/s]29382 examples [00:29, 974.34 examples/s]29486 examples [00:29, 991.31 examples/s]29588 examples [00:29, 998.04 examples/s]29691 examples [00:30, 1005.89 examples/s]29792 examples [00:30, 1007.04 examples/s]29893 examples [00:30, 1000.12 examples/s]29994 examples [00:30, 990.24 examples/s] 30094 examples [00:30, 939.81 examples/s]30197 examples [00:30, 964.39 examples/s]30296 examples [00:30, 969.66 examples/s]30399 examples [00:30, 983.80 examples/s]30498 examples [00:30, 949.97 examples/s]30600 examples [00:30, 967.57 examples/s]30698 examples [00:31, 967.80 examples/s]30798 examples [00:31, 976.09 examples/s]30902 examples [00:31, 993.84 examples/s]31008 examples [00:31, 1012.27 examples/s]31113 examples [00:31, 1021.95 examples/s]31216 examples [00:31, 1017.40 examples/s]31322 examples [00:31, 1029.12 examples/s]31426 examples [00:31, 1026.04 examples/s]31529 examples [00:31, 1009.11 examples/s]31631 examples [00:31, 972.85 examples/s] 31733 examples [00:32, 985.07 examples/s]31838 examples [00:32, 1003.56 examples/s]31939 examples [00:32, 992.16 examples/s] 32039 examples [00:32, 981.76 examples/s]32143 examples [00:32, 995.92 examples/s]32243 examples [00:32, 985.35 examples/s]32342 examples [00:32, 969.17 examples/s]32441 examples [00:32, 974.01 examples/s]32539 examples [00:32, 912.63 examples/s]32640 examples [00:33, 938.00 examples/s]32736 examples [00:33, 941.77 examples/s]32831 examples [00:33, 937.75 examples/s]32928 examples [00:33, 946.72 examples/s]33026 examples [00:33, 956.43 examples/s]33129 examples [00:33, 976.70 examples/s]33227 examples [00:33, 973.83 examples/s]33325 examples [00:33, 961.75 examples/s]33426 examples [00:33, 974.23 examples/s]33524 examples [00:33, 933.27 examples/s]33618 examples [00:34, 890.40 examples/s]33712 examples [00:34, 903.65 examples/s]33803 examples [00:34, 904.84 examples/s]33894 examples [00:34, 888.47 examples/s]33987 examples [00:34, 900.35 examples/s]34081 examples [00:34, 910.96 examples/s]34179 examples [00:34, 930.22 examples/s]34273 examples [00:34, 916.31 examples/s]34365 examples [00:34, 901.48 examples/s]34459 examples [00:34, 911.23 examples/s]34553 examples [00:35, 919.55 examples/s]34646 examples [00:35, 920.22 examples/s]34749 examples [00:35, 949.88 examples/s]34845 examples [00:35, 937.19 examples/s]34943 examples [00:35, 947.67 examples/s]35038 examples [00:35, 936.51 examples/s]35135 examples [00:35, 944.34 examples/s]35235 examples [00:35, 958.89 examples/s]35333 examples [00:35, 963.28 examples/s]35435 examples [00:35, 979.15 examples/s]35537 examples [00:36, 989.17 examples/s]35637 examples [00:36, 984.16 examples/s]35737 examples [00:36, 986.32 examples/s]35845 examples [00:36, 1010.86 examples/s]35953 examples [00:36, 1030.39 examples/s]36058 examples [00:36, 1035.27 examples/s]36162 examples [00:36, 1033.72 examples/s]36266 examples [00:36, 1029.95 examples/s]36370 examples [00:36, 1024.89 examples/s]36473 examples [00:37, 955.24 examples/s] 36570 examples [00:37, 957.22 examples/s]36667 examples [00:37, 950.66 examples/s]36763 examples [00:37, 937.94 examples/s]36867 examples [00:37, 965.75 examples/s]36967 examples [00:37, 974.17 examples/s]37068 examples [00:37, 983.95 examples/s]37173 examples [00:37, 1002.32 examples/s]37275 examples [00:37, 1003.95 examples/s]37376 examples [00:37, 961.62 examples/s] 37484 examples [00:38, 991.66 examples/s]37586 examples [00:38, 999.80 examples/s]37687 examples [00:38, 999.12 examples/s]37788 examples [00:38, 981.03 examples/s]37887 examples [00:38, 982.90 examples/s]37991 examples [00:38, 999.11 examples/s]38094 examples [00:38, 1006.10 examples/s]38195 examples [00:38, 997.08 examples/s] 38295 examples [00:38, 991.85 examples/s]38395 examples [00:38, 990.66 examples/s]38499 examples [00:39, 1003.29 examples/s]38601 examples [00:39, 1007.05 examples/s]38702 examples [00:39, 1004.57 examples/s]38803 examples [00:39, 986.52 examples/s] 38910 examples [00:39, 1008.74 examples/s]39013 examples [00:39, 1013.34 examples/s]39115 examples [00:39, 1011.75 examples/s]39217 examples [00:39, 1010.99 examples/s]39319 examples [00:39, 1010.41 examples/s]39421 examples [00:39, 1008.32 examples/s]39523 examples [00:40, 1010.06 examples/s]39625 examples [00:40, 992.70 examples/s] 39725 examples [00:40, 987.75 examples/s]39824 examples [00:40, 980.70 examples/s]39923 examples [00:40, 935.19 examples/s]40018 examples [00:40, 908.73 examples/s]40126 examples [00:40, 953.30 examples/s]40226 examples [00:40, 965.83 examples/s]40331 examples [00:40, 988.34 examples/s]40434 examples [00:41, 999.33 examples/s]40535 examples [00:41, 994.91 examples/s]40635 examples [00:41, 989.29 examples/s]40737 examples [00:41, 997.64 examples/s]40837 examples [00:41, 995.20 examples/s]40937 examples [00:41, 992.47 examples/s]41037 examples [00:41, 983.27 examples/s]41136 examples [00:41, 980.86 examples/s]41238 examples [00:41, 990.53 examples/s]41338 examples [00:41, 993.32 examples/s]41439 examples [00:42, 995.59 examples/s]41539 examples [00:42, 988.53 examples/s]41638 examples [00:42, 982.02 examples/s]41737 examples [00:42, 982.77 examples/s]41840 examples [00:42, 994.30 examples/s]41940 examples [00:42, 994.22 examples/s]42040 examples [00:42, 949.86 examples/s]42143 examples [00:42, 970.73 examples/s]42246 examples [00:42, 985.60 examples/s]42345 examples [00:42, 985.60 examples/s]42445 examples [00:43, 988.15 examples/s]42545 examples [00:43, 990.97 examples/s]42650 examples [00:43, 1005.94 examples/s]42751 examples [00:43, 997.02 examples/s] 42851 examples [00:43, 997.66 examples/s]42952 examples [00:43, 998.86 examples/s]43052 examples [00:43, 991.40 examples/s]43152 examples [00:43, 989.92 examples/s]43252 examples [00:43, 987.73 examples/s]43351 examples [00:43, 986.48 examples/s]43453 examples [00:44, 995.95 examples/s]43556 examples [00:44, 1003.57 examples/s]43657 examples [00:44, 992.69 examples/s] 43757 examples [00:44, 983.64 examples/s]43857 examples [00:44, 987.84 examples/s]43959 examples [00:44, 996.76 examples/s]44059 examples [00:44, 993.04 examples/s]44163 examples [00:44, 1004.70 examples/s]44264 examples [00:44, 1001.99 examples/s]44367 examples [00:45, 1009.33 examples/s]44469 examples [00:45, 1012.04 examples/s]44571 examples [00:45, 1013.83 examples/s]44674 examples [00:45, 1015.29 examples/s]44776 examples [00:45, 1002.32 examples/s]44877 examples [00:45, 982.40 examples/s] 44976 examples [00:45, 975.72 examples/s]45081 examples [00:45, 996.13 examples/s]45181 examples [00:45, 977.01 examples/s]45281 examples [00:45, 983.05 examples/s]45380 examples [00:46, 973.25 examples/s]45484 examples [00:46, 990.53 examples/s]45588 examples [00:46, 1002.92 examples/s]45691 examples [00:46, 1010.44 examples/s]45796 examples [00:46, 1020.96 examples/s]45899 examples [00:46, 1018.33 examples/s]46001 examples [00:46, 1013.85 examples/s]46103 examples [00:46, 1012.96 examples/s]46209 examples [00:46, 1025.02 examples/s]46312 examples [00:46, 1015.11 examples/s]46414 examples [00:47, 1010.58 examples/s]46516 examples [00:47, 997.29 examples/s] 46616 examples [00:47, 935.12 examples/s]46720 examples [00:47, 964.21 examples/s]46818 examples [00:47, 966.04 examples/s]46917 examples [00:47, 972.76 examples/s]47019 examples [00:47, 985.44 examples/s]47118 examples [00:47, 974.18 examples/s]47219 examples [00:47, 983.77 examples/s]47318 examples [00:47, 977.45 examples/s]47416 examples [00:48, 972.03 examples/s]47514 examples [00:48, 966.32 examples/s]47617 examples [00:48, 980.78 examples/s]47718 examples [00:48, 988.56 examples/s]47817 examples [00:48, 986.97 examples/s]47926 examples [00:48, 1014.11 examples/s]48032 examples [00:48, 1025.31 examples/s]48135 examples [00:48, 1001.82 examples/s]48236 examples [00:48, 984.82 examples/s] 48335 examples [00:49, 974.21 examples/s]48434 examples [00:49, 976.57 examples/s]48538 examples [00:49, 991.56 examples/s]48641 examples [00:49, 1002.03 examples/s]48742 examples [00:49, 994.24 examples/s] 48842 examples [00:49, 978.57 examples/s]48940 examples [00:49, 970.85 examples/s]49044 examples [00:49, 989.48 examples/s]49150 examples [00:49, 1008.70 examples/s]49253 examples [00:49, 1013.14 examples/s]49355 examples [00:50, 999.69 examples/s] 49458 examples [00:50, 1006.44 examples/s]49562 examples [00:50, 1014.04 examples/s]49668 examples [00:50, 1024.96 examples/s]49774 examples [00:50, 1031.96 examples/s]49878 examples [00:50, 966.52 examples/s] 49976 examples [00:50, 962.40 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 8866/50000 [00:00<00:00, 88653.11 examples/s] 46%|████▌     | 23019/50000 [00:00<00:00, 99843.10 examples/s] 72%|███████▏  | 36149/50000 [00:00<00:00, 107574.60 examples/s] 98%|█████████▊| 48788/50000 [00:00<00:00, 112603.39 examples/s]                                                                0 examples [00:00, ? examples/s]81 examples [00:00, 805.79 examples/s]186 examples [00:00, 864.65 examples/s]289 examples [00:00, 907.11 examples/s]389 examples [00:00, 932.03 examples/s]488 examples [00:00, 947.48 examples/s]579 examples [00:00, 935.85 examples/s]686 examples [00:00, 971.14 examples/s]792 examples [00:00, 994.41 examples/s]894 examples [00:00, 1000.40 examples/s]996 examples [00:01, 1005.98 examples/s]1095 examples [00:01, 998.28 examples/s]1194 examples [00:01, 994.87 examples/s]1296 examples [00:01, 1001.54 examples/s]1396 examples [00:01, 992.87 examples/s] 1499 examples [00:01, 1001.37 examples/s]1603 examples [00:01, 1011.25 examples/s]1705 examples [00:01, 1013.48 examples/s]1809 examples [00:01, 1019.59 examples/s]1911 examples [00:01, 1013.67 examples/s]2015 examples [00:02, 1019.17 examples/s]2117 examples [00:02, 1013.87 examples/s]2219 examples [00:02, 1001.92 examples/s]2320 examples [00:02, 1000.88 examples/s]2421 examples [00:02, 986.24 examples/s] 2520 examples [00:02, 978.37 examples/s]2620 examples [00:02, 984.11 examples/s]2719 examples [00:02, 924.51 examples/s]2813 examples [00:02, 922.91 examples/s]2912 examples [00:02, 940.62 examples/s]3011 examples [00:03, 953.89 examples/s]3107 examples [00:03, 953.82 examples/s]3206 examples [00:03, 963.95 examples/s]3308 examples [00:03, 979.25 examples/s]3409 examples [00:03, 987.07 examples/s]3510 examples [00:03, 991.94 examples/s]3611 examples [00:03, 995.37 examples/s]3719 examples [00:03, 1019.29 examples/s]3825 examples [00:03, 1028.89 examples/s]3929 examples [00:03, 1024.12 examples/s]4035 examples [00:04, 1032.85 examples/s]4139 examples [00:04, 1024.55 examples/s]4245 examples [00:04, 1033.19 examples/s]4349 examples [00:04, 1027.66 examples/s]4452 examples [00:04, 1005.60 examples/s]4553 examples [00:04, 990.49 examples/s] 4653 examples [00:04, 981.25 examples/s]4758 examples [00:04, 999.24 examples/s]4859 examples [00:04, 997.62 examples/s]4959 examples [00:04, 996.85 examples/s]5065 examples [00:05, 1012.66 examples/s]5175 examples [00:05, 1036.83 examples/s]5281 examples [00:05, 1042.37 examples/s]5386 examples [00:05, 1039.44 examples/s]5491 examples [00:05, 1027.03 examples/s]5594 examples [00:05, 1026.38 examples/s]5698 examples [00:05, 1030.35 examples/s]5802 examples [00:05, 1031.05 examples/s]5909 examples [00:05, 1040.62 examples/s]6014 examples [00:06, 1035.26 examples/s]6118 examples [00:06, 995.06 examples/s] 6218 examples [00:06, 990.67 examples/s]6318 examples [00:06, 988.86 examples/s]6418 examples [00:06, 951.91 examples/s]6518 examples [00:06, 964.29 examples/s]6623 examples [00:06, 988.08 examples/s]6728 examples [00:06, 1004.15 examples/s]6829 examples [00:06, 1002.37 examples/s]6938 examples [00:06, 1026.65 examples/s]7046 examples [00:07, 1039.70 examples/s]7154 examples [00:07, 1051.19 examples/s]7260 examples [00:07, 1014.68 examples/s]7362 examples [00:07, 995.05 examples/s] 7465 examples [00:07, 1003.01 examples/s]7566 examples [00:07, 1003.95 examples/s]7672 examples [00:07, 1019.55 examples/s]7776 examples [00:07, 1025.17 examples/s]7879 examples [00:07, 997.45 examples/s] 7980 examples [00:07, 968.16 examples/s]8082 examples [00:08, 981.74 examples/s]8187 examples [00:08, 998.90 examples/s]8288 examples [00:08, 995.99 examples/s]8388 examples [00:08, 992.23 examples/s]8488 examples [00:08, 987.69 examples/s]8587 examples [00:08, 979.83 examples/s]8694 examples [00:08, 1002.93 examples/s]8795 examples [00:08, 997.30 examples/s] 8900 examples [00:08, 1011.36 examples/s]9002 examples [00:08, 1007.69 examples/s]9104 examples [00:09, 1009.88 examples/s]9212 examples [00:09, 1027.77 examples/s]9318 examples [00:09, 1036.10 examples/s]9424 examples [00:09, 1041.53 examples/s]9529 examples [00:09, 1035.69 examples/s]9633 examples [00:09, 1005.34 examples/s]9740 examples [00:09, 1021.66 examples/s]9843 examples [00:09, 1022.73 examples/s]9947 examples [00:09, 1024.63 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1IKN0V/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1IKN0V/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fade6805950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fade6805950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fade6805950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fada4bdfeb8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fada4b782e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fade6805950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fade6805950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fade6805950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fada4b782e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fade4d760f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fad5f1a0ea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fad5f1a0ea0> 

  function with postional parmater data_info <function split_train_valid at 0x7fad5f1a0ea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fad5f227048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fad5f227048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fad5f227048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=16c1a2d849d68bee8f44e706f4cabf8974e2d30b4e5c89419f6c449d6a0d34ca
  Stored in directory: /tmp/pip-ephem-wheel-cache-5n7az3ws/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip: 0.00B [01:19, ?B/s]Preprocessing the text...
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.75 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.51 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.66 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.38 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.38 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.59 file/s]2020-07-17 00:11:10.971028: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-17 00:11:10.974762: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-17 00:11:10.974912: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5607006689e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-17 00:11:10.974928: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:22, 429050.70it/s]9920512it [00:00, 47986856.45it/s]                         
0it [00:00, ?it/s]32768it [00:00, 557037.41it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1038995.42it/s]1654784it [00:00, 12194580.00it/s]                           
0it [00:00, ?it/s]8192it [00:00, 221266.03it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
