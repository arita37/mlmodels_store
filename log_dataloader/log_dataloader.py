
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f061ca88a60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f061ca88a60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f06876f3510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f06876f3510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f06a6942ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f06a6942ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0634a24950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0634a24950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0634a24950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 472204.69it/s] 64%|██████▍   | 6348800/9912422 [00:00<00:05, 672379.06it/s]9920512it [00:00, 39181082.15it/s]                           
0it [00:00, ?it/s]32768it [00:00, 627603.79it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1048216.74it/s]1654784it [00:00, 12770598.71it/s]                           
0it [00:00, ?it/s]8192it [00:00, 251679.13it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0632f895c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0632f97860>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0634a24598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0634a24598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f0634a24598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:24,  1.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:24,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:23,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:23,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:22,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:22,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:21,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:21,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:57,  2.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:57,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:56,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:56,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:56,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:55,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:55,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:54,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:54,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:54,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:53,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:53,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:37,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:37,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:36,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:36,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:36,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:36,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:35,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:35,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:35,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:34,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:24,  5.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:24,  5.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:24,  5.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:23,  5.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:23,  5.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:23,  5.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:23,  5.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:23,  5.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:16,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:16,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:16,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:15,  7.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:15,  7.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:15,  7.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 18.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 18.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 18.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 18.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 18.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 18.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 18.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 24.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 24.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 24.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 24.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 24.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 30.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 45.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 45.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 45.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 45.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 45.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 45.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 45.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 45.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 53.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 53.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 53.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 53.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 53.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 53.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 53.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 53.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 53.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 53.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 59.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 59.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 59.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 59.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 59.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 59.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 59.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 59.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.39s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.39s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.39s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.42 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.66s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.39s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.42 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.66s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.66s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.66s/ url]
0 examples [00:00, ? examples/s]2020-07-17 18:10:58.986268: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-17 18:10:59.029713: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-07-17 18:10:59.029977: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f9808ea8a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-17 18:10:59.030004: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.13 examples/s]104 examples [00:00,  1.62 examples/s]208 examples [00:01,  2.31 examples/s]312 examples [00:01,  3.29 examples/s]416 examples [00:01,  4.70 examples/s]516 examples [00:01,  6.70 examples/s]620 examples [00:01,  9.54 examples/s]725 examples [00:01, 13.58 examples/s]825 examples [00:01, 19.28 examples/s]925 examples [00:01, 27.32 examples/s]1023 examples [00:01, 38.54 examples/s]1119 examples [00:02, 54.05 examples/s]1215 examples [00:02, 75.39 examples/s]1310 examples [00:02, 104.07 examples/s]1409 examples [00:02, 142.23 examples/s]1507 examples [00:02, 191.25 examples/s]1608 examples [00:02, 252.56 examples/s]1712 examples [00:02, 326.69 examples/s]1812 examples [00:02, 406.13 examples/s]1910 examples [00:02, 489.21 examples/s]2008 examples [00:02, 575.29 examples/s]2109 examples [00:03, 660.22 examples/s]2214 examples [00:03, 741.68 examples/s]2316 examples [00:03, 807.31 examples/s]2421 examples [00:03, 865.51 examples/s]2524 examples [00:03, 908.14 examples/s]2626 examples [00:03, 913.24 examples/s]2728 examples [00:03, 942.83 examples/s]2828 examples [00:03, 926.41 examples/s]2934 examples [00:03, 961.46 examples/s]3037 examples [00:03, 979.74 examples/s]3138 examples [00:04, 986.86 examples/s]3239 examples [00:04, 989.50 examples/s]3341 examples [00:04, 997.31 examples/s]3446 examples [00:04, 1011.23 examples/s]3552 examples [00:04, 1023.69 examples/s]3658 examples [00:04, 1032.09 examples/s]3762 examples [00:04, 1033.07 examples/s]3866 examples [00:04, 1027.17 examples/s]3969 examples [00:04, 1025.82 examples/s]4074 examples [00:04, 1030.65 examples/s]4178 examples [00:05, 993.59 examples/s] 4281 examples [00:05, 1003.02 examples/s]4382 examples [00:05, 995.57 examples/s] 4484 examples [00:05, 1000.64 examples/s]4585 examples [00:05, 982.41 examples/s] 4688 examples [00:05, 994.34 examples/s]4789 examples [00:05, 997.52 examples/s]4893 examples [00:05, 1007.92 examples/s]4994 examples [00:05, 995.72 examples/s] 5094 examples [00:05, 974.52 examples/s]5192 examples [00:06, 969.03 examples/s]5290 examples [00:06, 968.36 examples/s]5391 examples [00:06, 978.68 examples/s]5495 examples [00:06, 993.88 examples/s]5595 examples [00:06, 992.34 examples/s]5697 examples [00:06, 998.04 examples/s]5798 examples [00:06, 1001.59 examples/s]5901 examples [00:06, 1007.84 examples/s]6003 examples [00:06, 1011.34 examples/s]6105 examples [00:07, 1003.44 examples/s]6206 examples [00:07, 997.69 examples/s] 6306 examples [00:07, 988.74 examples/s]6405 examples [00:07, 957.37 examples/s]6508 examples [00:07, 975.85 examples/s]6607 examples [00:07, 979.18 examples/s]6706 examples [00:07, 981.40 examples/s]6806 examples [00:07, 986.31 examples/s]6905 examples [00:07, 987.36 examples/s]7008 examples [00:07, 998.27 examples/s]7111 examples [00:08, 1005.22 examples/s]7215 examples [00:08, 1015.23 examples/s]7320 examples [00:08, 1022.56 examples/s]7423 examples [00:08, 998.16 examples/s] 7526 examples [00:08, 1005.88 examples/s]7627 examples [00:08, 1007.03 examples/s]7728 examples [00:08, 1000.70 examples/s]7829 examples [00:08, 982.01 examples/s] 7928 examples [00:08, 961.44 examples/s]8025 examples [00:08, 943.43 examples/s]8122 examples [00:09, 949.37 examples/s]8227 examples [00:09, 976.21 examples/s]8330 examples [00:09, 990.51 examples/s]8434 examples [00:09, 1002.36 examples/s]8538 examples [00:09, 1012.86 examples/s]8643 examples [00:09, 1022.89 examples/s]8750 examples [00:09, 1033.94 examples/s]8854 examples [00:09, 1024.69 examples/s]8958 examples [00:09, 1028.28 examples/s]9063 examples [00:09, 1034.47 examples/s]9167 examples [00:10, 1029.65 examples/s]9271 examples [00:10, 1015.61 examples/s]9373 examples [00:10, 1002.59 examples/s]9474 examples [00:10, 1001.27 examples/s]9577 examples [00:10, 1007.51 examples/s]9679 examples [00:10, 1007.09 examples/s]9780 examples [00:10, 959.80 examples/s] 9877 examples [00:10, 940.61 examples/s]9979 examples [00:10, 962.44 examples/s]10076 examples [00:11, 928.09 examples/s]10178 examples [00:11, 951.14 examples/s]10282 examples [00:11, 974.48 examples/s]10384 examples [00:11, 985.04 examples/s]10488 examples [00:11, 999.84 examples/s]10591 examples [00:11, 1008.63 examples/s]10696 examples [00:11, 1019.65 examples/s]10799 examples [00:11, 1020.40 examples/s]10902 examples [00:11, 997.45 examples/s] 11002 examples [00:11, 976.17 examples/s]11104 examples [00:12, 986.48 examples/s]11203 examples [00:12, 957.34 examples/s]11301 examples [00:12, 962.84 examples/s]11399 examples [00:12, 967.87 examples/s]11499 examples [00:12, 976.12 examples/s]11603 examples [00:12, 993.97 examples/s]11704 examples [00:12, 998.28 examples/s]11810 examples [00:12, 1014.15 examples/s]11912 examples [00:12, 1003.50 examples/s]12013 examples [00:12, 1003.80 examples/s]12116 examples [00:13, 1008.15 examples/s]12217 examples [00:13, 1000.94 examples/s]12320 examples [00:13, 1007.64 examples/s]12423 examples [00:13, 1014.16 examples/s]12528 examples [00:13, 1024.39 examples/s]12633 examples [00:13, 1029.85 examples/s]12737 examples [00:13, 1030.99 examples/s]12842 examples [00:13, 1034.00 examples/s]12946 examples [00:13, 969.37 examples/s] 13052 examples [00:13, 992.52 examples/s]13157 examples [00:14, 1008.22 examples/s]13259 examples [00:14, 995.38 examples/s] 13365 examples [00:14, 1012.49 examples/s]13469 examples [00:14, 1018.31 examples/s]13572 examples [00:14, 1017.16 examples/s]13678 examples [00:14, 1027.39 examples/s]13781 examples [00:14, 1025.18 examples/s]13884 examples [00:14, 1025.40 examples/s]13987 examples [00:14, 1019.61 examples/s]14091 examples [00:15, 1025.16 examples/s]14196 examples [00:15, 1029.90 examples/s]14300 examples [00:15, 1007.25 examples/s]14401 examples [00:15, 985.25 examples/s] 14505 examples [00:15, 998.85 examples/s]14610 examples [00:15, 1011.56 examples/s]14716 examples [00:15, 1025.57 examples/s]14821 examples [00:15, 1032.53 examples/s]14925 examples [00:15, 1027.41 examples/s]15028 examples [00:15, 1015.54 examples/s]15130 examples [00:16, 990.60 examples/s] 15234 examples [00:16, 1002.96 examples/s]15335 examples [00:16, 1001.38 examples/s]15441 examples [00:16, 1015.96 examples/s]15543 examples [00:16, 1003.86 examples/s]15645 examples [00:16, 1008.23 examples/s]15750 examples [00:16, 1018.87 examples/s]15856 examples [00:16, 1028.15 examples/s]15959 examples [00:16, 1014.16 examples/s]16061 examples [00:16, 1013.10 examples/s]16163 examples [00:17, 968.41 examples/s] 16267 examples [00:17, 987.95 examples/s]16372 examples [00:17, 1005.06 examples/s]16475 examples [00:17, 1011.24 examples/s]16577 examples [00:17, 1008.46 examples/s]16679 examples [00:17, 1005.40 examples/s]16783 examples [00:17, 1015.50 examples/s]16887 examples [00:17, 1022.03 examples/s]16990 examples [00:17, 1024.08 examples/s]17093 examples [00:17, 1024.59 examples/s]17196 examples [00:18, 1025.75 examples/s]17299 examples [00:18, 1013.04 examples/s]17404 examples [00:18, 1022.02 examples/s]17509 examples [00:18, 1027.97 examples/s]17613 examples [00:18, 1029.88 examples/s]17717 examples [00:18, 1026.05 examples/s]17820 examples [00:18, 1016.06 examples/s]17923 examples [00:18, 1018.39 examples/s]18028 examples [00:18, 1025.89 examples/s]18131 examples [00:18, 1018.08 examples/s]18233 examples [00:19, 1015.38 examples/s]18336 examples [00:19, 1016.92 examples/s]18438 examples [00:19, 1010.26 examples/s]18540 examples [00:19, 1011.84 examples/s]18643 examples [00:19, 1016.15 examples/s]18745 examples [00:19, 1017.08 examples/s]18847 examples [00:19, 1016.91 examples/s]18949 examples [00:19, 983.08 examples/s] 19050 examples [00:19, 989.82 examples/s]19150 examples [00:20, 970.61 examples/s]19253 examples [00:20, 986.06 examples/s]19354 examples [00:20, 992.87 examples/s]19454 examples [00:20, 981.06 examples/s]19554 examples [00:20, 984.13 examples/s]19657 examples [00:20, 994.86 examples/s]19759 examples [00:20, 999.95 examples/s]19861 examples [00:20, 1002.03 examples/s]19962 examples [00:20, 1001.19 examples/s]20063 examples [00:20, 945.98 examples/s] 20159 examples [00:21, 940.99 examples/s]20254 examples [00:21, 915.16 examples/s]20360 examples [00:21, 953.95 examples/s]20465 examples [00:21, 978.95 examples/s]20564 examples [00:21, 972.58 examples/s]20664 examples [00:21, 980.48 examples/s]20764 examples [00:21, 984.86 examples/s]20868 examples [00:21, 1000.16 examples/s]20971 examples [00:21, 1008.63 examples/s]21075 examples [00:21, 1015.77 examples/s]21178 examples [00:22, 1017.73 examples/s]21280 examples [00:22, 966.30 examples/s] 21382 examples [00:22, 981.24 examples/s]21486 examples [00:22, 997.84 examples/s]21588 examples [00:22, 1001.18 examples/s]21689 examples [00:22, 988.01 examples/s] 21789 examples [00:22, 970.86 examples/s]21894 examples [00:22, 992.71 examples/s]22001 examples [00:22, 1012.26 examples/s]22107 examples [00:22, 1025.32 examples/s]22210 examples [00:23, 931.51 examples/s] 22305 examples [00:23, 882.96 examples/s]22396 examples [00:23, 888.17 examples/s]22492 examples [00:23, 905.65 examples/s]22593 examples [00:23, 933.16 examples/s]22695 examples [00:23, 956.09 examples/s]22795 examples [00:23, 966.47 examples/s]22893 examples [00:23, 949.12 examples/s]22996 examples [00:23, 971.39 examples/s]23098 examples [00:24, 985.20 examples/s]23200 examples [00:24, 991.13 examples/s]23303 examples [00:24, 999.68 examples/s]23406 examples [00:24, 1007.16 examples/s]23511 examples [00:24, 1016.26 examples/s]23613 examples [00:24, 1002.83 examples/s]23717 examples [00:24, 1011.45 examples/s]23823 examples [00:24, 1023.04 examples/s]23928 examples [00:24, 1028.11 examples/s]24031 examples [00:24, 1005.74 examples/s]24134 examples [00:25, 1011.92 examples/s]24237 examples [00:25, 1014.87 examples/s]24339 examples [00:25, 1005.35 examples/s]24443 examples [00:25, 1012.51 examples/s]24545 examples [00:25, 1013.68 examples/s]24647 examples [00:25, 1000.42 examples/s]24748 examples [00:25, 985.77 examples/s] 24852 examples [00:25, 1000.09 examples/s]24958 examples [00:25, 1015.01 examples/s]25060 examples [00:26, 1004.32 examples/s]25161 examples [00:26, 968.58 examples/s] 25261 examples [00:26, 975.82 examples/s]25368 examples [00:26, 999.72 examples/s]25474 examples [00:26, 1016.86 examples/s]25576 examples [00:26, 1016.77 examples/s]25679 examples [00:26, 1020.48 examples/s]25782 examples [00:26, 1021.21 examples/s]25885 examples [00:26, 983.22 examples/s] 25987 examples [00:26, 993.65 examples/s]26093 examples [00:27, 1010.85 examples/s]26196 examples [00:27, 1013.82 examples/s]26298 examples [00:27, 1013.42 examples/s]26400 examples [00:27, 970.45 examples/s] 26498 examples [00:27, 972.08 examples/s]26596 examples [00:27, 957.27 examples/s]26693 examples [00:27, 927.93 examples/s]26790 examples [00:27, 939.15 examples/s]26885 examples [00:27, 940.50 examples/s]26987 examples [00:27, 960.49 examples/s]27090 examples [00:28, 978.34 examples/s]27191 examples [00:28, 985.65 examples/s]27293 examples [00:28, 995.16 examples/s]27396 examples [00:28, 1004.79 examples/s]27498 examples [00:28, 1006.36 examples/s]27599 examples [00:28, 987.31 examples/s] 27698 examples [00:28, 980.40 examples/s]27800 examples [00:28, 990.65 examples/s]27902 examples [00:28, 996.99 examples/s]28002 examples [00:28, 993.86 examples/s]28105 examples [00:29, 1003.44 examples/s]28206 examples [00:29, 983.05 examples/s] 28309 examples [00:29, 993.83 examples/s]28409 examples [00:29, 995.02 examples/s]28509 examples [00:29, 985.94 examples/s]28612 examples [00:29, 996.19 examples/s]28713 examples [00:29, 997.00 examples/s]28815 examples [00:29, 1003.54 examples/s]28917 examples [00:29, 1006.17 examples/s]29020 examples [00:30, 1012.98 examples/s]29122 examples [00:30, 1014.59 examples/s]29224 examples [00:30, 1002.90 examples/s]29325 examples [00:30, 999.82 examples/s] 29427 examples [00:30, 1004.41 examples/s]29528 examples [00:30, 996.31 examples/s] 29630 examples [00:30, 1003.11 examples/s]29731 examples [00:30, 1003.01 examples/s]29834 examples [00:30, 1007.46 examples/s]29936 examples [00:30, 1008.95 examples/s]30037 examples [00:31, 931.05 examples/s] 30137 examples [00:31, 949.80 examples/s]30236 examples [00:31, 959.53 examples/s]30341 examples [00:31, 982.95 examples/s]30440 examples [00:31, 968.68 examples/s]30539 examples [00:31, 973.11 examples/s]30637 examples [00:31, 962.46 examples/s]30737 examples [00:31, 970.86 examples/s]30838 examples [00:31, 981.06 examples/s]30939 examples [00:31, 988.99 examples/s]31043 examples [00:32, 1002.43 examples/s]31148 examples [00:32, 1014.55 examples/s]31251 examples [00:32, 1019.01 examples/s]31357 examples [00:32, 1029.51 examples/s]31463 examples [00:32, 1035.74 examples/s]31569 examples [00:32, 1041.15 examples/s]31675 examples [00:32, 1044.33 examples/s]31780 examples [00:32, 1044.12 examples/s]31885 examples [00:32, 1030.00 examples/s]31989 examples [00:32, 1024.22 examples/s]32094 examples [00:33, 1030.55 examples/s]32199 examples [00:33, 1033.55 examples/s]32303 examples [00:33, 999.21 examples/s] 32408 examples [00:33, 1011.82 examples/s]32511 examples [00:33, 1016.53 examples/s]32616 examples [00:33, 1025.19 examples/s]32720 examples [00:33, 1026.66 examples/s]32823 examples [00:33, 1005.61 examples/s]32925 examples [00:33, 1008.38 examples/s]33028 examples [00:33, 1014.69 examples/s]33130 examples [00:34, 974.23 examples/s] 33228 examples [00:34, 968.56 examples/s]33330 examples [00:34, 980.38 examples/s]33435 examples [00:34, 999.50 examples/s]33537 examples [00:34, 1004.28 examples/s]33638 examples [00:34, 1000.31 examples/s]33739 examples [00:34, 999.50 examples/s] 33842 examples [00:34, 1005.06 examples/s]33947 examples [00:34, 1016.05 examples/s]34049 examples [00:35, 993.38 examples/s] 34149 examples [00:35, 971.87 examples/s]34248 examples [00:35, 976.78 examples/s]34352 examples [00:35, 991.96 examples/s]34455 examples [00:35, 1000.54 examples/s]34559 examples [00:35, 1009.81 examples/s]34661 examples [00:35, 1011.43 examples/s]34763 examples [00:35, 1007.80 examples/s]34864 examples [00:35, 999.88 examples/s] 34970 examples [00:35, 1016.10 examples/s]35072 examples [00:36, 974.71 examples/s] 35176 examples [00:36, 991.50 examples/s]35276 examples [00:36, 971.29 examples/s]35375 examples [00:36, 974.53 examples/s]35477 examples [00:36, 985.42 examples/s]35581 examples [00:36, 998.40 examples/s]35685 examples [00:36, 1008.68 examples/s]35787 examples [00:36, 1009.51 examples/s]35890 examples [00:36, 1015.49 examples/s]35993 examples [00:36, 1019.80 examples/s]36097 examples [00:37, 1023.37 examples/s]36201 examples [00:37, 1028.21 examples/s]36304 examples [00:37, 1025.57 examples/s]36408 examples [00:37, 1028.16 examples/s]36511 examples [00:37, 1027.05 examples/s]36616 examples [00:37, 1031.53 examples/s]36721 examples [00:37, 1035.46 examples/s]36825 examples [00:37, 1030.57 examples/s]36929 examples [00:37, 1026.66 examples/s]37034 examples [00:37, 1031.72 examples/s]37138 examples [00:38, 1032.14 examples/s]37242 examples [00:38, 1033.55 examples/s]37346 examples [00:38, 1013.88 examples/s]37448 examples [00:38, 967.90 examples/s] 37552 examples [00:38, 987.16 examples/s]37656 examples [00:38, 1000.83 examples/s]37760 examples [00:38, 1011.69 examples/s]37862 examples [00:38, 1008.72 examples/s]37964 examples [00:38, 1002.60 examples/s]38065 examples [00:39, 1000.80 examples/s]38166 examples [00:39, 994.73 examples/s] 38266 examples [00:39, 994.15 examples/s]38368 examples [00:39, 1000.12 examples/s]38469 examples [00:39, 1002.16 examples/s]38572 examples [00:39, 1007.76 examples/s]38673 examples [00:39, 1006.25 examples/s]38774 examples [00:39, 960.71 examples/s] 38878 examples [00:39, 982.08 examples/s]38983 examples [00:39, 1000.78 examples/s]39088 examples [00:40, 1013.63 examples/s]39193 examples [00:40, 1021.73 examples/s]39297 examples [00:40, 1024.63 examples/s]39400 examples [00:40, 1021.35 examples/s]39503 examples [00:40, 1006.22 examples/s]39604 examples [00:40, 998.16 examples/s] 39704 examples [00:40, 969.27 examples/s]39802 examples [00:40, 964.23 examples/s]39899 examples [00:40, 948.54 examples/s]40001 examples [00:40, 919.23 examples/s]40104 examples [00:41, 947.95 examples/s]40207 examples [00:41, 969.09 examples/s]40306 examples [00:41, 974.83 examples/s]40404 examples [00:41, 961.93 examples/s]40506 examples [00:41, 978.34 examples/s]40611 examples [00:41, 998.66 examples/s]40716 examples [00:41, 1012.54 examples/s]40820 examples [00:41, 1020.61 examples/s]40923 examples [00:41, 1014.06 examples/s]41027 examples [00:41, 1020.29 examples/s]41132 examples [00:42, 1026.61 examples/s]41235 examples [00:42, 1027.40 examples/s]41338 examples [00:42, 1005.91 examples/s]41439 examples [00:42, 982.66 examples/s] 41539 examples [00:42, 987.50 examples/s]41643 examples [00:42, 1001.12 examples/s]41745 examples [00:42, 1005.06 examples/s]41846 examples [00:42, 978.75 examples/s] 41945 examples [00:42, 972.82 examples/s]42043 examples [00:43, 934.07 examples/s]42141 examples [00:43, 945.79 examples/s]42238 examples [00:43, 950.29 examples/s]42341 examples [00:43, 971.16 examples/s]42441 examples [00:43, 979.38 examples/s]42544 examples [00:43, 992.91 examples/s]42646 examples [00:43, 1000.60 examples/s]42749 examples [00:43, 1008.32 examples/s]42851 examples [00:43, 1011.52 examples/s]42953 examples [00:43, 983.57 examples/s] 43056 examples [00:44, 996.38 examples/s]43161 examples [00:44, 1008.99 examples/s]43263 examples [00:44, 994.47 examples/s] 43363 examples [00:44, 979.32 examples/s]43466 examples [00:44, 991.85 examples/s]43569 examples [00:44, 1000.58 examples/s]43670 examples [00:44, 1001.68 examples/s]43772 examples [00:44, 1005.74 examples/s]43873 examples [00:44, 1002.18 examples/s]43974 examples [00:44, 1000.28 examples/s]44076 examples [00:45, 1005.00 examples/s]44177 examples [00:45, 992.92 examples/s] 44277 examples [00:45, 950.70 examples/s]44373 examples [00:45, 944.48 examples/s]44477 examples [00:45, 970.11 examples/s]44580 examples [00:45, 986.21 examples/s]44682 examples [00:45, 993.57 examples/s]44784 examples [00:45, 999.18 examples/s]44886 examples [00:45, 1003.48 examples/s]44987 examples [00:46, 1001.53 examples/s]45091 examples [00:46, 1011.22 examples/s]45193 examples [00:46, 959.10 examples/s] 45297 examples [00:46, 980.84 examples/s]45396 examples [00:46, 955.77 examples/s]45499 examples [00:46, 975.42 examples/s]45605 examples [00:46, 999.08 examples/s]45706 examples [00:46, 1001.72 examples/s]45812 examples [00:46, 1018.13 examples/s]45915 examples [00:46, 1014.11 examples/s]46018 examples [00:47, 1016.35 examples/s]46120 examples [00:47, 1000.70 examples/s]46221 examples [00:47, 981.68 examples/s] 46320 examples [00:47, 962.18 examples/s]46417 examples [00:47, 941.56 examples/s]46517 examples [00:47, 958.30 examples/s]46614 examples [00:47, 920.57 examples/s]46711 examples [00:47, 933.12 examples/s]46814 examples [00:47, 958.21 examples/s]46915 examples [00:47, 972.35 examples/s]47013 examples [00:48, 966.87 examples/s]47110 examples [00:48, 960.13 examples/s]47207 examples [00:48, 959.07 examples/s]47304 examples [00:48, 939.37 examples/s]47405 examples [00:48, 958.36 examples/s]47509 examples [00:48, 980.08 examples/s]47613 examples [00:48, 997.25 examples/s]47714 examples [00:48, 998.91 examples/s]47815 examples [00:48, 993.84 examples/s]47915 examples [00:49, 981.43 examples/s]48014 examples [00:49, 979.01 examples/s]48116 examples [00:49, 990.05 examples/s]48216 examples [00:49, 954.03 examples/s]48319 examples [00:49, 973.43 examples/s]48422 examples [00:49, 988.70 examples/s]48526 examples [00:49, 1002.26 examples/s]48630 examples [00:49, 1013.15 examples/s]48733 examples [00:49, 1015.34 examples/s]48835 examples [00:49, 1013.41 examples/s]48937 examples [00:50, 999.37 examples/s] 49043 examples [00:50, 1014.05 examples/s]49148 examples [00:50, 1023.13 examples/s]49251 examples [00:50, 1022.89 examples/s]49356 examples [00:50, 1029.47 examples/s]49460 examples [00:50, 1028.99 examples/s]49565 examples [00:50, 1034.00 examples/s]49671 examples [00:50, 1038.78 examples/s]49775 examples [00:50, 1031.19 examples/s]49879 examples [00:50, 1026.77 examples/s]49982 examples [00:51, 1005.47 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 5991/50000 [00:00<00:00, 59907.49 examples/s] 38%|███▊      | 18852/50000 [00:00<00:00, 71340.03 examples/s] 63%|██████▎   | 31671/50000 [00:00<00:00, 82287.12 examples/s] 89%|████████▉ | 44700/50000 [00:00<00:00, 92511.53 examples/s]                                                               0 examples [00:00, ? examples/s]83 examples [00:00, 824.85 examples/s]184 examples [00:00, 871.49 examples/s]288 examples [00:00, 914.42 examples/s]392 examples [00:00, 948.77 examples/s]495 examples [00:00, 971.49 examples/s]599 examples [00:00, 988.20 examples/s]702 examples [00:00, 999.40 examples/s]804 examples [00:00, 1004.75 examples/s]911 examples [00:00, 1022.85 examples/s]1017 examples [00:01, 1031.63 examples/s]1124 examples [00:01, 1040.71 examples/s]1229 examples [00:01, 1041.46 examples/s]1334 examples [00:01, 1043.90 examples/s]1441 examples [00:01, 1049.57 examples/s]1546 examples [00:01, 1047.36 examples/s]1653 examples [00:01, 1051.67 examples/s]1758 examples [00:01, 1036.23 examples/s]1862 examples [00:01, 1033.48 examples/s]1966 examples [00:01, 1032.23 examples/s]2073 examples [00:02, 1040.87 examples/s]2179 examples [00:02, 1045.60 examples/s]2285 examples [00:02, 1048.23 examples/s]2391 examples [00:02, 1050.17 examples/s]2497 examples [00:02, 1023.27 examples/s]2601 examples [00:02, 1025.77 examples/s]2704 examples [00:02, 986.49 examples/s] 2809 examples [00:02, 1004.35 examples/s]2912 examples [00:02, 1011.57 examples/s]3014 examples [00:02, 1007.22 examples/s]3116 examples [00:03, 1010.92 examples/s]3222 examples [00:03, 1024.53 examples/s]3325 examples [00:03, 1019.52 examples/s]3428 examples [00:03, 999.56 examples/s] 3534 examples [00:03, 1014.37 examples/s]3639 examples [00:03, 1023.79 examples/s]3742 examples [00:03, 1007.56 examples/s]3849 examples [00:03, 1023.20 examples/s]3954 examples [00:03, 1029.35 examples/s]4060 examples [00:03, 1036.48 examples/s]4166 examples [00:04, 1041.32 examples/s]4271 examples [00:04, 1039.87 examples/s]4376 examples [00:04, 1040.81 examples/s]4481 examples [00:04, 1041.13 examples/s]4586 examples [00:04, 1041.01 examples/s]4691 examples [00:04, 1014.74 examples/s]4793 examples [00:04, 979.58 examples/s] 4898 examples [00:04, 999.16 examples/s]4999 examples [00:04, 982.45 examples/s]5104 examples [00:04, 1001.49 examples/s]5210 examples [00:05, 1015.77 examples/s]5317 examples [00:05, 1031.23 examples/s]5424 examples [00:05, 1040.30 examples/s]5529 examples [00:05, 1030.59 examples/s]5633 examples [00:05, 1023.31 examples/s]5736 examples [00:05, 1024.05 examples/s]5841 examples [00:05, 1029.46 examples/s]5947 examples [00:05, 1036.09 examples/s]6053 examples [00:05, 1041.96 examples/s]6158 examples [00:06, 1030.66 examples/s]6262 examples [00:06, 1030.32 examples/s]6367 examples [00:06, 1033.79 examples/s]6471 examples [00:06, 1034.14 examples/s]6575 examples [00:06, 1032.87 examples/s]6679 examples [00:06, 1029.73 examples/s]6782 examples [00:06, 1023.06 examples/s]6885 examples [00:06, 1001.86 examples/s]6986 examples [00:06, 970.56 examples/s] 7092 examples [00:06, 993.52 examples/s]7192 examples [00:07, 994.01 examples/s]7292 examples [00:07, 940.54 examples/s]7397 examples [00:07, 970.02 examples/s]7499 examples [00:07, 984.47 examples/s]7600 examples [00:07, 989.58 examples/s]7703 examples [00:07, 999.20 examples/s]7809 examples [00:07, 1013.98 examples/s]7914 examples [00:07, 1022.07 examples/s]8017 examples [00:07, 1020.34 examples/s]8120 examples [00:07, 1017.31 examples/s]8222 examples [00:08, 1013.36 examples/s]8325 examples [00:08, 1015.29 examples/s]8428 examples [00:08, 1018.14 examples/s]8531 examples [00:08, 1018.94 examples/s]8635 examples [00:08, 1024.91 examples/s]8740 examples [00:08, 1029.14 examples/s]8843 examples [00:08, 1020.21 examples/s]8946 examples [00:08, 992.45 examples/s] 9047 examples [00:08, 997.57 examples/s]9149 examples [00:08, 1003.14 examples/s]9253 examples [00:09, 1011.23 examples/s]9357 examples [00:09, 1017.84 examples/s]9460 examples [00:09, 1019.64 examples/s]9563 examples [00:09, 997.39 examples/s] 9666 examples [00:09, 1005.89 examples/s]9770 examples [00:09, 1015.39 examples/s]9872 examples [00:09, 1014.17 examples/s]9975 examples [00:09, 1018.69 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI3JF3K/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI3JF3K/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0634a24950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0634a24950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0634a24950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f061353e748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f05f404e438>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0634a24950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0634a24950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0634a24950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f05f404ef28>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0632f97400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f05ad3c3ea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f05ad3c3ea0> 

  function with postional parmater data_info <function split_train_valid at 0x7f05ad3c3ea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f05ad3be048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f05ad3be048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f05ad3be048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=0a9a6bd1f0d90323ace80469f68e899941f0777fdf330144c0d9ab7c84d8f616
  Stored in directory: /tmp/pip-ephem-wheel-cache-1q279hfv/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<16:19:20, 14.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<11:39:50, 20.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:13:09, 29.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:00<5:45:48, 41.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:01:30, 59.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.65M/862M [00:01<2:47:58, 84.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.9M/862M [00:01<1:56:56, 121kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.3M/862M [00:01<1:21:39, 172kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.9M/862M [00:01<56:51, 246kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.9M/862M [00:01<39:48, 350kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:01<27:45, 498kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:01<19:29, 707kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.3M/862M [00:01<13:37, 1.00MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.5M/862M [00:02<09:38, 1.41MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.2M/862M [00:02<06:47, 1.99MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<05:21, 2.52MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<05:39, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<05:46, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:04<04:25, 3.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<05:40, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<07:05, 1.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<05:35, 2.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:06<04:10, 3.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<06:40, 1.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:04, 2.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:08<04:35, 2.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<06:15, 2.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<07:06, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<05:38, 2.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<06:04, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<05:35, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<04:14, 3.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<06:02, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<06:53, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<05:30, 2.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:14<03:59, 3.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<1:26:21, 151kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<1:01:46, 211kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:16<43:29, 299kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<33:23, 388kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<26:08, 495kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<18:57, 683kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:18<13:23, 963kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<1:41:23, 127kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<1:12:16, 178kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<50:49, 253kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<38:29, 333kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<29:32, 434kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<21:12, 604kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<15:00, 851kB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<15:18, 833kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<12:00, 1.06MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<08:43, 1.46MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<09:03, 1.40MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:00, 1.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:58, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<05:01, 2.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<1:33:42, 135kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:06:52, 189kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<47:00, 268kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<35:44, 351kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<26:15, 478kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<18:39, 671kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<15:58, 781kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:29, 999kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:02, 1.38MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<09:14, 1.34MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<09:00, 1.38MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<06:55, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<06:49, 1.81MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<06:02, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:32, 2.71MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:02, 2.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:44, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:15, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:50, 3.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<08:13, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<07:01, 1.73MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:39<05:13, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:29, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:34, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:41<04:10, 2.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:41<03:04, 3.93MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<50:47, 238kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<38:00, 317kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<27:10, 443kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<20:53, 574kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<15:53, 754kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<11:24, 1.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<10:44, 1.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<09:54, 1.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<07:32, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<07:10, 1.65MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:16, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<04:39, 2.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:58, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:23, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<04:03, 2.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:36, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<04:55, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<03:40, 3.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:53<02:43, 4.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<48:53, 238kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<35:11, 331kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<25:02, 465kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<17:34, 659kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<1:06:12, 175kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<48:42, 238kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<34:38, 334kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<26:00, 443kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<19:22, 594kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<13:47, 832kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<12:19, 929kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:47, 1.17MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<07:05, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<07:39, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:31, 1.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<04:51, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<06:03, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:32, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:03, 2.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<03:41, 3.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:49, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:37, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:54, 2.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:03, 1.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:31, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:04, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<03:40, 3.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:42, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:55, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:49, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:38, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:54, 1.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:20, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<03:50, 2.86MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<14:27, 759kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<11:15, 974kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<08:08, 1.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<08:14, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:54, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:06, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<06:06, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:22, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:01, 2.68MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:20, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:49, 2.23MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:39, 2.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<05:03, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:37, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<03:30, 3.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:56, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<05:43, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:33, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:17, 3.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<1:09:47, 151kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<49:55, 211kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<35:05, 300kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<26:55, 390kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<20:59, 500kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<15:07, 693kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<10:44, 972kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<10:51, 960kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<08:40, 1.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<06:19, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:50, 1.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:40, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:16, 2.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:07, 3.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<1:03:22, 162kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<45:22, 226kB/s]  .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<31:55, 321kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<24:41, 414kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<18:18, 557kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<13:02, 781kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<11:30, 882kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<10:08, 1.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<07:36, 1.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:55, 1.45MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:52, 1.71MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<04:21, 2.30MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:23, 1.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:50, 1.72MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:31, 2.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<03:17, 3.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:41, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:43, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:13, 2.35MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:15, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:36, 1.76MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:21, 2.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<03:09, 3.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<09:16, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:30, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<05:29, 1.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<06:07, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:16, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:53, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:01, 1.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:30, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:23, 2.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:38, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:13, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:11, 3.00MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:28, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:06, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<03:04, 3.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:24, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:03, 2.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:01, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:21, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:00, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:02, 3.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:19, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<03:59, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:01, 3.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:17, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:56, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:59, 3.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:15, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:51, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:51, 2.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:10, 2.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:52, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:54, 3.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:09, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:45, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:46, 2.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:05, 2.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:47, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<02:52, 3.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:05, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:40, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:43, 2.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:01, 2.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<03:44, 2.36MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<02:50, 3.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:00, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:37, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:41, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:58, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:40, 2.37MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:47, 3.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:58, 2.18MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:32, 1.90MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<03:33, 2.43MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:37, 3.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:51, 1.77MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:16, 2.00MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:09, 2.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:12, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:41, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:42, 2.29MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:56, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:38, 2.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:45, 3.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:53, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:25, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:30, 2.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<02:32, 3.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<1:01:32, 135kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<43:53, 189kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<30:50, 268kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<23:26, 351kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<17:14, 477kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<12:12, 672kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<10:25, 783kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<08:07, 1.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<05:52, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:00, 1.35MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:51, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:30, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:26, 1.81MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:56, 2.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:55, 2.73MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:54, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:32, 2.25MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:38, 3.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:43, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:24, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:34, 3.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:38, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:20, 2.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:31, 3.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:36, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:17, 2.35MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:28, 3.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:33, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:02, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:13, 2.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:19, 3.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<56:31, 135kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<40:18, 189kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<28:19, 268kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<21:30, 351kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<15:48, 477kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<11:13, 669kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<09:35, 780kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<08:16, 904kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<06:06, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<04:22, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<05:42, 1.30MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<04:45, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:30, 2.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:09, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:40, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:42, 2.69MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:37, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:16, 2.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:26, 2.96MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:25, 2.10MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:07, 2.30MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:21, 3.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:20, 2.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:47, 1.88MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:00, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:14, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<02:59, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:16, 3.10MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:12, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:40, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<02:53, 2.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:05, 3.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<10:55, 634kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<08:21, 828kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<06:00, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<05:47, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:45, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<03:29, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:02, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:31, 1.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:37, 2.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:25, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:04, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:19, 2.89MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:11, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:54, 2.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:09, 3.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:05, 2.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:27, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:45, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<01:59, 3.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<6:15:11, 17.4kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<4:23:03, 24.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<3:03:34, 35.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<2:09:17, 49.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<1:31:48, 70.2kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<1:04:27, 99.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<45:47, 139kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<32:40, 195kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<22:56, 276kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<17:27, 361kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<12:50, 490kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<09:07, 688kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<07:48, 799kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<06:40, 933kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:56, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<03:35, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:07, 1.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:31, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:35, 2.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:14, 1.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:30, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:43, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<01:57, 3.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<22:17, 271kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<16:12, 372kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<11:26, 524kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<09:22, 637kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<07:45, 768kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:44, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<04:03, 1.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<45:02, 131kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<32:06, 183kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<22:31, 260kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<17:02, 342kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<13:06, 445kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<09:23, 618kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<06:37, 872kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<07:04, 814kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:32, 1.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<04:00, 1.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:06, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:27, 1.65MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:31, 2.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:06, 1.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:18, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:35, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<01:51, 2.98MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<5:21:23, 17.3kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<3:45:16, 24.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<2:37:05, 35.1kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<1:50:32, 49.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<1:18:26, 69.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<55:04, 99.3kB/s]  .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<38:15, 142kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<1:11:05, 76.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<50:15, 108kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<35:08, 153kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<25:40, 208kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<19:03, 280kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<13:34, 393kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<09:28, 558kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<17:37, 299kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<12:52, 410kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<09:04, 578kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<07:32, 691kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<05:47, 897kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<04:10, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<04:07, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:55, 1.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:00, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:54, 1.75MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:34, 1.97MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<01:55, 2.62MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:29, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:15, 2.22MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:41, 2.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:20, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:40, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:05, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<01:32, 3.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:35, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:18, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<01:44, 2.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:19, 2.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:07, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:34, 3.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:12, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:01, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:31, 3.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:10, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:59, 2.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:30, 3.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:08, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:57, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:28, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:05, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:56, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:27, 3.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:04, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:21, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:52, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:00, 2.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:52, 2.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:23, 3.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:58, 2.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:48, 2.39MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:22, 3.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:57, 2.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:47, 2.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:21, 3.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:56, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:46, 2.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:19, 3.13MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:53, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:45, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:19, 3.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:52, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:43, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:17, 3.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:50, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:41, 2.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:15, 3.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:48, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:03, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:36, 2.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:11, 3.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:58, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:47, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:20, 2.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:48, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:01, 1.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:36, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:42, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:30, 2.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:08, 3.20MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:39, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:53, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:28, 2.45MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:04, 3.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:35, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:11, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:36, 2.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:56, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:42, 2.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:15, 2.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:43, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<01:21, 2.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<00:58, 3.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<33:52, 99.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<24:22, 138kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<17:08, 195kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<11:55, 277kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<09:46, 337kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<07:09, 458kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<05:03, 644kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<04:15, 758kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<03:17, 975kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<02:21, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:23, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:59, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:27, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:44, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:50, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:25, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:01, 2.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:58, 1.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:42, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:15, 2.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:33, 1.90MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:43, 1.71MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:21, 2.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<00:57, 2.99MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<07:34, 380kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<05:35, 513kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<03:57, 720kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<03:23, 828kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:36, 1.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:53, 1.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:08<01:20, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<11:59, 229kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<08:39, 316kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<06:04, 446kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<04:49, 554kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<03:38, 732kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<02:35, 1.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:24, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:56, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:24, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:34, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:21, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:00, 2.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:16, 1.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:08, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:51, 2.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:09, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:18, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:00, 2.35MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<00:43, 3.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:50, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:31, 1.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:07, 2.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:17, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:21, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:02, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:45, 2.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:20, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:09, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<00:51, 2.51MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:05, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:58, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:43, 2.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:59, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:53, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:40, 3.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:56, 2.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:51, 2.32MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:38, 3.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:53, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:49, 2.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:36, 3.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:51, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:47, 2.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:34, 3.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:49, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:45, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:34, 3.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:47, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:43, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:32, 3.08MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:45, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:41, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:31, 3.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:43, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:40, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:29, 3.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:41, 2.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:47, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:36, 2.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:26, 3.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:45, 1.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:40, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:30, 2.80MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:39, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:36, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:26, 3.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:36, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:41, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:32, 2.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:51<00:22, 3.28MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:57, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:47, 1.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:34, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:39, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:42, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:32, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:55<00:23, 2.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:39, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:34, 1.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:25, 2.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:31, 1.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:34, 1.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:26, 2.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:27, 2.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:24, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:18, 3.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:24, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:22, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:16, 3.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:22, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:20, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:15, 3.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:19, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:13, 3.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:21, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:16, 2.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:09<00:12, 3.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:17, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 2.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:17, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:13, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:13<00:09, 3.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:18, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:15, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:12, 1.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:14, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:10, 2.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:07, 3.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:18, 1.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:14, 1.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:09, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:10, 1.57MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:04, 2.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:39, 308kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:28, 419kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:17, 591kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:11, 708kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:09, 832kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 1.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:02, 1.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:13, 290kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:09, 396kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:03, 558kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 797/400000 [00:00<00:50, 7967.48it/s]  0%|          | 1636/400000 [00:00<00:49, 8087.14it/s]  1%|          | 2476/400000 [00:00<00:48, 8177.73it/s]  1%|          | 3323/400000 [00:00<00:48, 8261.23it/s]  1%|          | 4173/400000 [00:00<00:47, 8331.21it/s]  1%|          | 4988/400000 [00:00<00:47, 8274.35it/s]  1%|▏         | 5790/400000 [00:00<00:48, 8196.32it/s]  2%|▏         | 6619/400000 [00:00<00:47, 8223.01it/s]  2%|▏         | 7471/400000 [00:00<00:47, 8308.88it/s]  2%|▏         | 8323/400000 [00:01<00:46, 8368.80it/s]  2%|▏         | 9163/400000 [00:01<00:46, 8377.97it/s]  2%|▏         | 9991/400000 [00:01<00:46, 8346.17it/s]  3%|▎         | 10841/400000 [00:01<00:46, 8389.12it/s]  3%|▎         | 11686/400000 [00:01<00:46, 8406.97it/s]  3%|▎         | 12538/400000 [00:01<00:45, 8439.49it/s]  3%|▎         | 13379/400000 [00:01<00:45, 8427.83it/s]  4%|▎         | 14223/400000 [00:01<00:45, 8430.66it/s]  4%|▍         | 15073/400000 [00:01<00:45, 8451.01it/s]  4%|▍         | 15917/400000 [00:01<00:45, 8445.15it/s]  4%|▍         | 16769/400000 [00:02<00:45, 8466.41it/s]  4%|▍         | 17616/400000 [00:02<00:45, 8429.11it/s]  5%|▍         | 18459/400000 [00:02<00:46, 8280.08it/s]  5%|▍         | 19288/400000 [00:02<00:46, 8113.78it/s]  5%|▌         | 20101/400000 [00:02<00:47, 8034.12it/s]  5%|▌         | 20916/400000 [00:02<00:46, 8067.58it/s]  5%|▌         | 21743/400000 [00:02<00:46, 8126.86it/s]  6%|▌         | 22589/400000 [00:02<00:45, 8222.26it/s]  6%|▌         | 23435/400000 [00:02<00:45, 8290.01it/s]  6%|▌         | 24277/400000 [00:02<00:45, 8328.26it/s]  6%|▋         | 25127/400000 [00:03<00:44, 8377.22it/s]  6%|▋         | 25970/400000 [00:03<00:44, 8390.44it/s]  7%|▋         | 26815/400000 [00:03<00:44, 8405.87it/s]  7%|▋         | 27656/400000 [00:03<00:44, 8404.92it/s]  7%|▋         | 28505/400000 [00:03<00:44, 8428.47it/s]  7%|▋         | 29348/400000 [00:03<00:44, 8392.84it/s]  8%|▊         | 30191/400000 [00:03<00:44, 8401.80it/s]  8%|▊         | 31036/400000 [00:03<00:43, 8415.65it/s]  8%|▊         | 31879/400000 [00:03<00:43, 8419.05it/s]  8%|▊         | 32724/400000 [00:03<00:43, 8427.84it/s]  8%|▊         | 33567/400000 [00:04<00:43, 8393.76it/s]  9%|▊         | 34407/400000 [00:04<00:43, 8372.50it/s]  9%|▉         | 35245/400000 [00:04<00:43, 8361.91it/s]  9%|▉         | 36082/400000 [00:04<00:43, 8361.69it/s]  9%|▉         | 36931/400000 [00:04<00:43, 8396.95it/s]  9%|▉         | 37772/400000 [00:04<00:43, 8400.73it/s] 10%|▉         | 38621/400000 [00:04<00:42, 8426.64it/s] 10%|▉         | 39473/400000 [00:04<00:42, 8452.22it/s] 10%|█         | 40319/400000 [00:04<00:42, 8436.12it/s] 10%|█         | 41165/400000 [00:04<00:42, 8440.41it/s] 11%|█         | 42012/400000 [00:05<00:42, 8448.27it/s] 11%|█         | 42863/400000 [00:05<00:42, 8463.90it/s] 11%|█         | 43710/400000 [00:05<00:42, 8422.68it/s] 11%|█         | 44553/400000 [00:05<00:42, 8342.39it/s] 11%|█▏        | 45388/400000 [00:05<00:45, 7778.87it/s] 12%|█▏        | 46218/400000 [00:05<00:44, 7926.43it/s] 12%|█▏        | 47047/400000 [00:05<00:43, 8029.97it/s] 12%|█▏        | 47886/400000 [00:05<00:43, 8134.61it/s] 12%|█▏        | 48704/400000 [00:05<00:46, 7530.73it/s] 12%|█▏        | 49469/400000 [00:05<00:47, 7327.16it/s] 13%|█▎        | 50314/400000 [00:06<00:45, 7630.77it/s] 13%|█▎        | 51094/400000 [00:06<00:45, 7679.22it/s] 13%|█▎        | 51869/400000 [00:06<00:45, 7676.14it/s] 13%|█▎        | 52718/400000 [00:06<00:43, 7903.23it/s] 13%|█▎        | 53567/400000 [00:06<00:42, 8069.92it/s] 14%|█▎        | 54415/400000 [00:06<00:42, 8187.51it/s] 14%|█▍        | 55259/400000 [00:06<00:41, 8260.62it/s] 14%|█▍        | 56111/400000 [00:06<00:41, 8335.44it/s] 14%|█▍        | 56959/400000 [00:06<00:40, 8375.99it/s] 14%|█▍        | 57799/400000 [00:06<00:40, 8376.01it/s] 15%|█▍        | 58641/400000 [00:07<00:40, 8388.74it/s] 15%|█▍        | 59481/400000 [00:07<00:40, 8385.91it/s] 15%|█▌        | 60329/400000 [00:07<00:40, 8411.59it/s] 15%|█▌        | 61171/400000 [00:07<00:41, 8157.43it/s] 15%|█▌        | 61989/400000 [00:07<00:42, 7912.05it/s] 16%|█▌        | 62820/400000 [00:07<00:42, 8027.27it/s] 16%|█▌        | 63641/400000 [00:07<00:41, 8075.72it/s] 16%|█▌        | 64474/400000 [00:07<00:41, 8149.19it/s] 16%|█▋        | 65311/400000 [00:07<00:40, 8212.38it/s] 17%|█▋        | 66134/400000 [00:08<00:40, 8152.23it/s] 17%|█▋        | 66974/400000 [00:08<00:40, 8223.61it/s] 17%|█▋        | 67813/400000 [00:08<00:40, 8270.66it/s] 17%|█▋        | 68663/400000 [00:08<00:39, 8336.30it/s] 17%|█▋        | 69514/400000 [00:08<00:39, 8387.44it/s] 18%|█▊        | 70359/400000 [00:08<00:39, 8404.12it/s] 18%|█▊        | 71200/400000 [00:08<00:39, 8359.58it/s] 18%|█▊        | 72037/400000 [00:08<00:39, 8340.43it/s] 18%|█▊        | 72881/400000 [00:08<00:39, 8368.53it/s] 18%|█▊        | 73732/400000 [00:08<00:38, 8407.82it/s] 19%|█▊        | 74573/400000 [00:09<00:38, 8387.19it/s] 19%|█▉        | 75412/400000 [00:09<00:39, 8317.45it/s] 19%|█▉        | 76244/400000 [00:09<00:39, 8264.33it/s] 19%|█▉        | 77089/400000 [00:09<00:38, 8318.53it/s] 19%|█▉        | 77943/400000 [00:09<00:38, 8381.41it/s] 20%|█▉        | 78782/400000 [00:09<00:40, 7972.40it/s] 20%|█▉        | 79617/400000 [00:09<00:39, 8081.60it/s] 20%|██        | 80452/400000 [00:09<00:39, 8159.99it/s] 20%|██        | 81280/400000 [00:09<00:38, 8195.38it/s] 21%|██        | 82102/400000 [00:09<00:39, 8112.19it/s] 21%|██        | 82942/400000 [00:10<00:38, 8196.09it/s] 21%|██        | 83783/400000 [00:10<00:38, 8258.88it/s] 21%|██        | 84618/400000 [00:10<00:38, 8285.60it/s] 21%|██▏       | 85457/400000 [00:10<00:37, 8315.38it/s] 22%|██▏       | 86290/400000 [00:10<00:39, 8000.69it/s] 22%|██▏       | 87139/400000 [00:10<00:38, 8140.75it/s] 22%|██▏       | 87980/400000 [00:10<00:37, 8216.96it/s] 22%|██▏       | 88819/400000 [00:10<00:37, 8266.27it/s] 22%|██▏       | 89648/400000 [00:10<00:38, 7993.91it/s] 23%|██▎       | 90497/400000 [00:10<00:38, 8134.90it/s] 23%|██▎       | 91351/400000 [00:11<00:37, 8251.19it/s] 23%|██▎       | 92194/400000 [00:11<00:37, 8303.58it/s] 23%|██▎       | 93027/400000 [00:11<00:36, 8302.60it/s] 23%|██▎       | 93879/400000 [00:11<00:36, 8365.17it/s] 24%|██▎       | 94732/400000 [00:11<00:36, 8413.92it/s] 24%|██▍       | 95584/400000 [00:11<00:36, 8443.72it/s] 24%|██▍       | 96429/400000 [00:11<00:36, 8432.27it/s] 24%|██▍       | 97273/400000 [00:11<00:35, 8425.61it/s] 25%|██▍       | 98116/400000 [00:11<00:35, 8398.87it/s] 25%|██▍       | 98957/400000 [00:11<00:35, 8388.02it/s] 25%|██▍       | 99796/400000 [00:12<00:35, 8380.70it/s] 25%|██▌       | 100635/400000 [00:12<00:35, 8362.22it/s] 25%|██▌       | 101472/400000 [00:12<00:35, 8332.02it/s] 26%|██▌       | 102306/400000 [00:12<00:35, 8318.23it/s] 26%|██▌       | 103142/400000 [00:12<00:35, 8327.88it/s] 26%|██▌       | 103975/400000 [00:12<00:35, 8327.71it/s] 26%|██▌       | 104812/400000 [00:12<00:35, 8339.44it/s] 26%|██▋       | 105647/400000 [00:12<00:35, 8340.31it/s] 27%|██▋       | 106488/400000 [00:12<00:35, 8358.76it/s] 27%|██▋       | 107324/400000 [00:12<00:35, 8297.64it/s] 27%|██▋       | 108177/400000 [00:13<00:34, 8364.00it/s] 27%|██▋       | 109024/400000 [00:13<00:34, 8394.54it/s] 27%|██▋       | 109864/400000 [00:13<00:34, 8368.23it/s] 28%|██▊       | 110719/400000 [00:13<00:34, 8419.26it/s] 28%|██▊       | 111572/400000 [00:13<00:34, 8449.68it/s] 28%|██▊       | 112424/400000 [00:13<00:33, 8470.17it/s] 28%|██▊       | 113275/400000 [00:13<00:33, 8481.45it/s] 29%|██▊       | 114124/400000 [00:13<00:33, 8412.35it/s] 29%|██▊       | 114966/400000 [00:13<00:34, 8369.49it/s] 29%|██▉       | 115804/400000 [00:13<00:34, 8349.51it/s] 29%|██▉       | 116640/400000 [00:14<00:34, 8288.21it/s] 29%|██▉       | 117470/400000 [00:14<00:34, 8290.51it/s] 30%|██▉       | 118300/400000 [00:14<00:34, 8176.80it/s] 30%|██▉       | 119122/400000 [00:14<00:34, 8187.40it/s] 30%|██▉       | 119974/400000 [00:14<00:33, 8282.51it/s] 30%|███       | 120828/400000 [00:14<00:33, 8355.70it/s] 30%|███       | 121682/400000 [00:14<00:33, 8408.83it/s] 31%|███       | 122524/400000 [00:14<00:33, 8402.83it/s] 31%|███       | 123374/400000 [00:14<00:32, 8429.96it/s] 31%|███       | 124228/400000 [00:14<00:32, 8459.95it/s] 31%|███▏      | 125083/400000 [00:15<00:32, 8486.10it/s] 31%|███▏      | 125936/400000 [00:15<00:32, 8496.89it/s] 32%|███▏      | 126786/400000 [00:15<00:32, 8457.24it/s] 32%|███▏      | 127635/400000 [00:15<00:32, 8466.14it/s] 32%|███▏      | 128482/400000 [00:15<00:32, 8462.22it/s] 32%|███▏      | 129338/400000 [00:15<00:31, 8489.19it/s] 33%|███▎      | 130192/400000 [00:15<00:31, 8502.06it/s] 33%|███▎      | 131043/400000 [00:15<00:31, 8486.08it/s] 33%|███▎      | 131892/400000 [00:15<00:32, 8230.25it/s] 33%|███▎      | 132740/400000 [00:16<00:32, 8302.23it/s] 33%|███▎      | 133589/400000 [00:16<00:31, 8355.67it/s] 34%|███▎      | 134444/400000 [00:16<00:31, 8411.45it/s] 34%|███▍      | 135286/400000 [00:16<00:31, 8407.11it/s] 34%|███▍      | 136128/400000 [00:16<00:31, 8367.11it/s] 34%|███▍      | 136968/400000 [00:16<00:31, 8375.72it/s] 34%|███▍      | 137813/400000 [00:16<00:31, 8396.78it/s] 35%|███▍      | 138665/400000 [00:16<00:30, 8433.02it/s] 35%|███▍      | 139509/400000 [00:16<00:30, 8406.73it/s] 35%|███▌      | 140350/400000 [00:16<00:30, 8402.53it/s] 35%|███▌      | 141203/400000 [00:17<00:30, 8439.57it/s] 36%|███▌      | 142052/400000 [00:17<00:30, 8452.21it/s] 36%|███▌      | 142903/400000 [00:17<00:30, 8469.34it/s] 36%|███▌      | 143751/400000 [00:17<00:30, 8437.59it/s] 36%|███▌      | 144595/400000 [00:17<00:30, 8295.68it/s] 36%|███▋      | 145426/400000 [00:17<00:31, 8197.37it/s] 37%|███▋      | 146247/400000 [00:17<00:31, 8133.55it/s] 37%|███▋      | 147076/400000 [00:17<00:30, 8173.99it/s] 37%|███▋      | 147912/400000 [00:17<00:30, 8227.61it/s] 37%|███▋      | 148736/400000 [00:17<00:30, 8189.63it/s] 37%|███▋      | 149557/400000 [00:18<00:30, 8194.75it/s] 38%|███▊      | 150408/400000 [00:18<00:30, 8285.00it/s] 38%|███▊      | 151257/400000 [00:18<00:29, 8344.01it/s] 38%|███▊      | 152107/400000 [00:18<00:29, 8390.04it/s] 38%|███▊      | 152947/400000 [00:18<00:29, 8379.40it/s] 38%|███▊      | 153798/400000 [00:18<00:29, 8415.58it/s] 39%|███▊      | 154653/400000 [00:18<00:29, 8454.30it/s] 39%|███▉      | 155508/400000 [00:18<00:28, 8479.11it/s] 39%|███▉      | 156360/400000 [00:18<00:28, 8489.78it/s] 39%|███▉      | 157210/400000 [00:18<00:28, 8462.58it/s] 40%|███▉      | 158065/400000 [00:19<00:28, 8486.19it/s] 40%|███▉      | 158919/400000 [00:19<00:28, 8501.21it/s] 40%|███▉      | 159770/400000 [00:19<00:28, 8494.26it/s] 40%|████      | 160620/400000 [00:19<00:28, 8438.21it/s] 40%|████      | 161464/400000 [00:19<00:28, 8396.45it/s] 41%|████      | 162304/400000 [00:19<00:28, 8297.64it/s] 41%|████      | 163135/400000 [00:19<00:28, 8299.73it/s] 41%|████      | 163966/400000 [00:19<00:28, 8293.91it/s] 41%|████      | 164818/400000 [00:19<00:28, 8359.70it/s] 41%|████▏     | 165658/400000 [00:19<00:28, 8369.12it/s] 42%|████▏     | 166509/400000 [00:20<00:27, 8410.77it/s] 42%|████▏     | 167363/400000 [00:20<00:27, 8446.47it/s] 42%|████▏     | 168213/400000 [00:20<00:27, 8459.78it/s] 42%|████▏     | 169060/400000 [00:20<00:27, 8421.52it/s] 42%|████▏     | 169903/400000 [00:20<00:27, 8380.19it/s] 43%|████▎     | 170749/400000 [00:20<00:27, 8403.19it/s] 43%|████▎     | 171598/400000 [00:20<00:27, 8428.32it/s] 43%|████▎     | 172447/400000 [00:20<00:26, 8445.54it/s] 43%|████▎     | 173295/400000 [00:20<00:26, 8453.64it/s] 44%|████▎     | 174141/400000 [00:20<00:26, 8436.85it/s] 44%|████▎     | 174985/400000 [00:21<00:26, 8401.10it/s] 44%|████▍     | 175826/400000 [00:21<00:27, 8210.71it/s] 44%|████▍     | 176654/400000 [00:21<00:27, 8229.72it/s] 44%|████▍     | 177478/400000 [00:21<00:27, 8206.32it/s] 45%|████▍     | 178300/400000 [00:21<00:27, 8200.24it/s] 45%|████▍     | 179123/400000 [00:21<00:26, 8208.69it/s] 45%|████▍     | 179955/400000 [00:21<00:26, 8239.32it/s] 45%|████▌     | 180791/400000 [00:21<00:26, 8273.48it/s] 45%|████▌     | 181629/400000 [00:21<00:26, 8304.67it/s] 46%|████▌     | 182460/400000 [00:21<00:26, 8275.41it/s] 46%|████▌     | 183288/400000 [00:22<00:26, 8257.49it/s] 46%|████▌     | 184124/400000 [00:22<00:26, 8282.91it/s] 46%|████▌     | 184960/400000 [00:22<00:25, 8303.26it/s] 46%|████▋     | 185791/400000 [00:22<00:25, 8268.56it/s] 47%|████▋     | 186622/400000 [00:22<00:25, 8278.09it/s] 47%|████▋     | 187450/400000 [00:22<00:25, 8232.27it/s] 47%|████▋     | 188274/400000 [00:22<00:25, 8219.98it/s] 47%|████▋     | 189097/400000 [00:22<00:25, 8218.51it/s] 47%|████▋     | 189919/400000 [00:22<00:26, 8060.22it/s] 48%|████▊     | 190726/400000 [00:22<00:26, 7952.36it/s] 48%|████▊     | 191571/400000 [00:23<00:25, 8093.62it/s] 48%|████▊     | 192426/400000 [00:23<00:25, 8224.14it/s] 48%|████▊     | 193250/400000 [00:23<00:25, 8220.98it/s] 49%|████▊     | 194074/400000 [00:23<00:25, 8176.82it/s] 49%|████▊     | 194893/400000 [00:23<00:25, 8170.66it/s] 49%|████▉     | 195711/400000 [00:23<00:25, 8170.04it/s] 49%|████▉     | 196529/400000 [00:23<00:25, 8127.79it/s] 49%|████▉     | 197348/400000 [00:23<00:24, 8146.30it/s] 50%|████▉     | 198176/400000 [00:23<00:24, 8185.27it/s] 50%|████▉     | 199013/400000 [00:23<00:24, 8237.26it/s] 50%|████▉     | 199837/400000 [00:24<00:24, 8152.98it/s] 50%|█████     | 200653/400000 [00:24<00:24, 8145.76it/s] 50%|█████     | 201491/400000 [00:24<00:24, 8214.50it/s] 51%|█████     | 202341/400000 [00:24<00:23, 8295.63it/s] 51%|█████     | 203184/400000 [00:24<00:23, 8334.79it/s] 51%|█████     | 204018/400000 [00:24<00:24, 8072.99it/s] 51%|█████     | 204870/400000 [00:24<00:23, 8199.40it/s] 51%|█████▏    | 205727/400000 [00:24<00:23, 8305.35it/s] 52%|█████▏    | 206583/400000 [00:24<00:23, 8377.82it/s] 52%|█████▏    | 207431/400000 [00:24<00:22, 8407.52it/s] 52%|█████▏    | 208273/400000 [00:25<00:22, 8383.95it/s] 52%|█████▏    | 209120/400000 [00:25<00:22, 8407.86it/s] 52%|█████▏    | 209962/400000 [00:25<00:22, 8408.81it/s] 53%|█████▎    | 210809/400000 [00:25<00:22, 8425.37it/s] 53%|█████▎    | 211658/400000 [00:25<00:22, 8443.05it/s] 53%|█████▎    | 212503/400000 [00:25<00:22, 8398.11it/s] 53%|█████▎    | 213349/400000 [00:25<00:22, 8414.07it/s] 54%|█████▎    | 214201/400000 [00:25<00:22, 8445.11it/s] 54%|█████▍    | 215046/400000 [00:25<00:21, 8441.43it/s] 54%|█████▍    | 215891/400000 [00:26<00:21, 8423.40it/s] 54%|█████▍    | 216734/400000 [00:26<00:21, 8396.39it/s] 54%|█████▍    | 217574/400000 [00:26<00:21, 8339.69it/s] 55%|█████▍    | 218409/400000 [00:26<00:21, 8340.51it/s] 55%|█████▍    | 219261/400000 [00:26<00:21, 8391.95it/s] 55%|█████▌    | 220111/400000 [00:26<00:21, 8423.13it/s] 55%|█████▌    | 220954/400000 [00:26<00:21, 8394.46it/s] 55%|█████▌    | 221806/400000 [00:26<00:21, 8431.54it/s] 56%|█████▌    | 222657/400000 [00:26<00:20, 8454.77it/s] 56%|█████▌    | 223503/400000 [00:26<00:21, 8379.52it/s] 56%|█████▌    | 224350/400000 [00:27<00:20, 8405.80it/s] 56%|█████▋    | 225191/400000 [00:27<00:20, 8367.97it/s] 57%|█████▋    | 226028/400000 [00:27<00:20, 8343.77it/s] 57%|█████▋    | 226863/400000 [00:27<00:20, 8317.99it/s] 57%|█████▋    | 227708/400000 [00:27<00:20, 8355.28it/s] 57%|█████▋    | 228544/400000 [00:27<00:20, 8354.83it/s] 57%|█████▋    | 229380/400000 [00:27<00:20, 8336.47it/s] 58%|█████▊    | 230214/400000 [00:27<00:20, 8336.50it/s] 58%|█████▊    | 231048/400000 [00:27<00:20, 8244.85it/s] 58%|█████▊    | 231873/400000 [00:27<00:20, 8125.07it/s] 58%|█████▊    | 232724/400000 [00:28<00:20, 8235.01it/s] 58%|█████▊    | 233564/400000 [00:28<00:20, 8282.66it/s] 59%|█████▊    | 234400/400000 [00:28<00:19, 8305.71it/s] 59%|█████▉    | 235241/400000 [00:28<00:19, 8336.71it/s] 59%|█████▉    | 236094/400000 [00:28<00:19, 8391.79it/s] 59%|█████▉    | 236942/400000 [00:28<00:19, 8417.79it/s] 59%|█████▉    | 237785/400000 [00:28<00:19, 8401.66it/s] 60%|█████▉    | 238626/400000 [00:28<00:19, 8226.40it/s] 60%|█████▉    | 239458/400000 [00:28<00:19, 8252.07it/s] 60%|██████    | 240297/400000 [00:28<00:19, 8291.38it/s] 60%|██████    | 241127/400000 [00:29<00:19, 8183.55it/s] 60%|██████    | 241963/400000 [00:29<00:19, 8233.76it/s] 61%|██████    | 242808/400000 [00:29<00:18, 8295.84it/s] 61%|██████    | 243659/400000 [00:29<00:18, 8356.20it/s] 61%|██████    | 244512/400000 [00:29<00:18, 8406.81it/s] 61%|██████▏   | 245354/400000 [00:29<00:18, 8300.09it/s] 62%|██████▏   | 246185/400000 [00:29<00:18, 8257.99it/s] 62%|██████▏   | 247013/400000 [00:29<00:18, 8264.17it/s] 62%|██████▏   | 247846/400000 [00:29<00:18, 8281.26it/s] 62%|██████▏   | 248697/400000 [00:29<00:18, 8345.83it/s] 62%|██████▏   | 249543/400000 [00:30<00:17, 8378.92it/s] 63%|██████▎   | 250382/400000 [00:30<00:17, 8377.07it/s] 63%|██████▎   | 251220/400000 [00:30<00:17, 8357.41it/s] 63%|██████▎   | 252065/400000 [00:30<00:17, 8383.54it/s] 63%|██████▎   | 252912/400000 [00:30<00:17, 8408.07it/s] 63%|██████▎   | 253761/400000 [00:30<00:17, 8430.34it/s] 64%|██████▎   | 254605/400000 [00:30<00:17, 8417.14it/s] 64%|██████▍   | 255449/400000 [00:30<00:17, 8422.65it/s] 64%|██████▍   | 256299/400000 [00:30<00:17, 8445.54it/s] 64%|██████▍   | 257151/400000 [00:30<00:16, 8466.70it/s] 65%|██████▍   | 258003/400000 [00:31<00:16, 8481.57it/s] 65%|██████▍   | 258852/400000 [00:31<00:16, 8464.06it/s] 65%|██████▍   | 259699/400000 [00:31<00:16, 8390.81it/s] 65%|██████▌   | 260539/400000 [00:31<00:16, 8335.32it/s] 65%|██████▌   | 261391/400000 [00:31<00:16, 8387.40it/s] 66%|██████▌   | 262240/400000 [00:31<00:16, 8416.08it/s] 66%|██████▌   | 263082/400000 [00:31<00:16, 8408.97it/s] 66%|██████▌   | 263924/400000 [00:31<00:16, 8385.57it/s] 66%|██████▌   | 264763/400000 [00:31<00:16, 8371.26it/s] 66%|██████▋   | 265601/400000 [00:31<00:16, 8278.39it/s] 67%|██████▋   | 266442/400000 [00:32<00:16, 8315.36it/s] 67%|██████▋   | 267274/400000 [00:32<00:15, 8304.07it/s] 67%|██████▋   | 268105/400000 [00:32<00:15, 8265.99it/s] 67%|██████▋   | 268932/400000 [00:32<00:15, 8263.49it/s] 67%|██████▋   | 269771/400000 [00:32<00:15, 8300.12it/s] 68%|██████▊   | 270611/400000 [00:32<00:15, 8327.55it/s] 68%|██████▊   | 271444/400000 [00:32<00:15, 8230.53it/s] 68%|██████▊   | 272268/400000 [00:32<00:15, 8171.28it/s] 68%|██████▊   | 273090/400000 [00:32<00:15, 8185.68it/s] 68%|██████▊   | 273909/400000 [00:32<00:15, 8134.91it/s] 69%|██████▊   | 274736/400000 [00:33<00:15, 8172.10it/s] 69%|██████▉   | 275559/400000 [00:33<00:15, 8187.13it/s] 69%|██████▉   | 276386/400000 [00:33<00:15, 8211.48it/s] 69%|██████▉   | 277232/400000 [00:33<00:14, 8282.68it/s] 70%|██████▉   | 278083/400000 [00:33<00:14, 8347.04it/s] 70%|██████▉   | 278937/400000 [00:33<00:14, 8401.66it/s] 70%|██████▉   | 279778/400000 [00:33<00:14, 8402.72it/s] 70%|███████   | 280621/400000 [00:33<00:14, 8409.40it/s] 70%|███████   | 281469/400000 [00:33<00:14, 8429.19it/s] 71%|███████   | 282319/400000 [00:33<00:13, 8448.27it/s] 71%|███████   | 283170/400000 [00:34<00:13, 8466.40it/s] 71%|███████   | 284018/400000 [00:34<00:13, 8467.78it/s] 71%|███████   | 284865/400000 [00:34<00:13, 8413.02it/s] 71%|███████▏  | 285707/400000 [00:34<00:13, 8388.79it/s] 72%|███████▏  | 286560/400000 [00:34<00:13, 8428.35it/s] 72%|███████▏  | 287413/400000 [00:34<00:13, 8458.15it/s] 72%|███████▏  | 288259/400000 [00:34<00:13, 8351.96it/s] 72%|███████▏  | 289110/400000 [00:34<00:13, 8395.93it/s] 72%|███████▏  | 289954/400000 [00:34<00:13, 8408.54it/s] 73%|███████▎  | 290805/400000 [00:34<00:12, 8437.31it/s] 73%|███████▎  | 291652/400000 [00:35<00:12, 8444.86it/s] 73%|███████▎  | 292501/400000 [00:35<00:12, 8456.35it/s] 73%|███████▎  | 293347/400000 [00:35<00:12, 8448.41it/s] 74%|███████▎  | 294192/400000 [00:35<00:12, 8393.91it/s] 74%|███████▍  | 295043/400000 [00:35<00:12, 8426.44it/s] 74%|███████▍  | 295891/400000 [00:35<00:12, 8440.91it/s] 74%|███████▍  | 296736/400000 [00:35<00:12, 8436.34it/s] 74%|███████▍  | 297580/400000 [00:35<00:12, 8425.66it/s] 75%|███████▍  | 298427/400000 [00:35<00:12, 8436.19it/s] 75%|███████▍  | 299279/400000 [00:35<00:11, 8458.23it/s] 75%|███████▌  | 300130/400000 [00:36<00:11, 8472.69it/s] 75%|███████▌  | 300978/400000 [00:36<00:11, 8437.57it/s] 75%|███████▌  | 301822/400000 [00:36<00:11, 8300.21it/s] 76%|███████▌  | 302653/400000 [00:36<00:11, 8301.85it/s] 76%|███████▌  | 303504/400000 [00:36<00:11, 8362.80it/s] 76%|███████▌  | 304356/400000 [00:36<00:11, 8406.84it/s] 76%|███████▋  | 305201/400000 [00:36<00:11, 8418.54it/s] 77%|███████▋  | 306044/400000 [00:36<00:11, 8406.84it/s] 77%|███████▋  | 306885/400000 [00:36<00:11, 8396.52it/s] 77%|███████▋  | 307735/400000 [00:36<00:10, 8426.91it/s] 77%|███████▋  | 308587/400000 [00:37<00:10, 8454.15it/s] 77%|███████▋  | 309433/400000 [00:37<00:10, 8452.05it/s] 78%|███████▊  | 310279/400000 [00:37<00:10, 8440.42it/s] 78%|███████▊  | 311124/400000 [00:37<00:10, 8332.61it/s] 78%|███████▊  | 311958/400000 [00:37<00:10, 8329.32it/s] 78%|███████▊  | 312792/400000 [00:37<00:10, 8250.69it/s] 78%|███████▊  | 313631/400000 [00:37<00:10, 8291.41it/s] 79%|███████▊  | 314461/400000 [00:37<00:10, 8211.27it/s] 79%|███████▉  | 315283/400000 [00:37<00:10, 8146.71it/s] 79%|███████▉  | 316099/400000 [00:38<00:10, 8023.92it/s] 79%|███████▉  | 316903/400000 [00:38<00:10, 8021.40it/s] 79%|███████▉  | 317722/400000 [00:38<00:10, 8069.97it/s] 80%|███████▉  | 318563/400000 [00:38<00:09, 8167.03it/s] 80%|███████▉  | 319403/400000 [00:38<00:09, 8234.19it/s] 80%|████████  | 320252/400000 [00:38<00:09, 8309.02it/s] 80%|████████  | 321099/400000 [00:38<00:09, 8354.81it/s] 80%|████████  | 321945/400000 [00:38<00:09, 8384.07it/s] 81%|████████  | 322795/400000 [00:38<00:09, 8416.22it/s] 81%|████████  | 323638/400000 [00:38<00:09, 8417.96it/s] 81%|████████  | 324486/400000 [00:39<00:08, 8435.68it/s] 81%|████████▏ | 325337/400000 [00:39<00:08, 8455.61it/s] 82%|████████▏ | 326183/400000 [00:39<00:08, 8345.36it/s] 82%|████████▏ | 327018/400000 [00:39<00:08, 8310.91it/s] 82%|████████▏ | 327852/400000 [00:39<00:08, 8317.25it/s] 82%|████████▏ | 328685/400000 [00:39<00:08, 8318.13it/s] 82%|████████▏ | 329523/400000 [00:39<00:08, 8336.53it/s] 83%|████████▎ | 330357/400000 [00:39<00:08, 8310.48it/s] 83%|████████▎ | 331198/400000 [00:39<00:08, 8337.64it/s] 83%|████████▎ | 332032/400000 [00:39<00:08, 8304.51it/s] 83%|████████▎ | 332863/400000 [00:40<00:08, 8303.99it/s] 83%|████████▎ | 333694/400000 [00:40<00:08, 8214.00it/s] 84%|████████▎ | 334543/400000 [00:40<00:07, 8294.27it/s] 84%|████████▍ | 335373/400000 [00:40<00:07, 8260.46it/s] 84%|████████▍ | 336209/400000 [00:40<00:07, 8288.24it/s] 84%|████████▍ | 337057/400000 [00:40<00:07, 8343.76it/s] 84%|████████▍ | 337908/400000 [00:40<00:07, 8392.79it/s] 85%|████████▍ | 338748/400000 [00:40<00:07, 8222.34it/s] 85%|████████▍ | 339572/400000 [00:40<00:07, 8148.95it/s] 85%|████████▌ | 340419/400000 [00:40<00:07, 8241.20it/s] 85%|████████▌ | 341267/400000 [00:41<00:07, 8309.87it/s] 86%|████████▌ | 342115/400000 [00:41<00:06, 8357.93it/s] 86%|████████▌ | 342964/400000 [00:41<00:06, 8395.99it/s] 86%|████████▌ | 343805/400000 [00:41<00:06, 8304.68it/s] 86%|████████▌ | 344636/400000 [00:41<00:06, 8299.54it/s] 86%|████████▋ | 345467/400000 [00:41<00:06, 8278.81it/s] 87%|████████▋ | 346308/400000 [00:41<00:06, 8315.19it/s] 87%|████████▋ | 347140/400000 [00:41<00:06, 8315.70it/s] 87%|████████▋ | 347972/400000 [00:41<00:06, 8315.09it/s] 87%|████████▋ | 348804/400000 [00:41<00:06, 8295.36it/s] 87%|████████▋ | 349634/400000 [00:42<00:06, 8275.42it/s] 88%|████████▊ | 350474/400000 [00:42<00:05, 8309.81it/s] 88%|████████▊ | 351318/400000 [00:42<00:05, 8347.99it/s] 88%|████████▊ | 352153/400000 [00:42<00:05, 8318.54it/s] 88%|████████▊ | 352987/400000 [00:42<00:05, 8324.44it/s] 88%|████████▊ | 353820/400000 [00:42<00:05, 8298.79it/s] 89%|████████▊ | 354657/400000 [00:42<00:05, 8319.94it/s] 89%|████████▉ | 355493/400000 [00:42<00:05, 8331.47it/s] 89%|████████▉ | 356327/400000 [00:42<00:05, 8271.22it/s] 89%|████████▉ | 357180/400000 [00:42<00:05, 8345.98it/s] 90%|████████▉ | 358015/400000 [00:43<00:05, 8281.58it/s] 90%|████████▉ | 358867/400000 [00:43<00:04, 8350.71it/s] 90%|████████▉ | 359717/400000 [00:43<00:04, 8393.29it/s] 90%|█████████ | 360557/400000 [00:43<00:04, 8379.60it/s] 90%|█████████ | 361404/400000 [00:43<00:04, 8405.73it/s] 91%|█████████ | 362245/400000 [00:43<00:04, 8405.66it/s] 91%|█████████ | 363088/400000 [00:43<00:04, 8412.36it/s] 91%|█████████ | 363934/400000 [00:43<00:04, 8425.03it/s] 91%|█████████ | 364777/400000 [00:43<00:04, 8410.15it/s] 91%|█████████▏| 365622/400000 [00:43<00:04, 8420.34it/s] 92%|█████████▏| 366465/400000 [00:44<00:04, 8276.49it/s] 92%|█████████▏| 367299/400000 [00:44<00:03, 8292.84it/s] 92%|█████████▏| 368129/400000 [00:44<00:03, 8190.30it/s] 92%|█████████▏| 368949/400000 [00:44<00:03, 8147.61it/s] 92%|█████████▏| 369765/400000 [00:44<00:03, 8124.86it/s] 93%|█████████▎| 370602/400000 [00:44<00:03, 8195.19it/s] 93%|█████████▎| 371440/400000 [00:44<00:03, 8247.19it/s] 93%|█████████▎| 372289/400000 [00:44<00:03, 8317.41it/s] 93%|█████████▎| 373130/400000 [00:44<00:03, 8344.95it/s] 93%|█████████▎| 373982/400000 [00:44<00:03, 8396.21it/s] 94%|█████████▎| 374828/400000 [00:45<00:02, 8413.52it/s] 94%|█████████▍| 375676/400000 [00:45<00:02, 8431.25it/s] 94%|█████████▍| 376526/400000 [00:45<00:02, 8444.68it/s] 94%|█████████▍| 377371/400000 [00:45<00:02, 8424.59it/s] 95%|█████████▍| 378222/400000 [00:45<00:02, 8449.92it/s] 95%|█████████▍| 379072/400000 [00:45<00:02, 8462.07it/s] 95%|█████████▍| 379919/400000 [00:45<00:02, 8463.98it/s] 95%|█████████▌| 380768/400000 [00:45<00:02, 8471.08it/s] 95%|█████████▌| 381616/400000 [00:45<00:02, 8451.38it/s] 96%|█████████▌| 382467/400000 [00:45<00:02, 8467.01it/s] 96%|█████████▌| 383314/400000 [00:46<00:02, 8316.54it/s] 96%|█████████▌| 384147/400000 [00:46<00:01, 8318.92it/s] 96%|█████████▌| 384981/400000 [00:46<00:01, 8322.72it/s] 96%|█████████▋| 385814/400000 [00:46<00:01, 8018.51it/s] 97%|█████████▋| 386642/400000 [00:46<00:01, 8093.65it/s] 97%|█████████▋| 387459/400000 [00:46<00:01, 8114.66it/s] 97%|█████████▋| 388294/400000 [00:46<00:01, 8182.08it/s] 97%|█████████▋| 389131/400000 [00:46<00:01, 8235.23it/s] 97%|█████████▋| 389969/400000 [00:46<00:01, 8276.13it/s] 98%|█████████▊| 390804/400000 [00:46<00:01, 8295.57it/s] 98%|█████████▊| 391635/400000 [00:47<00:01, 8278.07it/s] 98%|█████████▊| 392482/400000 [00:47<00:00, 8332.76it/s] 98%|█████████▊| 393334/400000 [00:47<00:00, 8386.51it/s] 99%|█████████▊| 394173/400000 [00:47<00:00, 8385.25it/s] 99%|█████████▉| 395012/400000 [00:47<00:00, 8324.20it/s] 99%|█████████▉| 395845/400000 [00:47<00:00, 8274.51it/s] 99%|█████████▉| 396677/400000 [00:47<00:00, 8287.01it/s] 99%|█████████▉| 397506/400000 [00:47<00:00, 8286.82it/s]100%|█████████▉| 398349/400000 [00:47<00:00, 8327.24it/s]100%|█████████▉| 399190/400000 [00:47<00:00, 8350.99it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8317.31it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f05ad6a4b70>, <torchtext.data.dataset.TabularDataset object at 0x7f05ad6a4cc0>, <torchtext.vocab.Vocab object at 0x7f05ad6a4be0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.38 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.38 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.38 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.76 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.76 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.06 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.06 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.88 file/s]2020-07-17 18:20:17.951533: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-17 18:20:17.955795: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-07-17 18:20:17.955965: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ab561694b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-17 18:20:17.955979: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3137536/9912422 [00:00<00:00, 30596169.82it/s]9920512it [00:00, 32706866.74it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1465562.16it/s]
0it [00:00, ?it/s]  2%|▏         | 32768/1648877 [00:00<00:04, 327638.13it/s]1654784it [00:00, 12386351.25it/s]                         
0it [00:00, ?it/s]8192it [00:00, 197176.26it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
