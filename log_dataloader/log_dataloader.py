
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f0f27252ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f0f27252ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f0f925121e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f0f925121e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0fb0859e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0fb0859e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0f3f839488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0f3f839488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0f3f839488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 3022848/9912422 [00:00<00:00, 30192300.81it/s]9920512it [00:00, 32609412.66it/s]                             
0it [00:00, ?it/s]32768it [00:00, 538186.95it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162291.65it/s]1654784it [00:00, 10563586.45it/s]                         
0it [00:00, ?it/s]8192it [00:00, 178411.51it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0f269623c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0f2694b630>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0f3f8390d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0f3f8390d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f0f3f8390d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:37,  1.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:37,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:37,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:36,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:35,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:35,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:34,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:34,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:06,  2.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:06,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:05,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:05,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:04,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:04,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:03,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:03,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:03,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:02,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:02,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:43,  3.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:43,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:43,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:43,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:42,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:42,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:42,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:41,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:41,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:41,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:41,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:28,  4.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:28,  4.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:28,  4.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:28,  4.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:27,  4.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:27,  4.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:27,  4.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:27,  4.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:26,  4.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:19,  6.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:19,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:18,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:18,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:18,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:18,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:17,  6.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  8.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:11,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:08, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:05, 16.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 22.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 28.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 36.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 36.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 44.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:01, 44.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 53.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 53.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 61.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 80.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 80.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 80.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 80.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.35s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.35s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.35s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.02 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.21s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.35s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.02 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.21s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.21s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.49 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.21s/ url]
0 examples [00:00, ? examples/s]2020-06-22 00:53:26.098847: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-22 00:53:26.112451: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-22 00:53:26.112650: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563254cf8530 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-22 00:53:26.112670: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
19 examples [00:00, 188.81 examples/s]137 examples [00:00, 252.36 examples/s]258 examples [00:00, 330.93 examples/s]371 examples [00:00, 420.03 examples/s]493 examples [00:00, 522.81 examples/s]614 examples [00:00, 630.04 examples/s]733 examples [00:00, 732.61 examples/s]853 examples [00:00, 828.64 examples/s]971 examples [00:00, 908.52 examples/s]1084 examples [00:01, 927.91 examples/s]1192 examples [00:01, 926.25 examples/s]1304 examples [00:01, 976.52 examples/s]1425 examples [00:01, 1035.35 examples/s]1548 examples [00:01, 1085.63 examples/s]1664 examples [00:01, 1104.69 examples/s]1782 examples [00:01, 1126.21 examples/s]1898 examples [00:01, 1034.15 examples/s]2017 examples [00:01, 1075.58 examples/s]2137 examples [00:01, 1109.04 examples/s]2251 examples [00:02, 1107.11 examples/s]2364 examples [00:02, 1103.49 examples/s]2484 examples [00:02, 1128.50 examples/s]2601 examples [00:02, 1140.54 examples/s]2721 examples [00:02, 1155.34 examples/s]2840 examples [00:02, 1162.53 examples/s]2957 examples [00:02, 1156.32 examples/s]3077 examples [00:02, 1166.75 examples/s]3196 examples [00:02, 1171.75 examples/s]3314 examples [00:02, 1133.33 examples/s]3431 examples [00:03, 1141.38 examples/s]3550 examples [00:03, 1155.14 examples/s]3674 examples [00:03, 1177.17 examples/s]3794 examples [00:03, 1182.79 examples/s]3915 examples [00:03, 1190.04 examples/s]4035 examples [00:03, 1189.24 examples/s]4159 examples [00:03, 1202.80 examples/s]4280 examples [00:03, 1203.00 examples/s]4401 examples [00:03, 1193.91 examples/s]4521 examples [00:04, 1167.71 examples/s]4638 examples [00:04, 1152.02 examples/s]4754 examples [00:04, 1120.68 examples/s]4877 examples [00:04, 1149.96 examples/s]4999 examples [00:04, 1168.31 examples/s]5122 examples [00:04, 1184.57 examples/s]5244 examples [00:04, 1191.29 examples/s]5364 examples [00:04, 1164.04 examples/s]5481 examples [00:04, 1130.68 examples/s]5595 examples [00:04, 1131.88 examples/s]5710 examples [00:05, 1133.93 examples/s]5833 examples [00:05, 1160.16 examples/s]5959 examples [00:05, 1187.20 examples/s]6081 examples [00:05, 1194.92 examples/s]6201 examples [00:05, 1192.42 examples/s]6330 examples [00:05, 1217.88 examples/s]6453 examples [00:05, 1206.13 examples/s]6574 examples [00:05, 1204.70 examples/s]6695 examples [00:05, 1202.33 examples/s]6816 examples [00:05, 1146.56 examples/s]6932 examples [00:06, 1149.86 examples/s]7048 examples [00:06, 1151.65 examples/s]7164 examples [00:06, 1150.37 examples/s]7280 examples [00:06, 1146.68 examples/s]7397 examples [00:06, 1152.71 examples/s]7513 examples [00:06, 1153.34 examples/s]7629 examples [00:06, 1141.33 examples/s]7746 examples [00:06, 1148.88 examples/s]7863 examples [00:06, 1154.83 examples/s]7979 examples [00:06, 1117.98 examples/s]8096 examples [00:07, 1131.61 examples/s]8210 examples [00:07, 1114.74 examples/s]8322 examples [00:07, 1007.35 examples/s]8429 examples [00:07, 1025.32 examples/s]8534 examples [00:07, 1032.42 examples/s]8651 examples [00:07, 1069.18 examples/s]8766 examples [00:07, 1089.98 examples/s]8876 examples [00:07, 1088.87 examples/s]8989 examples [00:07, 1099.92 examples/s]9100 examples [00:08, 1070.47 examples/s]9216 examples [00:08, 1093.61 examples/s]9336 examples [00:08, 1122.35 examples/s]9451 examples [00:08, 1129.14 examples/s]9572 examples [00:08, 1149.33 examples/s]9693 examples [00:08, 1163.90 examples/s]9812 examples [00:08, 1171.34 examples/s]9938 examples [00:08, 1194.19 examples/s]10058 examples [00:08, 1127.87 examples/s]10172 examples [00:08, 1128.91 examples/s]10286 examples [00:09, 1128.39 examples/s]10408 examples [00:09, 1153.33 examples/s]10524 examples [00:09, 1153.50 examples/s]10642 examples [00:09, 1160.52 examples/s]10764 examples [00:09, 1175.17 examples/s]10882 examples [00:09, 1165.00 examples/s]10999 examples [00:09, 1149.43 examples/s]11115 examples [00:09, 1149.87 examples/s]11237 examples [00:09, 1168.56 examples/s]11358 examples [00:09, 1177.40 examples/s]11476 examples [00:10, 1116.25 examples/s]11599 examples [00:10, 1146.89 examples/s]11715 examples [00:10, 1113.37 examples/s]11832 examples [00:10, 1127.87 examples/s]11948 examples [00:10, 1135.07 examples/s]12064 examples [00:10, 1141.47 examples/s]12180 examples [00:10, 1144.43 examples/s]12295 examples [00:10, 1106.02 examples/s]12407 examples [00:10, 1046.21 examples/s]12513 examples [00:11, 1027.47 examples/s]12617 examples [00:11, 1027.48 examples/s]12721 examples [00:11, 1020.01 examples/s]12824 examples [00:11, 1008.75 examples/s]12942 examples [00:11, 1052.68 examples/s]13065 examples [00:11, 1098.17 examples/s]13189 examples [00:11, 1136.00 examples/s]13304 examples [00:11, 1118.92 examples/s]13426 examples [00:11, 1145.13 examples/s]13548 examples [00:11, 1164.18 examples/s]13672 examples [00:12, 1181.95 examples/s]13791 examples [00:12, 1176.67 examples/s]13909 examples [00:12, 1174.73 examples/s]14028 examples [00:12, 1179.13 examples/s]14147 examples [00:12, 1179.42 examples/s]14266 examples [00:12, 1177.23 examples/s]14384 examples [00:12, 1162.40 examples/s]14503 examples [00:12, 1169.58 examples/s]14624 examples [00:12, 1181.39 examples/s]14743 examples [00:13, 1123.17 examples/s]14856 examples [00:13, 1121.73 examples/s]14969 examples [00:13, 1112.54 examples/s]15082 examples [00:13, 1115.62 examples/s]15194 examples [00:13, 1116.77 examples/s]15313 examples [00:13, 1136.03 examples/s]15431 examples [00:13, 1147.93 examples/s]15546 examples [00:13, 1147.71 examples/s]15663 examples [00:13, 1153.74 examples/s]15784 examples [00:13, 1169.07 examples/s]15902 examples [00:14, 1144.43 examples/s]16017 examples [00:14, 1138.62 examples/s]16135 examples [00:14, 1148.92 examples/s]16252 examples [00:14, 1155.08 examples/s]16372 examples [00:14, 1167.66 examples/s]16490 examples [00:14, 1171.19 examples/s]16611 examples [00:14, 1179.72 examples/s]16732 examples [00:14, 1187.73 examples/s]16859 examples [00:14, 1209.14 examples/s]16981 examples [00:14, 1204.29 examples/s]17103 examples [00:15, 1207.69 examples/s]17224 examples [00:15, 1172.85 examples/s]17342 examples [00:15, 1170.34 examples/s]17460 examples [00:15, 1157.03 examples/s]17590 examples [00:15, 1195.47 examples/s]17710 examples [00:15, 1154.90 examples/s]17829 examples [00:15, 1163.27 examples/s]17952 examples [00:15, 1182.13 examples/s]18072 examples [00:15, 1185.59 examples/s]18191 examples [00:15, 1175.61 examples/s]18309 examples [00:16, 1144.64 examples/s]18424 examples [00:16, 1123.19 examples/s]18538 examples [00:16, 1126.38 examples/s]18651 examples [00:16, 1097.83 examples/s]18764 examples [00:16, 1104.98 examples/s]18875 examples [00:16, 1093.80 examples/s]18989 examples [00:16, 1104.80 examples/s]19103 examples [00:16, 1114.29 examples/s]19217 examples [00:16, 1119.49 examples/s]19332 examples [00:16, 1126.86 examples/s]19447 examples [00:17, 1132.60 examples/s]19561 examples [00:17, 1129.15 examples/s]19674 examples [00:17, 1097.86 examples/s]19789 examples [00:17, 1110.26 examples/s]19901 examples [00:17, 1112.68 examples/s]20013 examples [00:17, 1081.93 examples/s]20129 examples [00:17, 1103.25 examples/s]20248 examples [00:17, 1125.43 examples/s]20364 examples [00:17, 1134.36 examples/s]20480 examples [00:18, 1139.80 examples/s]20599 examples [00:18, 1153.96 examples/s]20715 examples [00:18, 1143.57 examples/s]20830 examples [00:18, 1143.42 examples/s]20949 examples [00:18, 1155.68 examples/s]21065 examples [00:18, 1150.45 examples/s]21181 examples [00:18, 1143.53 examples/s]21297 examples [00:18, 1146.17 examples/s]21412 examples [00:18, 1144.57 examples/s]21529 examples [00:18, 1150.26 examples/s]21645 examples [00:19, 1150.34 examples/s]21761 examples [00:19, 1147.41 examples/s]21878 examples [00:19, 1152.39 examples/s]21996 examples [00:19, 1158.28 examples/s]22112 examples [00:19, 1118.96 examples/s]22229 examples [00:19, 1132.34 examples/s]22350 examples [00:19, 1153.36 examples/s]22470 examples [00:19, 1164.48 examples/s]22593 examples [00:19, 1181.62 examples/s]22716 examples [00:19, 1193.72 examples/s]22836 examples [00:20, 1175.75 examples/s]22954 examples [00:20, 1170.99 examples/s]23072 examples [00:20, 1162.20 examples/s]23197 examples [00:20, 1184.80 examples/s]23323 examples [00:20, 1203.73 examples/s]23450 examples [00:20, 1222.84 examples/s]23573 examples [00:20, 1218.22 examples/s]23695 examples [00:20, 1216.52 examples/s]23820 examples [00:20, 1224.98 examples/s]23946 examples [00:20, 1233.81 examples/s]24070 examples [00:21, 1207.16 examples/s]24195 examples [00:21, 1218.88 examples/s]24318 examples [00:21, 1205.92 examples/s]24441 examples [00:21, 1211.19 examples/s]24570 examples [00:21, 1233.39 examples/s]24694 examples [00:21, 1223.34 examples/s]24817 examples [00:21, 1201.89 examples/s]24938 examples [00:21, 1156.42 examples/s]25058 examples [00:21, 1166.55 examples/s]25176 examples [00:22, 1166.85 examples/s]25296 examples [00:22, 1176.04 examples/s]25424 examples [00:22, 1202.76 examples/s]25545 examples [00:22, 1141.35 examples/s]25660 examples [00:22, 1135.69 examples/s]25775 examples [00:22, 1032.84 examples/s]25881 examples [00:22, 1028.85 examples/s]26003 examples [00:22, 1078.96 examples/s]26128 examples [00:22, 1123.91 examples/s]26251 examples [00:22, 1153.43 examples/s]26376 examples [00:23, 1178.76 examples/s]26500 examples [00:23, 1195.69 examples/s]26621 examples [00:23, 1198.65 examples/s]26742 examples [00:23, 1189.54 examples/s]26863 examples [00:23, 1194.63 examples/s]26983 examples [00:23, 1181.44 examples/s]27106 examples [00:23, 1194.50 examples/s]27231 examples [00:23, 1208.00 examples/s]27352 examples [00:23, 1188.84 examples/s]27477 examples [00:23, 1205.74 examples/s]27598 examples [00:24, 1204.04 examples/s]27726 examples [00:24, 1224.36 examples/s]27849 examples [00:24, 1216.91 examples/s]27971 examples [00:24, 1184.11 examples/s]28092 examples [00:24, 1191.24 examples/s]28212 examples [00:24, 1174.98 examples/s]28334 examples [00:24, 1186.38 examples/s]28457 examples [00:24, 1197.41 examples/s]28579 examples [00:24, 1202.82 examples/s]28705 examples [00:25, 1218.58 examples/s]28827 examples [00:25, 1185.55 examples/s]28950 examples [00:25, 1196.46 examples/s]29072 examples [00:25, 1201.54 examples/s]29194 examples [00:25, 1206.38 examples/s]29324 examples [00:25, 1231.24 examples/s]29448 examples [00:25, 1162.08 examples/s]29566 examples [00:25, 1149.03 examples/s]29691 examples [00:25, 1176.81 examples/s]29810 examples [00:25, 1156.75 examples/s]29927 examples [00:26, 1113.70 examples/s]30040 examples [00:26, 1085.37 examples/s]30159 examples [00:26, 1114.35 examples/s]30281 examples [00:26, 1143.89 examples/s]30405 examples [00:26, 1168.76 examples/s]30530 examples [00:26, 1190.69 examples/s]30650 examples [00:26, 1187.90 examples/s]30771 examples [00:26, 1194.20 examples/s]30891 examples [00:26, 1177.71 examples/s]31010 examples [00:26, 1143.72 examples/s]31125 examples [00:27, 1134.70 examples/s]31241 examples [00:27, 1141.43 examples/s]31360 examples [00:27, 1154.89 examples/s]31477 examples [00:27, 1158.57 examples/s]31593 examples [00:27, 1158.33 examples/s]31712 examples [00:27, 1164.76 examples/s]31834 examples [00:27, 1177.77 examples/s]31960 examples [00:27, 1201.09 examples/s]32081 examples [00:27, 1200.77 examples/s]32202 examples [00:28, 1182.61 examples/s]32321 examples [00:28, 1181.43 examples/s]32441 examples [00:28, 1186.41 examples/s]32560 examples [00:28, 1180.86 examples/s]32679 examples [00:28, 1174.46 examples/s]32797 examples [00:28, 1157.41 examples/s]32913 examples [00:28, 1152.07 examples/s]33029 examples [00:28, 1145.50 examples/s]33144 examples [00:28, 1120.09 examples/s]33258 examples [00:28, 1124.59 examples/s]33371 examples [00:29, 1118.49 examples/s]33483 examples [00:29, 1118.64 examples/s]33596 examples [00:29, 1121.61 examples/s]33718 examples [00:29, 1147.81 examples/s]33842 examples [00:29, 1172.27 examples/s]33960 examples [00:29, 1169.40 examples/s]34080 examples [00:29, 1178.32 examples/s]34198 examples [00:29, 1178.29 examples/s]34316 examples [00:29, 1164.90 examples/s]34433 examples [00:29, 1165.60 examples/s]34550 examples [00:30, 1149.12 examples/s]34666 examples [00:30, 1151.27 examples/s]34782 examples [00:30, 1153.75 examples/s]34898 examples [00:30, 1137.56 examples/s]35019 examples [00:30, 1156.30 examples/s]35140 examples [00:30, 1168.51 examples/s]35261 examples [00:30, 1179.96 examples/s]35381 examples [00:30, 1185.25 examples/s]35500 examples [00:30, 1186.45 examples/s]35623 examples [00:30, 1196.47 examples/s]35743 examples [00:31, 1166.57 examples/s]35860 examples [00:31, 1142.54 examples/s]35975 examples [00:31, 1136.36 examples/s]36089 examples [00:31, 1127.35 examples/s]36209 examples [00:31, 1147.39 examples/s]36325 examples [00:31, 1149.50 examples/s]36441 examples [00:31, 1152.43 examples/s]36557 examples [00:31, 1143.94 examples/s]36672 examples [00:31, 1110.74 examples/s]36788 examples [00:31, 1123.07 examples/s]36906 examples [00:32, 1138.16 examples/s]37022 examples [00:32, 1143.73 examples/s]37137 examples [00:32, 1145.50 examples/s]37256 examples [00:32, 1156.75 examples/s]37380 examples [00:32, 1177.90 examples/s]37499 examples [00:32, 1179.92 examples/s]37625 examples [00:32, 1200.86 examples/s]37746 examples [00:32, 1184.77 examples/s]37865 examples [00:32, 1165.23 examples/s]37982 examples [00:33, 1162.48 examples/s]38099 examples [00:33, 1160.80 examples/s]38220 examples [00:33, 1173.12 examples/s]38340 examples [00:33, 1180.70 examples/s]38468 examples [00:33, 1208.46 examples/s]38596 examples [00:33, 1227.41 examples/s]38719 examples [00:33, 1183.31 examples/s]38846 examples [00:33, 1206.98 examples/s]38968 examples [00:33, 1205.57 examples/s]39089 examples [00:33, 1197.69 examples/s]39210 examples [00:34, 1181.33 examples/s]39329 examples [00:34, 1143.86 examples/s]39447 examples [00:34, 1152.73 examples/s]39567 examples [00:34, 1165.89 examples/s]39684 examples [00:34, 1161.93 examples/s]39801 examples [00:34, 1147.46 examples/s]39916 examples [00:34, 1137.34 examples/s]40030 examples [00:34, 1078.29 examples/s]40139 examples [00:34, 1073.62 examples/s]40264 examples [00:34, 1118.71 examples/s]40388 examples [00:35, 1150.78 examples/s]40510 examples [00:35, 1169.86 examples/s]40637 examples [00:35, 1195.68 examples/s]40758 examples [00:35, 1199.75 examples/s]40879 examples [00:35, 1188.25 examples/s]40999 examples [00:35, 1171.42 examples/s]41117 examples [00:35, 1166.44 examples/s]41242 examples [00:35, 1188.54 examples/s]41362 examples [00:35, 1189.76 examples/s]41482 examples [00:35, 1181.47 examples/s]41601 examples [00:36, 1168.85 examples/s]41719 examples [00:36, 1149.29 examples/s]41839 examples [00:36, 1162.55 examples/s]41961 examples [00:36, 1179.06 examples/s]42084 examples [00:36, 1191.55 examples/s]42204 examples [00:36, 1184.53 examples/s]42323 examples [00:36, 1185.02 examples/s]42442 examples [00:36, 1179.23 examples/s]42567 examples [00:36, 1196.94 examples/s]42687 examples [00:37, 1152.45 examples/s]42808 examples [00:37, 1168.72 examples/s]42934 examples [00:37, 1192.61 examples/s]43054 examples [00:37, 1179.05 examples/s]43173 examples [00:37, 1179.79 examples/s]43296 examples [00:37, 1193.21 examples/s]43416 examples [00:37, 1190.75 examples/s]43536 examples [00:37, 1189.10 examples/s]43655 examples [00:37, 1184.08 examples/s]43774 examples [00:37, 1122.35 examples/s]43897 examples [00:38, 1151.96 examples/s]44016 examples [00:38, 1162.12 examples/s]44141 examples [00:38, 1185.97 examples/s]44264 examples [00:38, 1197.82 examples/s]44389 examples [00:38, 1209.79 examples/s]44514 examples [00:38, 1219.58 examples/s]44637 examples [00:38, 1212.91 examples/s]44761 examples [00:38, 1218.55 examples/s]44883 examples [00:38, 1211.01 examples/s]45005 examples [00:38, 1201.71 examples/s]45126 examples [00:39, 1194.32 examples/s]45246 examples [00:39, 1185.02 examples/s]45365 examples [00:39, 1184.90 examples/s]45487 examples [00:39, 1193.42 examples/s]45607 examples [00:39, 1188.83 examples/s]45730 examples [00:39, 1200.72 examples/s]45851 examples [00:39, 1202.76 examples/s]45980 examples [00:39, 1226.08 examples/s]46106 examples [00:39, 1235.97 examples/s]46230 examples [00:39, 1213.05 examples/s]46352 examples [00:40, 1168.16 examples/s]46472 examples [00:40, 1176.60 examples/s]46591 examples [00:40, 1162.94 examples/s]46712 examples [00:40, 1174.75 examples/s]46833 examples [00:40, 1184.30 examples/s]46952 examples [00:40, 1180.55 examples/s]47071 examples [00:40, 1173.08 examples/s]47190 examples [00:40, 1176.01 examples/s]47313 examples [00:40, 1191.01 examples/s]47433 examples [00:41, 1132.37 examples/s]47547 examples [00:41, 1133.41 examples/s]47661 examples [00:41, 1108.60 examples/s]47773 examples [00:41, 1098.30 examples/s]47893 examples [00:41, 1124.80 examples/s]48006 examples [00:41, 1104.68 examples/s]48117 examples [00:41, 1099.14 examples/s]48230 examples [00:41, 1106.67 examples/s]48341 examples [00:41, 1063.02 examples/s]48451 examples [00:41, 1073.74 examples/s]48559 examples [00:42, 1068.46 examples/s]48675 examples [00:42, 1093.71 examples/s]48785 examples [00:42, 1095.50 examples/s]48903 examples [00:42, 1117.23 examples/s]49019 examples [00:42, 1127.76 examples/s]49133 examples [00:42, 1129.30 examples/s]49247 examples [00:42, 1124.07 examples/s]49361 examples [00:42, 1127.70 examples/s]49482 examples [00:42, 1150.44 examples/s]49602 examples [00:42, 1162.65 examples/s]49719 examples [00:43, 1140.48 examples/s]49841 examples [00:43, 1162.00 examples/s]49958 examples [00:43, 1110.06 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7351/50000 [00:00<00:00, 73504.46 examples/s] 39%|███▉      | 19597/50000 [00:00<00:00, 83521.16 examples/s] 60%|██████    | 30087/50000 [00:00<00:00, 88960.05 examples/s] 87%|████████▋ | 43476/50000 [00:00<00:00, 98917.03 examples/s]                                                               0 examples [00:00, ? examples/s]79 examples [00:00, 789.45 examples/s]186 examples [00:00, 856.44 examples/s]285 examples [00:00, 890.82 examples/s]402 examples [00:00, 957.19 examples/s]512 examples [00:00, 995.56 examples/s]621 examples [00:00, 1021.68 examples/s]742 examples [00:00, 1071.25 examples/s]857 examples [00:00, 1091.27 examples/s]973 examples [00:00, 1109.60 examples/s]1090 examples [00:01, 1127.05 examples/s]1207 examples [00:01, 1138.78 examples/s]1325 examples [00:01, 1149.64 examples/s]1440 examples [00:01, 1139.25 examples/s]1557 examples [00:01, 1145.82 examples/s]1673 examples [00:01, 1148.14 examples/s]1795 examples [00:01, 1167.54 examples/s]1913 examples [00:01, 1171.04 examples/s]2031 examples [00:01, 1150.64 examples/s]2154 examples [00:01, 1172.15 examples/s]2273 examples [00:02, 1174.46 examples/s]2396 examples [00:02, 1189.38 examples/s]2516 examples [00:02, 1175.14 examples/s]2634 examples [00:02, 1138.28 examples/s]2749 examples [00:02, 1137.32 examples/s]2863 examples [00:02, 1073.84 examples/s]2985 examples [00:02, 1111.68 examples/s]3106 examples [00:02, 1138.31 examples/s]3221 examples [00:02, 1129.73 examples/s]3338 examples [00:02, 1140.91 examples/s]3455 examples [00:03, 1147.79 examples/s]3571 examples [00:03, 1146.07 examples/s]3686 examples [00:03, 1051.63 examples/s]3801 examples [00:03, 1078.52 examples/s]3914 examples [00:03, 1088.11 examples/s]4024 examples [00:03, 1088.48 examples/s]4140 examples [00:03, 1108.57 examples/s]4260 examples [00:03, 1132.31 examples/s]4374 examples [00:03, 1108.70 examples/s]4491 examples [00:04, 1124.31 examples/s]4608 examples [00:04, 1137.22 examples/s]4723 examples [00:04, 1136.19 examples/s]4837 examples [00:04, 1134.84 examples/s]4959 examples [00:04, 1159.08 examples/s]5076 examples [00:04, 1161.24 examples/s]5195 examples [00:04, 1168.35 examples/s]5319 examples [00:04, 1188.16 examples/s]5438 examples [00:04, 1170.71 examples/s]5560 examples [00:04, 1182.91 examples/s]5681 examples [00:05, 1189.75 examples/s]5801 examples [00:05, 1161.73 examples/s]5919 examples [00:05, 1164.91 examples/s]6036 examples [00:05, 1157.53 examples/s]6156 examples [00:05, 1169.54 examples/s]6276 examples [00:05, 1177.91 examples/s]6394 examples [00:05, 1177.12 examples/s]6512 examples [00:05, 1177.61 examples/s]6630 examples [00:05, 1151.14 examples/s]6748 examples [00:05, 1158.83 examples/s]6871 examples [00:06, 1177.31 examples/s]6989 examples [00:06, 1168.47 examples/s]7106 examples [00:06, 1165.67 examples/s]7223 examples [00:06, 1105.68 examples/s]7347 examples [00:06, 1141.90 examples/s]7468 examples [00:06, 1159.99 examples/s]7590 examples [00:06, 1177.13 examples/s]7714 examples [00:06, 1193.22 examples/s]7834 examples [00:06, 1192.68 examples/s]7954 examples [00:06, 1194.76 examples/s]8075 examples [00:07, 1198.95 examples/s]8196 examples [00:07, 1191.10 examples/s]8316 examples [00:07, 1169.96 examples/s]8435 examples [00:07, 1175.46 examples/s]8554 examples [00:07, 1179.55 examples/s]8674 examples [00:07, 1184.25 examples/s]8793 examples [00:07, 1182.42 examples/s]8912 examples [00:07, 1183.58 examples/s]9031 examples [00:07, 1179.40 examples/s]9153 examples [00:07, 1189.97 examples/s]9273 examples [00:08, 1169.58 examples/s]9391 examples [00:08, 1127.36 examples/s]9508 examples [00:08, 1139.62 examples/s]9630 examples [00:08, 1160.71 examples/s]9747 examples [00:08, 1157.03 examples/s]9863 examples [00:08, 1151.48 examples/s]9980 examples [00:08, 1155.05 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9WO803/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9WO803/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0f3f839488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0f3f839488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0f3f839488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0f3ca63550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0ec8cba1d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0f3f839488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0f3f839488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0f3f839488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0f2694b400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0f2694b128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f0eb760f1e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f0eb760f1e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f0eb760f1e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f0eb760f2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f0eb760f2f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f0eb760f2f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=8311758a67e122c1da61be737eff3fafec095eb2fa0a11412b2564d3ab9c481b
  Stored in directory: /tmp/pip-ephem-wheel-cache-cf5gj11c/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:01:48, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:22:35, 14.6kB/s].vector_cache/glove.6B.zip:   0%|          | 164k/862M [00:01<11:31:45, 20.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 434k/862M [00:01<8:05:41, 29.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.29M/862M [00:01<5:40:09, 42.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.47M/862M [00:01<3:57:42, 60.2kB/s].vector_cache/glove.6B.zip:   1%|          | 7.05M/862M [00:01<2:45:53, 85.9kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<1:55:30, 123kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:20:27, 175kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<56:10, 249kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<39:10, 356kB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:02<27:25, 506kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.9M/862M [00:02<19:09, 719kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.6M/862M [00:02<13:28, 1.02MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:02<09:27, 1.44MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.3M/862M [00:02<06:42, 2.03MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<04:53, 2.76MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<05:19, 2.52MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<06:29, 2.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<05:48, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:04<04:24, 3.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<03:15, 4.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<26:57, 496kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<20:17, 659kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<14:29, 920kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<13:07, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<12:02, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<09:02, 1.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:08<06:31, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<09:06, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<07:44, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:10<05:44, 2.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<07:07, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<06:19, 2.08MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<04:45, 2.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<06:25, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<08:02, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<07:38, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<06:08, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:14<04:39, 2.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<03:27, 3.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<19:17, 675kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<15:21, 848kB/s].vector_cache/glove.6B.zip:  10%|▉         | 81.9M/862M [00:16<11:12, 1.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<10:14, 1.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<08:43, 1.49MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<06:29, 1.99MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<07:11, 1.79MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<08:43, 1.48MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<08:07, 1.59MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<06:36, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<04:58, 2.58MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:20<03:39, 3.51MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<40:58, 313kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<30:22, 422kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:22<21:37, 592kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<17:35, 725kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<13:36, 936kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<10:03, 1.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<07:11, 1.76MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<6:06:37, 34.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<4:21:49, 48.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<3:04:28, 68.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<2:09:11, 97.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<1:33:00, 136kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:10:14, 180kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<50:25, 250kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<35:32, 354kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<27:39, 453kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<24:29, 512kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<18:20, 683kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<13:04, 957kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<09:25, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:57, 962kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<15:49, 788kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:49, 972kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:19, 1.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<06:56, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<05:08, 2.41MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<13:31, 918kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<12:41, 977kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<11:25, 1.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<08:16, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<06:04, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:51, 1.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<13:41, 901kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<11:07, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<08:10, 1.51MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<05:57, 2.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<11:20, 1.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<13:26, 912kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<10:51, 1.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<07:58, 1.53MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<05:47, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<11:15, 1.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<13:50, 881kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<11:06, 1.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<08:06, 1.50MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<05:55, 2.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<12:32, 967kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<14:27, 839kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:33, 1.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<08:28, 1.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<06:13, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<04:35, 2.63MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:18:27, 19.5kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<7:18:44, 27.5kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<5:08:22, 39.1kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<3:35:52, 55.7kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<2:30:58, 79.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:53:32, 106kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:24:46, 141kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:00:39, 198kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<42:44, 280kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<30:04, 397kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<33:52, 352kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<29:02, 410kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<21:40, 550kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<15:39, 760kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<11:17, 1.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<08:13, 1.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<11:45, 1.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<11:28, 1.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:49, 1.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<06:26, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<07:47, 1.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<10:51, 1.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:43, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:48, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:01, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:46, 3.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<11:47, 994kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<13:43, 854kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<10:54, 1.07MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:04, 1.45MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:32, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:47, 2.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:45, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<9:09:45, 21.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<6:29:09, 29.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<4:33:08, 42.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<3:11:11, 60.8kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<2:14:13, 86.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<1:34:08, 123kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<1:12:57, 159kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<59:26, 195kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<43:42, 265kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<31:04, 372kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<22:02, 523kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<15:44, 731kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<22:28, 512kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<19:45, 582kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<14:39, 784kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<10:45, 1.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<07:48, 1.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<05:52, 1.95MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:54, 1.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<10:40, 1.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<08:15, 1.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:08, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:40, 2.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:42, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<14:10, 802kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<11:30, 987kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:31, 1.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<06:17, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:43, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<13:04, 864kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<12:41, 890kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<09:46, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<07:19, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:27, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:08, 2.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:33, 1.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<12:56, 868kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<10:47, 1.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:01, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:55, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:25, 2.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<4:31:32, 41.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<3:13:26, 57.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<2:16:08, 81.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<1:35:27, 117kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<1:06:56, 166kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<53:00, 209kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<43:54, 253kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<32:31, 341kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<23:07, 479kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<16:37, 665kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<11:56, 925kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<08:40, 1.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<49:26, 223kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<38:02, 290kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<27:19, 403kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<19:26, 565kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<13:54, 789kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<10:01, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<49:58, 219kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<41:50, 262kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<31:01, 353kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<22:08, 494kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<15:46, 692kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<11:19, 961kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<1:34:31, 115kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<1:09:34, 156kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<49:21, 220kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<35:01, 310kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<24:45, 438kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<17:40, 613kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<18:10, 595kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<19:29, 555kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<15:23, 702kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<11:16, 958kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<08:12, 1.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:02, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:32, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<1:15:49, 142kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<56:24, 190kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<40:10, 267kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<28:24, 377kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<20:10, 531kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<14:24, 741kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<19:49, 539kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<20:36, 518kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<16:06, 662kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<11:40, 913kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<08:50, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<06:24, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:53, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<19:15, 551kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<17:48, 596kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<14:18, 741kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<11:34, 916kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<10:01, 1.06MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:30, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:47, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:51, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:38, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:16, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:43, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:47, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:20, 1.98MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<05:46, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<05:27, 1.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:30, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:22, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:16, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:08, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:09, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:01, 2.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:04, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<04:52, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:01, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:52, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:54, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:45, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:50, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:48, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:42, 2.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:43, 2.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:42, 2.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<04:41, 2.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:34, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:36, 2.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:31, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:29, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:30, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:23, 2.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:21, 2.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:20, 2.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:29, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:27, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:22, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:20, 2.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:15, 2.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:13, 2.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:09, 2.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:09, 2.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:05, 2.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:03, 2.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<08:44, 1.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<07:18, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:15, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:35, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:02, 2.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:41, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:27, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:14, 2.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:00, 2.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<03:46, 2.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:55, 2.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:56, 2.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:53, 2.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<20:58, 493kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<16:02, 645kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<13:18, 777kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<10:59, 941kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<09:20, 1.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:02, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:54, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:58, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:14, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:42, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:19, 2.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:05, 2.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<03:52, 2.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:45, 2.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:37, 2.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:34, 2.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<09:51, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<08:23, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:52, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:50, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:13, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:32, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:56, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:51, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:15, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:20, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:39, 2.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:08, 2.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<03:39, 2.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<03:23, 3.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<07:55, 1.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:45, 1.51MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:27, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:31, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:47, 2.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:24, 2.99MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:02, 3.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<02:38, 3.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<02:39, 3.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<25:36, 396kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<21:13, 478kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<15:39, 647kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<11:41, 865kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:37, 1.17MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<06:45, 1.50MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:17, 1.91MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<04:14, 2.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:26, 2.93MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<6:29:17, 25.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<4:35:05, 36.6kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<3:13:22, 52.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<2:16:00, 74.0kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<1:35:39, 105kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<1:07:31, 149kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<47:49, 210kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<33:51, 296kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<24:04, 416kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<33:20, 300kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<27:31, 363kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<20:16, 493kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<14:31, 688kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<10:39, 935kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<07:43, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<05:50, 1.70MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<14:24, 690kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<13:34, 732kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<10:21, 959kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:35, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:32, 1.79MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<04:16, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<13:05, 754kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<12:08, 813kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<09:24, 1.05MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:11, 1.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:23, 1.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:02, 2.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:06, 3.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<19:12, 510kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<15:59, 613kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<11:45, 832kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<08:27, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<06:06, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<14:43, 661kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<14:48, 657kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<11:35, 839kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<08:38, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:17, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<04:34, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<11:28, 841kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<12:00, 804kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<09:23, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:47, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:55, 1.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<10:00, 958kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<10:41, 897kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<08:42, 1.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<06:43, 1.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:58, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:39, 2.61MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<08:22, 1.14MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<08:58, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<06:55, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:13, 1.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:47, 2.50MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<08:00, 1.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<08:20, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:26, 1.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:42, 2.00MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:26, 2.73MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<19:22, 484kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<16:14, 578kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<12:09, 771kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:53, 1.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<06:27, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<04:45, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<09:56, 936kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<10:12, 912kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:59, 1.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:47, 1.60MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<06:27, 1.43MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:34, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:02, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:43, 2.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:58, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:53, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:42, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<03:25, 2.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:14, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:13, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:49, 1.88MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:30, 2.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:56, 1.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:16, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:54, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:32, 2.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:28, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:32, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:08, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:38, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:28, 2.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<02:35, 3.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<37:57, 234kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<28:59, 307kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<20:51, 426kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<14:41, 602kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<14:23, 614kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<11:27, 770kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<08:21, 1.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<05:55, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<18:49, 466kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<15:02, 582kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<10:54, 802kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<07:43, 1.13MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<09:42, 895kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<08:34, 1.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<06:26, 1.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<04:34, 1.89MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<14:55, 578kB/s] .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<12:17, 702kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<09:02, 953kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<06:23, 1.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<45:20, 189kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<33:33, 255kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<23:52, 358kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<16:42, 508kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<25:31, 333kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<19:02, 446kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<13:35, 623kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<11:03, 761kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<09:15, 909kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<06:47, 1.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<04:54, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:07, 1.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:14, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:49, 1.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<03:28, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<12:26, 665kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<10:28, 790kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<07:40, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<05:31, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:15, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:07, 1.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:42, 1.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<03:22, 2.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<19:12, 424kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<15:06, 539kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<10:56, 742kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<07:46, 1.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<08:25, 959kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:40, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:48, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:22, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<03:09, 2.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<57:28, 139kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<43:02, 186kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<30:49, 259kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<21:40, 367kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<17:11, 462kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<14:47, 536kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<11:04, 716kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<07:53, 1.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:33, 1.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<09:07, 862kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<07:22, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:20, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:53, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<06:07, 1.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<08:12, 949kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:59, 1.11MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:21, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:57, 1.96MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:56, 2.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<05:51, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<05:54, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:30, 1.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:19, 2.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:44, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:43, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:34, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:06, 1.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:01, 2.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<09:27, 802kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<10:00, 758kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<07:50, 967kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:40, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<04:04, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<10:17, 730kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<10:34, 712kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<08:13, 913kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:57, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<04:17, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<08:30, 876kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<09:16, 803kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<07:16, 1.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:15, 1.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<04:02, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<04:44, 1.56MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<07:52, 937kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<06:38, 1.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:52, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:38, 2.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:45, 2.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<06:53, 1.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<06:56, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:23, 1.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:58, 1.83MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:55, 2.47MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<08:37, 840kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<10:12, 710kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<08:20, 867kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<06:20, 1.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:37, 1.56MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:25, 2.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:55, 1.45MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<05:24, 1.33MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:16, 1.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:10, 2.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:04<02:23, 2.98MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<10:40, 666kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<11:19, 627kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<08:56, 794kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:27, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:43, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<03:26, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<12:38, 556kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<12:40, 555kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<09:46, 719kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:04, 991kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<05:06, 1.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<06:32, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<08:26, 825kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<06:42, 1.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:06, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:42, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:48, 2.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:04, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<05:17, 1.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:03, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:03, 2.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:18, 2.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:47, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<07:03, 968kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:50, 1.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:17, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:12, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:25, 2.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:30, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:35, 1.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:20, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:10, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:22, 2.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<06:26, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<08:08, 822kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<06:33, 1.02MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:48, 1.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<03:30, 1.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:17, 1.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:22, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:37, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:36, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:48, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:16, 2.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<01:52, 3.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<01:36, 4.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<05:34, 1.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<06:03, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<04:45, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:29, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:35, 2.51MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<10:21, 626kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<10:46, 601kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<08:22, 773kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:04, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:22, 1.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:58, 1.07MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<07:20, 874kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:59, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:22, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:13, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:25, 2.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<19:17, 329kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<15:03, 421kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<10:55, 580kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<07:47, 810kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<05:34, 1.13MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<21:28, 292kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<18:27, 340kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<13:43, 457kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<09:48, 638kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<06:58, 893kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<07:38, 813kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<08:28, 732kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<06:43, 922kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:54, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:36, 1.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:08, 1.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<07:54, 776kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<06:39, 922kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:55, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:37, 1.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:42, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<07:23, 821kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<07:09, 848kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<05:25, 1.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:04, 1.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:07, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:27, 2.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:00, 3.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<01:41, 3.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<09:34, 626kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<08:49, 680kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<06:40, 898kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<04:48, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:31, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:37, 2.27MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<35:27, 167kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<28:04, 211kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<20:31, 289kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<14:31, 407kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<10:18, 572kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<07:20, 800kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<09:24, 623kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<09:55, 591kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<07:56, 737kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:02, 970kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<04:33, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:28, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:39, 2.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:02, 2.84MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:30, 1.28MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:44, 1.22MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:41, 1.56MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:45, 2.09MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:03, 2.78MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<06:50, 838kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<07:54, 724kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<06:16, 912kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:35, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:20, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:45, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<06:33, 863kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:34, 1.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:19, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:18, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:34, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:03, 2.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<01:42, 3.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:27, 3.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<07:44, 722kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<06:19, 882kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<04:39, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<03:23, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:14, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<05:39, 976kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:37, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:24, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:30, 2.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:33, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<05:53, 925kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<04:47, 1.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:31, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:34, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<01:56, 2.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<55:30, 96.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<41:29, 130kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<29:41, 181kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<20:54, 256kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<14:41, 363kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<12:39, 419kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<12:34, 422kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<09:45, 543kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<07:02, 751kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<05:04, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<03:42, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<07:24, 707kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<06:53, 760kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:15, 995kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:50, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:48, 1.85MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<04:21, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<04:44, 1.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:44, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:44, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<02:03, 2.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<04:02, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<06:32, 781kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<05:25, 939kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<04:10, 1.22MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:05, 1.64MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:17, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:46, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:37, 1.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<05:16, 953kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<04:09, 1.21MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:12, 1.57MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:23, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:55, 2.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:29, 3.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:17, 3.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<1:52:40, 44.1kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<1:20:52, 61.4kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<57:01, 86.9kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<39:58, 124kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<28:03, 176kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<19:48, 248kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<13:59, 350kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<47:34, 103kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<35:18, 139kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<25:09, 194kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<17:44, 275kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<12:32, 387kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<08:54, 543kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<12:38, 382kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<10:40, 452kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<07:54, 609kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<05:41, 843kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<04:08, 1.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<03:02, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<12:57, 367kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<10:44, 443kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<07:56, 598kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:43, 828kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<04:07, 1.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<03:02, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<14:43, 319kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<12:03, 389kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<08:47, 533kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<06:19, 737kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<04:32, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<03:26, 1.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<05:53, 783kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<06:26, 717kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<05:07, 902kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:47, 1.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:50, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:11, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:43, 2.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<09:38, 472kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<15:06, 301kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<13:22, 340kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<09:52, 460kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<07:04, 641kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:06, 885kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<03:49, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<02:49, 1.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<05:12, 860kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<05:51, 766kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<04:45, 942kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<04:12, 1.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:39, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:29, 1.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:15, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:54, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:07, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:54, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:57, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:47, 1.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:49, 1.58MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:41, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:43, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:35, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:37, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:31, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:37, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:29, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:33, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:26, 1.81MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:31, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:24, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:27, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:20, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:23, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:44, 1.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<02:27, 1.79MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:37, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:44, 1.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:48, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:49, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:52, 1.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<02:48, 1.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:49, 1.55MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:45, 1.59MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:41, 1.62MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:36, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:35, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:37, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:32, 1.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:26, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:26, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:25, 1.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:27, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:21, 1.84MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:25, 1.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:18, 1.87MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:25, 1.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:16, 1.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:21, 1.84MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:22, 1.82MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:18, 1.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:16, 1.89MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:15, 1.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:16, 1.90MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<02:11, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<02:12, 1.95MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<02:14, 1.92MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<02:13, 1.94MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:12, 1.94MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:08, 2.00MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:13, 1.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:13, 1.92MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:10, 1.96MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<02:09, 1.98MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<02:11, 1.95MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<02:07, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<02:04, 2.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<02:10, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:10, 1.96MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:13, 1.92MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:14, 1.90MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:12, 1.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:13, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:29<02:08, 1.98MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:29<02:09, 1.96MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<02:04, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<02:04, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<02:01, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<01:59, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<01:57, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<01:55, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<01:54, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:52, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<01:51, 2.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<01:50, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<01:49, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<01:50, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<01:49, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<01:46, 2.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<01:47, 2.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<01:44, 2.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<01:44, 2.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<01:42, 2.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<01:41, 2.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<01:40, 2.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<01:41, 2.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<01:40, 2.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<01:38, 2.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<01:39, 2.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<01:36, 2.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<01:36, 2.56MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<01:35, 2.60MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<01:34, 2.61MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<01:32, 2.67MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<01:34, 2.62MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<01:31, 2.70MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<01:34, 2.62MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<01:32, 2.66MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<01:32, 2.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:28, 2.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:31, 2.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:28, 2.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<01:40, 2.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<01:44, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<01:54, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<01:49, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<01:48, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:48, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:46, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:51, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:43, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:50, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:41, 2.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:46, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:38, 2.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:42, 2.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:36, 2.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:37, 2.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:33, 2.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:35, 2.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:35, 2.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<01:28, 2.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:31, 2.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:27, 2.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:30, 2.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:31, 2.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:28, 2.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:31, 2.61MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:40, 2.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:51, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:01, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:08, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:12, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:14, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:11, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:59, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:52, 2.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:44, 2.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:50, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:45, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:39, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:33, 2.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:30, 2.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:27, 2.67MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:26, 2.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:24, 2.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:22, 2.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:47, 1.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<03:03, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:31, 1.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:12, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:54, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:47, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:36, 2.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:34, 2.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:27, 2.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:28, 2.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:22, 2.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:25, 2.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:21, 2.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:22, 2.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<04:21, 877kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:37, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:56, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:27, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:06, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:51, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:37, 2.32MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:34, 2.41MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:28, 2.56MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:21, 2.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:23, 2.70MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:21, 2.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<04:56, 760kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<04:04, 920kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<03:20, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:43, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:15, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:56, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:42, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:32, 2.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:24, 2.64MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:19, 2.80MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:15, 2.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:13, 3.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:10, 3.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:08, 3.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<11:33, 319kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<08:55, 413kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<06:34, 559kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<04:54, 748kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:43, 984kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:49, 1.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:19, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:55, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:36, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:19, 2.72MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<07:57, 455kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<06:14, 579kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<04:39, 774kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<03:29, 1.03MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:40, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:03, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:39, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:24, 2.52MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<04:49, 736kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<04:19, 819kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<03:43, 953kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:48, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:11, 1.61MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:42, 2.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:23, 2.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:09, 3.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:04, 3.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:00, 3.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:33, 979kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<03:48, 915kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:59, 1.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:32, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:58, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:41, 2.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:29, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:21, 2.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:14, 2.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:09, 2.97MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<01:07, 3.03MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<08:37, 395kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<06:51, 497kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<05:06, 667kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<03:52, 878kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:58, 1.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:22, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<01:55, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:37, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:23, 2.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:15, 2.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:08, 2.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:05, 3.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<03:39, 912kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<03:16, 1.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:34, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:03, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:41, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:26, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:17, 2.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:11, 2.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:07, 2.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:05, 3.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:00, 3.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:23, 964kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:02, 1.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:24, 1.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:56, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:36, 2.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:22, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:13, 2.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:06, 2.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<00:57, 3.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<00:57, 3.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<14:25, 222kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<10:45, 297kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<07:46, 411kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<05:44, 556kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<04:15, 748kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:13, 985kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:29, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:59, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:37, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:24, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:13, 2.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:06, 2.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<06:17, 499kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<05:02, 622kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<03:46, 829kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:51, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:11, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:44, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:23, 2.21MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:19, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:07, 2.74MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:18, 929kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:35, 854kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:55, 1.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:20, 1.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:52, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:31, 1.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:17, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:06, 2.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:00, 2.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<00:53, 3.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<00:53, 3.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<00:48, 3.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<06:11, 484kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<04:53, 612kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<03:38, 820kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:44, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:07, 1.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:40, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:20, 2.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:07, 2.60MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<00:58, 3.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<09:28, 309kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<07:38, 383kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<05:34, 524kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<04:03, 718kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<03:02, 956kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:17, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:46, 1.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:25, 2.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:09, 2.48MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<00:58, 2.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<11:21, 252kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<08:45, 326kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<06:21, 448kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<04:34, 621kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:23, 834kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:31, 1.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:54, 1.48MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:29, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:11, 2.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<07:20, 380kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<05:53, 473kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<04:18, 645kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<03:08, 881kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<02:18, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:43, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:20, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<04:36, 591kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<04:36, 590kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:32, 767kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:39, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:56, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:28, 1.81MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:08, 2.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<00:53, 2.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<04:50, 547kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<04:22, 605kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<03:19, 797kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:25, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:46, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:19, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<04:09, 621kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:48, 678kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:52, 896kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:04, 1.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:32, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<01:08, 2.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:26, 1.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<03:07, 807kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:32, 985kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:51, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:23, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:02, 2.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<00:47, 3.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<15:02, 163kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<11:08, 219kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<07:55, 307kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<05:33, 434kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<03:54, 611kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<04:10, 570kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<04:05, 580kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<03:08, 756kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:15, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<01:36, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:36, 886kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:47, 828kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:10, 1.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:38, 1.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:10, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<00:52, 2.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<06:12, 361kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<05:10, 432kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<03:49, 584kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<02:42, 816kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:20, 926kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:23, 904kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:50, 1.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:19, 1.61MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:57, 2.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:53, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:58, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:32, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:06, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:18, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:32, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:13, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:52, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:11, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:22, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:05, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:47, 2.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:11, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:18, 1.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:02, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:44, 2.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:10, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:20, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:04, 1.69MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:46, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:04, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:08, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:52, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:37, 2.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:04, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:13, 1.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:04, 1.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:48, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:35, 2.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:06, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:09, 1.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:53, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:39, 2.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:30, 3.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:11, 1.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:16, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:00, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:43, 2.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:49, 1.79MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:58, 1.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:46, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:34, 2.53MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:24, 3.42MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<08:06, 174kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<06:03, 232kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<04:18, 324kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<03:02, 453kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<02:18, 578kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<02:10, 617kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:37, 823kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:12, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:51, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:36, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<02:03, 618kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:54, 662kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<01:27, 864kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:01, 1.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:43, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:23, 860kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:26, 834kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:06, 1.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:48, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:34, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:57, 1.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:06, 1.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:51, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:39, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:27, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:45, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:54, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:42, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:30, 2.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:22, 2.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:49, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:57, 1.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:45, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:31, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:22, 2.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<03:24, 271kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<02:41, 341kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:57, 468kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:20, 659kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:55, 924kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:35, 539kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:24, 606kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:02, 815kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:43, 1.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:30, 1.57MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:52, 896kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:53, 885kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:40, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:29, 1.57MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:20, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:31, 1.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:35, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:28, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:20, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:14, 2.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:39, 976kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:41, 934kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:31, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:22, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:23, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:28, 1.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:22, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:15, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:10, 2.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<02:54, 175kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<02:11, 231kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<01:32, 323kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:04, 453kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:45, 628kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:32, 859kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:22, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:33, 787kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:34, 771kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:26, 990kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:18, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:09, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:30, 733kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:29, 752kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:21, 987kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:15, 1.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:20, 882kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:21, 853kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:16, 1.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:11, 1.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.99MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:05, 2.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:11, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:12, 1.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:09, 1.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:03, 3.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<01:08, 142kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:51, 188kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:35, 263kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:22, 371kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:14, 520kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:09, 724kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:05, 1.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:42, 132kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:31, 177kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:20, 247kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:11, 348kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:05, 489kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:02, 684kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<01:13, 20.3kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:48, 28.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:22, 41.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:06, 58.3kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<50:02:08,  2.22it/s]  0%|          | 666/400000 [00:00<34:58:18,  3.17it/s]  0%|          | 1337/400000 [00:00<24:26:38,  4.53it/s]  1%|          | 2017/400000 [00:00<17:05:11,  6.47it/s]  1%|          | 2673/400000 [00:00<11:56:45,  9.24it/s]  1%|          | 3463/400000 [00:00<8:20:58, 13.19it/s]   1%|          | 4282/400000 [00:01<5:50:12, 18.83it/s]  1%|▏         | 5099/400000 [00:01<4:04:52, 26.88it/s]  1%|▏         | 5883/400000 [00:01<2:51:19, 38.34it/s]  2%|▏         | 6707/400000 [00:01<1:59:54, 54.66it/s]  2%|▏         | 7518/400000 [00:01<1:24:00, 77.86it/s]  2%|▏         | 8292/400000 [00:01<58:58, 110.69it/s]   2%|▏         | 9037/400000 [00:01<41:30, 157.01it/s]  2%|▏         | 9810/400000 [00:01<29:14, 222.36it/s]  3%|▎         | 10547/400000 [00:01<20:42, 313.36it/s]  3%|▎         | 11271/400000 [00:01<14:45, 439.01it/s]  3%|▎         | 11997/400000 [00:02<10:34, 611.30it/s]  3%|▎         | 12759/400000 [00:02<07:38, 844.26it/s]  3%|▎         | 13539/400000 [00:02<05:35, 1152.56it/s]  4%|▎         | 14284/400000 [00:02<04:11, 1536.48it/s]  4%|▍         | 15084/400000 [00:02<03:09, 2028.02it/s]  4%|▍         | 15890/400000 [00:02<02:26, 2615.03it/s]  4%|▍         | 16723/400000 [00:02<01:56, 3292.65it/s]  4%|▍         | 17510/400000 [00:02<01:37, 3927.05it/s]  5%|▍         | 18275/400000 [00:02<01:24, 4510.28it/s]  5%|▍         | 19018/400000 [00:03<01:16, 4950.67it/s]  5%|▍         | 19907/400000 [00:03<01:06, 5708.83it/s]  5%|▌         | 20768/400000 [00:03<00:59, 6350.04it/s]  5%|▌         | 21653/400000 [00:03<00:54, 6937.80it/s]  6%|▌         | 22529/400000 [00:03<00:51, 7398.57it/s]  6%|▌         | 23403/400000 [00:03<00:48, 7755.35it/s]  6%|▌         | 24275/400000 [00:03<00:46, 8020.79it/s]  6%|▋         | 25142/400000 [00:03<00:45, 8202.85it/s]  7%|▋         | 26032/400000 [00:03<00:44, 8399.72it/s]  7%|▋         | 26912/400000 [00:03<00:43, 8515.36it/s]  7%|▋         | 27784/400000 [00:04<00:44, 8413.46it/s]  7%|▋         | 28640/400000 [00:04<00:47, 7756.88it/s]  7%|▋         | 29436/400000 [00:04<00:50, 7333.38it/s]  8%|▊         | 30189/400000 [00:04<00:51, 7181.33it/s]  8%|▊         | 30922/400000 [00:04<00:52, 6983.83it/s]  8%|▊         | 31724/400000 [00:04<00:50, 7263.43it/s]  8%|▊         | 32535/400000 [00:04<00:49, 7489.26it/s]  8%|▊         | 33357/400000 [00:04<00:47, 7693.62it/s]  9%|▊         | 34189/400000 [00:04<00:46, 7869.66it/s]  9%|▊         | 34983/400000 [00:05<00:47, 7684.91it/s]  9%|▉         | 35757/400000 [00:05<00:50, 7267.58it/s]  9%|▉         | 36493/400000 [00:05<00:51, 7075.20it/s]  9%|▉         | 37208/400000 [00:05<00:52, 6923.82it/s]  9%|▉         | 37907/400000 [00:05<00:52, 6866.79it/s] 10%|▉         | 38598/400000 [00:05<00:53, 6702.44it/s] 10%|▉         | 39272/400000 [00:05<00:54, 6666.94it/s] 10%|▉         | 39954/400000 [00:05<00:53, 6710.53it/s] 10%|█         | 40627/400000 [00:05<00:54, 6616.87it/s] 10%|█         | 41291/400000 [00:05<00:54, 6607.93it/s] 10%|█         | 41965/400000 [00:06<00:53, 6645.93it/s] 11%|█         | 42631/400000 [00:06<00:53, 6645.26it/s] 11%|█         | 43297/400000 [00:06<00:54, 6595.23it/s] 11%|█         | 43960/400000 [00:06<00:53, 6604.02it/s] 11%|█         | 44650/400000 [00:06<00:53, 6689.15it/s] 11%|█▏        | 45363/400000 [00:06<00:52, 6813.35it/s] 12%|█▏        | 46046/400000 [00:06<00:52, 6782.21it/s] 12%|█▏        | 46725/400000 [00:06<00:52, 6739.48it/s] 12%|█▏        | 47400/400000 [00:06<00:52, 6689.52it/s] 12%|█▏        | 48070/400000 [00:06<00:52, 6690.58it/s] 12%|█▏        | 48772/400000 [00:07<00:51, 6785.45it/s] 12%|█▏        | 49452/400000 [00:07<00:51, 6779.94it/s] 13%|█▎        | 50131/400000 [00:07<00:52, 6664.74it/s] 13%|█▎        | 50799/400000 [00:07<00:52, 6602.60it/s] 13%|█▎        | 51465/400000 [00:07<00:52, 6616.85it/s] 13%|█▎        | 52128/400000 [00:07<00:52, 6583.56it/s] 13%|█▎        | 52796/400000 [00:07<00:52, 6611.26it/s] 13%|█▎        | 53467/400000 [00:07<00:52, 6640.43it/s] 14%|█▎        | 54248/400000 [00:07<00:49, 6952.38it/s] 14%|█▍        | 55048/400000 [00:08<00:47, 7236.06it/s] 14%|█▍        | 55778/400000 [00:08<00:48, 7109.48it/s] 14%|█▍        | 56494/400000 [00:08<00:49, 6999.20it/s] 14%|█▍        | 57198/400000 [00:08<00:48, 7010.93it/s] 14%|█▍        | 57975/400000 [00:08<00:47, 7221.84it/s] 15%|█▍        | 58770/400000 [00:08<00:45, 7423.51it/s] 15%|█▍        | 59557/400000 [00:08<00:45, 7551.46it/s] 15%|█▌        | 60374/400000 [00:08<00:43, 7725.36it/s] 15%|█▌        | 61177/400000 [00:08<00:43, 7814.18it/s] 15%|█▌        | 61961/400000 [00:08<00:44, 7564.70it/s] 16%|█▌        | 62721/400000 [00:09<00:46, 7250.01it/s] 16%|█▌        | 63452/400000 [00:09<00:47, 7080.01it/s] 16%|█▌        | 64165/400000 [00:09<00:47, 7066.16it/s] 16%|█▌        | 64875/400000 [00:09<00:48, 6915.19it/s] 16%|█▋        | 65570/400000 [00:09<00:48, 6888.58it/s] 17%|█▋        | 66261/400000 [00:09<00:48, 6825.11it/s] 17%|█▋        | 66961/400000 [00:09<00:48, 6874.92it/s] 17%|█▋        | 67761/400000 [00:09<00:46, 7177.19it/s] 17%|█▋        | 68545/400000 [00:09<00:45, 7362.25it/s] 17%|█▋        | 69286/400000 [00:09<00:44, 7357.56it/s] 18%|█▊        | 70107/400000 [00:10<00:43, 7591.88it/s] 18%|█▊        | 70927/400000 [00:10<00:42, 7763.88it/s] 18%|█▊        | 71716/400000 [00:10<00:42, 7798.99it/s] 18%|█▊        | 72517/400000 [00:10<00:41, 7860.76it/s] 18%|█▊        | 73328/400000 [00:10<00:41, 7931.36it/s] 19%|█▊        | 74123/400000 [00:10<00:42, 7595.96it/s] 19%|█▊        | 74887/400000 [00:10<00:43, 7450.52it/s] 19%|█▉        | 75636/400000 [00:10<00:45, 7145.98it/s] 19%|█▉        | 76356/400000 [00:10<00:46, 7026.01it/s] 19%|█▉        | 77063/400000 [00:11<00:46, 6936.49it/s] 19%|█▉        | 77760/400000 [00:11<00:46, 6861.49it/s] 20%|█▉        | 78449/400000 [00:11<00:47, 6835.84it/s] 20%|█▉        | 79135/400000 [00:11<00:47, 6788.65it/s] 20%|█▉        | 79816/400000 [00:11<00:47, 6783.05it/s] 20%|██        | 80496/400000 [00:11<00:47, 6774.57it/s] 20%|██        | 81197/400000 [00:11<00:46, 6841.29it/s] 20%|██        | 81882/400000 [00:11<00:46, 6801.39it/s] 21%|██        | 82563/400000 [00:11<00:46, 6795.65it/s] 21%|██        | 83243/400000 [00:11<00:47, 6731.30it/s] 21%|██        | 83917/400000 [00:12<00:47, 6665.35it/s] 21%|██        | 84584/400000 [00:12<00:48, 6529.70it/s] 21%|██▏       | 85238/400000 [00:12<00:48, 6506.35it/s] 21%|██▏       | 85910/400000 [00:12<00:47, 6567.10it/s] 22%|██▏       | 86568/400000 [00:12<00:47, 6567.38it/s] 22%|██▏       | 87226/400000 [00:12<00:47, 6547.11it/s] 22%|██▏       | 87918/400000 [00:12<00:46, 6652.45it/s] 22%|██▏       | 88584/400000 [00:12<00:46, 6634.18it/s] 22%|██▏       | 89258/400000 [00:12<00:46, 6663.07it/s] 22%|██▏       | 89940/400000 [00:12<00:46, 6709.02it/s] 23%|██▎       | 90612/400000 [00:13<00:46, 6683.10it/s] 23%|██▎       | 91281/400000 [00:13<00:46, 6648.03it/s] 23%|██▎       | 91962/400000 [00:13<00:46, 6694.22it/s] 23%|██▎       | 92775/400000 [00:13<00:43, 7066.87it/s] 23%|██▎       | 93595/400000 [00:13<00:41, 7370.86it/s] 24%|██▎       | 94431/400000 [00:13<00:39, 7640.21it/s] 24%|██▍       | 95265/400000 [00:13<00:38, 7837.28it/s] 24%|██▍       | 96090/400000 [00:13<00:38, 7954.52it/s] 24%|██▍       | 96911/400000 [00:13<00:37, 8026.61it/s] 24%|██▍       | 97718/400000 [00:13<00:38, 7797.75it/s] 25%|██▍       | 98502/400000 [00:14<00:41, 7346.62it/s] 25%|██▍       | 99245/400000 [00:14<00:42, 7069.48it/s] 25%|██▍       | 99960/400000 [00:14<00:43, 6854.76it/s] 25%|██▌       | 100653/400000 [00:14<00:44, 6768.02it/s] 25%|██▌       | 101335/400000 [00:14<00:44, 6682.73it/s] 26%|██▌       | 102042/400000 [00:14<00:43, 6793.07it/s] 26%|██▌       | 102860/400000 [00:14<00:41, 7156.56it/s] 26%|██▌       | 103595/400000 [00:14<00:41, 7212.31it/s] 26%|██▌       | 104322/400000 [00:14<00:41, 7048.68it/s] 26%|██▋       | 105105/400000 [00:15<00:40, 7264.56it/s] 26%|██▋       | 105880/400000 [00:15<00:39, 7402.17it/s] 27%|██▋       | 106625/400000 [00:15<00:39, 7411.18it/s] 27%|██▋       | 107369/400000 [00:15<00:41, 7134.91it/s] 27%|██▋       | 108087/400000 [00:15<00:42, 6924.53it/s] 27%|██▋       | 108868/400000 [00:15<00:40, 7166.96it/s] 27%|██▋       | 109590/400000 [00:15<00:41, 7080.26it/s] 28%|██▊       | 110388/400000 [00:15<00:39, 7326.39it/s] 28%|██▊       | 111203/400000 [00:15<00:38, 7554.26it/s] 28%|██▊       | 112030/400000 [00:15<00:37, 7755.47it/s] 28%|██▊       | 112845/400000 [00:16<00:36, 7868.62it/s] 28%|██▊       | 113677/400000 [00:16<00:35, 7996.57it/s] 29%|██▊       | 114480/400000 [00:16<00:36, 7790.65it/s] 29%|██▉       | 115302/400000 [00:16<00:35, 7912.17it/s] 29%|██▉       | 116129/400000 [00:16<00:35, 8015.26it/s] 29%|██▉       | 116957/400000 [00:16<00:34, 8090.80it/s] 29%|██▉       | 117787/400000 [00:16<00:34, 8150.72it/s] 30%|██▉       | 118604/400000 [00:16<00:34, 8138.83it/s] 30%|██▉       | 119419/400000 [00:16<00:35, 7981.69it/s] 30%|███       | 120219/400000 [00:16<00:37, 7525.01it/s] 30%|███       | 120978/400000 [00:17<00:38, 7249.21it/s] 30%|███       | 121710/400000 [00:17<00:39, 7019.60it/s] 31%|███       | 122419/400000 [00:17<00:39, 6951.97it/s] 31%|███       | 123119/400000 [00:17<00:40, 6774.27it/s] 31%|███       | 123801/400000 [00:17<00:41, 6655.28it/s] 31%|███       | 124470/400000 [00:17<00:41, 6575.33it/s] 31%|███▏      | 125161/400000 [00:17<00:41, 6670.98it/s] 31%|███▏      | 125857/400000 [00:17<00:40, 6755.10it/s] 32%|███▏      | 126689/400000 [00:17<00:38, 7159.02it/s] 32%|███▏      | 127413/400000 [00:18<00:38, 7147.83it/s] 32%|███▏      | 128238/400000 [00:18<00:36, 7445.52it/s] 32%|███▏      | 129072/400000 [00:18<00:35, 7691.93it/s] 32%|███▏      | 129897/400000 [00:18<00:34, 7850.42it/s] 33%|███▎      | 130731/400000 [00:18<00:33, 7990.37it/s] 33%|███▎      | 131535/400000 [00:18<00:34, 7756.25it/s] 33%|███▎      | 132316/400000 [00:18<00:35, 7463.08it/s] 33%|███▎      | 133068/400000 [00:18<00:37, 7194.30it/s] 33%|███▎      | 133794/400000 [00:18<00:37, 7146.64it/s] 34%|███▎      | 134513/400000 [00:19<00:38, 6959.70it/s] 34%|███▍      | 135214/400000 [00:19<00:38, 6921.24it/s] 34%|███▍      | 135910/400000 [00:19<00:38, 6817.54it/s] 34%|███▍      | 136595/400000 [00:19<00:38, 6779.40it/s] 34%|███▍      | 137301/400000 [00:19<00:38, 6860.82it/s] 34%|███▍      | 137989/400000 [00:19<00:38, 6860.75it/s] 35%|███▍      | 138764/400000 [00:19<00:36, 7103.63it/s] 35%|███▍      | 139478/400000 [00:19<00:38, 6851.95it/s] 35%|███▌      | 140178/400000 [00:19<00:37, 6893.40it/s] 35%|███▌      | 140963/400000 [00:19<00:36, 7154.91it/s] 35%|███▌      | 141782/400000 [00:20<00:34, 7435.52it/s] 36%|███▌      | 142607/400000 [00:20<00:33, 7661.33it/s] 36%|███▌      | 143433/400000 [00:20<00:32, 7829.55it/s] 36%|███▌      | 144242/400000 [00:20<00:32, 7904.92it/s] 36%|███▋      | 145063/400000 [00:20<00:31, 7992.24it/s] 36%|███▋      | 145891/400000 [00:20<00:31, 8074.87it/s] 37%|███▋      | 146713/400000 [00:20<00:31, 8117.56it/s] 37%|███▋      | 147527/400000 [00:20<00:31, 7898.74it/s] 37%|███▋      | 148320/400000 [00:20<00:32, 7857.18it/s] 37%|███▋      | 149108/400000 [00:20<00:32, 7749.44it/s] 37%|███▋      | 149919/400000 [00:21<00:31, 7853.90it/s] 38%|███▊      | 150706/400000 [00:21<00:33, 7492.90it/s] 38%|███▊      | 151460/400000 [00:21<00:34, 7237.59it/s] 38%|███▊      | 152189/400000 [00:21<00:34, 7227.30it/s] 38%|███▊      | 152943/400000 [00:21<00:33, 7316.36it/s] 38%|███▊      | 153678/400000 [00:21<00:35, 6996.56it/s] 39%|███▊      | 154383/400000 [00:21<00:35, 6868.66it/s] 39%|███▉      | 155074/400000 [00:21<00:36, 6759.88it/s] 39%|███▉      | 155754/400000 [00:21<00:36, 6737.03it/s] 39%|███▉      | 156430/400000 [00:22<00:36, 6725.63it/s] 39%|███▉      | 157105/400000 [00:22<00:36, 6589.30it/s] 39%|███▉      | 157766/400000 [00:22<00:36, 6588.26it/s] 40%|███▉      | 158469/400000 [00:22<00:35, 6714.82it/s] 40%|███▉      | 159241/400000 [00:22<00:34, 6987.60it/s] 40%|████      | 160067/400000 [00:22<00:32, 7324.35it/s] 40%|████      | 160869/400000 [00:22<00:31, 7517.79it/s] 40%|████      | 161680/400000 [00:22<00:31, 7684.96it/s] 41%|████      | 162507/400000 [00:22<00:30, 7848.89it/s] 41%|████      | 163297/400000 [00:22<00:31, 7536.49it/s] 41%|████      | 164057/400000 [00:23<00:32, 7223.09it/s] 41%|████      | 164787/400000 [00:23<00:33, 7004.71it/s] 41%|████▏     | 165499/400000 [00:23<00:33, 7038.54it/s] 42%|████▏     | 166365/400000 [00:23<00:31, 7455.82it/s] 42%|████▏     | 167207/400000 [00:23<00:30, 7719.70it/s] 42%|████▏     | 167989/400000 [00:23<00:30, 7710.88it/s] 42%|████▏     | 168767/400000 [00:23<00:30, 7694.50it/s] 42%|████▏     | 169590/400000 [00:23<00:29, 7845.67it/s] 43%|████▎     | 170392/400000 [00:23<00:29, 7896.82it/s] 43%|████▎     | 171220/400000 [00:23<00:28, 8006.21it/s] 43%|████▎     | 172050/400000 [00:24<00:28, 8090.00it/s] 43%|████▎     | 172861/400000 [00:24<00:28, 8040.21it/s] 43%|████▎     | 173689/400000 [00:24<00:27, 8110.64it/s] 44%|████▎     | 174502/400000 [00:24<00:27, 8113.26it/s] 44%|████▍     | 175315/400000 [00:24<00:30, 7409.62it/s] 44%|████▍     | 176069/400000 [00:24<00:31, 7152.23it/s] 44%|████▍     | 176795/400000 [00:24<00:32, 6964.50it/s] 44%|████▍     | 177500/400000 [00:24<00:32, 6800.33it/s] 45%|████▍     | 178187/400000 [00:24<00:32, 6761.98it/s] 45%|████▍     | 178869/400000 [00:25<00:33, 6573.39it/s] 45%|████▍     | 179531/400000 [00:25<00:33, 6552.61it/s] 45%|████▌     | 180219/400000 [00:25<00:33, 6647.42it/s] 45%|████▌     | 181040/400000 [00:25<00:31, 7048.18it/s] 45%|████▌     | 181847/400000 [00:25<00:29, 7325.06it/s] 46%|████▌     | 182662/400000 [00:25<00:28, 7554.08it/s] 46%|████▌     | 183467/400000 [00:25<00:28, 7695.73it/s] 46%|████▌     | 184277/400000 [00:25<00:27, 7810.78it/s] 46%|████▋     | 185096/400000 [00:25<00:27, 7918.55it/s] 46%|████▋     | 185907/400000 [00:25<00:26, 7974.33it/s] 47%|████▋     | 186708/400000 [00:26<00:28, 7594.26it/s] 47%|████▋     | 187474/400000 [00:26<00:29, 7237.44it/s] 47%|████▋     | 188206/400000 [00:26<00:30, 7048.74it/s] 47%|████▋     | 188918/400000 [00:26<00:30, 6941.86it/s] 47%|████▋     | 189617/400000 [00:26<00:30, 6885.20it/s] 48%|████▊     | 190309/400000 [00:26<00:31, 6701.72it/s] 48%|████▊     | 191073/400000 [00:26<00:30, 6955.76it/s] 48%|████▊     | 191868/400000 [00:26<00:28, 7226.50it/s] 48%|████▊     | 192673/400000 [00:26<00:27, 7453.88it/s] 48%|████▊     | 193425/400000 [00:27<00:28, 7156.93it/s] 49%|████▊     | 194240/400000 [00:27<00:27, 7426.62it/s] 49%|████▉     | 195027/400000 [00:27<00:27, 7552.29it/s] 49%|████▉     | 195823/400000 [00:27<00:26, 7667.50it/s] 49%|████▉     | 196642/400000 [00:27<00:26, 7813.15it/s] 49%|████▉     | 197464/400000 [00:27<00:25, 7929.86it/s] 50%|████▉     | 198283/400000 [00:27<00:25, 8005.15it/s] 50%|████▉     | 199086/400000 [00:27<00:26, 7726.42it/s] 50%|████▉     | 199905/400000 [00:27<00:25, 7858.38it/s] 50%|█████     | 200696/400000 [00:27<00:25, 7872.37it/s] 50%|█████     | 201517/400000 [00:28<00:24, 7969.61it/s] 51%|█████     | 202316/400000 [00:28<00:26, 7456.97it/s] 51%|█████     | 203070/400000 [00:28<00:27, 7230.90it/s] 51%|█████     | 203815/400000 [00:28<00:26, 7293.06it/s] 51%|█████     | 204550/400000 [00:28<00:28, 6862.21it/s] 51%|█████▏    | 205269/400000 [00:28<00:27, 6957.31it/s] 52%|█████▏    | 206035/400000 [00:28<00:27, 7152.39it/s] 52%|█████▏    | 206827/400000 [00:28<00:26, 7365.99it/s] 52%|█████▏    | 207570/400000 [00:28<00:26, 7179.48it/s] 52%|█████▏    | 208293/400000 [00:29<00:27, 7006.10it/s] 52%|█████▏    | 208999/400000 [00:29<00:27, 6824.04it/s] 52%|█████▏    | 209750/400000 [00:29<00:27, 7014.43it/s] 53%|█████▎    | 210632/400000 [00:29<00:25, 7472.64it/s] 53%|█████▎    | 211520/400000 [00:29<00:24, 7844.83it/s] 53%|█████▎    | 212318/400000 [00:29<00:25, 7395.97it/s] 53%|█████▎    | 213072/400000 [00:29<00:26, 7025.79it/s] 53%|█████▎    | 213789/400000 [00:29<00:27, 6853.14it/s] 54%|█████▎    | 214485/400000 [00:29<00:27, 6740.59it/s] 54%|█████▍    | 215195/400000 [00:29<00:27, 6843.05it/s] 54%|█████▍    | 215993/400000 [00:30<00:25, 7145.28it/s] 54%|█████▍    | 216716/400000 [00:30<00:26, 6938.19it/s] 54%|█████▍    | 217417/400000 [00:30<00:26, 6826.18it/s] 55%|█████▍    | 218106/400000 [00:30<00:26, 6843.29it/s] 55%|█████▍    | 218812/400000 [00:30<00:26, 6905.41it/s] 55%|█████▍    | 219506/400000 [00:30<00:26, 6891.75it/s] 55%|█████▌    | 220198/400000 [00:30<00:26, 6839.83it/s] 55%|█████▌    | 220884/400000 [00:30<00:26, 6759.67it/s] 55%|█████▌    | 221568/400000 [00:30<00:26, 6779.67it/s] 56%|█████▌    | 222247/400000 [00:31<00:26, 6626.30it/s] 56%|█████▌    | 222911/400000 [00:31<00:26, 6618.54it/s] 56%|█████▌    | 223574/400000 [00:31<00:26, 6553.92it/s] 56%|█████▌    | 224231/400000 [00:31<00:26, 6515.60it/s] 56%|█████▌    | 224884/400000 [00:31<00:26, 6502.66it/s] 56%|█████▋    | 225536/400000 [00:31<00:26, 6506.85it/s] 57%|█████▋    | 226343/400000 [00:31<00:25, 6907.45it/s] 57%|█████▋    | 227168/400000 [00:31<00:23, 7259.90it/s] 57%|█████▋    | 227904/400000 [00:31<00:24, 7028.19it/s] 57%|█████▋    | 228616/400000 [00:31<00:24, 6885.44it/s] 57%|█████▋    | 229312/400000 [00:32<00:26, 6552.04it/s] 57%|█████▋    | 229976/400000 [00:32<00:26, 6403.19it/s] 58%|█████▊    | 230635/400000 [00:32<00:26, 6457.81it/s] 58%|█████▊    | 231334/400000 [00:32<00:25, 6606.86it/s] 58%|█████▊    | 231999/400000 [00:32<00:25, 6551.93it/s] 58%|█████▊    | 232658/400000 [00:32<00:25, 6561.03it/s] 58%|█████▊    | 233317/400000 [00:32<00:25, 6549.93it/s] 58%|█████▊    | 233974/400000 [00:32<00:25, 6539.45it/s] 59%|█████▊    | 234637/400000 [00:32<00:25, 6563.21it/s] 59%|█████▉    | 235433/400000 [00:32<00:23, 6926.28it/s] 59%|█████▉    | 236223/400000 [00:33<00:22, 7191.75it/s] 59%|█████▉    | 237009/400000 [00:33<00:22, 7379.17it/s] 59%|█████▉    | 237839/400000 [00:33<00:21, 7633.13it/s] 60%|█████▉    | 238647/400000 [00:33<00:20, 7758.77it/s] 60%|█████▉    | 239428/400000 [00:33<00:21, 7326.03it/s] 60%|██████    | 240201/400000 [00:33<00:21, 7441.77it/s] 60%|██████    | 241031/400000 [00:33<00:20, 7678.75it/s] 60%|██████    | 241824/400000 [00:33<00:20, 7752.11it/s] 61%|██████    | 242637/400000 [00:33<00:20, 7860.05it/s] 61%|██████    | 243459/400000 [00:33<00:19, 7963.64it/s] 61%|██████    | 244265/400000 [00:34<00:19, 7991.77it/s] 61%|██████▏   | 245067/400000 [00:34<00:19, 7820.00it/s] 61%|██████▏   | 245898/400000 [00:34<00:19, 7959.65it/s] 62%|██████▏   | 246718/400000 [00:34<00:19, 8029.34it/s] 62%|██████▏   | 247523/400000 [00:34<00:19, 7804.66it/s] 62%|██████▏   | 248307/400000 [00:34<00:20, 7452.28it/s] 62%|██████▏   | 249058/400000 [00:34<00:20, 7198.71it/s] 62%|██████▏   | 249784/400000 [00:34<00:21, 6983.86it/s] 63%|██████▎   | 250540/400000 [00:34<00:20, 7145.19it/s] 63%|██████▎   | 251313/400000 [00:35<00:20, 7308.90it/s] 63%|██████▎   | 252049/400000 [00:35<00:20, 7151.49it/s] 63%|██████▎   | 252768/400000 [00:35<00:20, 7039.35it/s] 63%|██████▎   | 253475/400000 [00:35<00:21, 6896.80it/s] 64%|██████▎   | 254168/400000 [00:35<00:21, 6868.67it/s] 64%|██████▎   | 254857/400000 [00:35<00:21, 6773.49it/s] 64%|██████▍   | 255536/400000 [00:35<00:21, 6739.34it/s] 64%|██████▍   | 256212/400000 [00:35<00:21, 6652.03it/s] 64%|██████▍   | 256879/400000 [00:35<00:21, 6550.00it/s] 64%|██████▍   | 257682/400000 [00:35<00:20, 6931.72it/s] 65%|██████▍   | 258382/400000 [00:36<00:20, 6872.10it/s] 65%|██████▍   | 259075/400000 [00:36<00:20, 6760.19it/s] 65%|██████▍   | 259755/400000 [00:36<00:20, 6730.76it/s] 65%|██████▌   | 260434/400000 [00:36<00:20, 6747.67it/s] 65%|██████▌   | 261111/400000 [00:36<00:20, 6736.06it/s] 65%|██████▌   | 261786/400000 [00:36<00:20, 6681.15it/s] 66%|██████▌   | 262456/400000 [00:36<00:20, 6659.45it/s] 66%|██████▌   | 263123/400000 [00:36<00:20, 6635.17it/s] 66%|██████▌   | 263799/400000 [00:36<00:20, 6670.52it/s] 66%|██████▌   | 264487/400000 [00:37<00:20, 6728.90it/s] 66%|██████▋   | 265176/400000 [00:37<00:19, 6775.69it/s] 66%|██████▋   | 265854/400000 [00:37<00:20, 6686.77it/s] 67%|██████▋   | 266524/400000 [00:37<00:20, 6665.26it/s] 67%|██████▋   | 267191/400000 [00:37<00:20, 6562.47it/s] 67%|██████▋   | 267977/400000 [00:37<00:19, 6901.97it/s] 67%|██████▋   | 268784/400000 [00:37<00:18, 7214.45it/s] 67%|██████▋   | 269528/400000 [00:37<00:17, 7278.26it/s] 68%|██████▊   | 270262/400000 [00:37<00:18, 7023.18it/s] 68%|██████▊   | 270970/400000 [00:37<00:18, 6900.68it/s] 68%|██████▊   | 271665/400000 [00:38<00:19, 6711.95it/s] 68%|██████▊   | 272341/400000 [00:38<00:19, 6592.64it/s] 68%|██████▊   | 273004/400000 [00:38<00:19, 6400.82it/s] 68%|██████▊   | 273784/400000 [00:38<00:18, 6764.29it/s] 69%|██████▊   | 274593/400000 [00:38<00:17, 7113.40it/s] 69%|██████▉   | 275412/400000 [00:38<00:16, 7403.61it/s] 69%|██████▉   | 276163/400000 [00:38<00:16, 7315.26it/s] 69%|██████▉   | 277032/400000 [00:38<00:16, 7678.82it/s] 69%|██████▉   | 277909/400000 [00:38<00:15, 7975.32it/s] 70%|██████▉   | 278778/400000 [00:38<00:14, 8174.62it/s] 70%|██████▉   | 279610/400000 [00:39<00:14, 8216.31it/s] 70%|███████   | 280490/400000 [00:39<00:14, 8381.62it/s] 70%|███████   | 281343/400000 [00:39<00:14, 8424.64it/s] 71%|███████   | 282228/400000 [00:39<00:13, 8545.62it/s] 71%|███████   | 283099/400000 [00:39<00:13, 8592.80it/s] 71%|███████   | 283968/400000 [00:39<00:13, 8620.15it/s] 71%|███████   | 284832/400000 [00:39<00:13, 8547.74it/s] 71%|███████▏  | 285688/400000 [00:39<00:13, 8438.99it/s] 72%|███████▏  | 286534/400000 [00:39<00:13, 8393.48it/s] 72%|███████▏  | 287377/400000 [00:39<00:13, 8403.40it/s] 72%|███████▏  | 288218/400000 [00:40<00:13, 8372.14it/s] 72%|███████▏  | 289056/400000 [00:40<00:13, 8165.31it/s] 72%|███████▏  | 289874/400000 [00:40<00:14, 7852.56it/s] 73%|███████▎  | 290713/400000 [00:40<00:13, 8005.42it/s] 73%|███████▎  | 291517/400000 [00:40<00:13, 7798.56it/s] 73%|███████▎  | 292301/400000 [00:40<00:13, 7727.91it/s] 73%|███████▎  | 293140/400000 [00:40<00:13, 7913.07it/s] 73%|███████▎  | 293935/400000 [00:40<00:13, 7866.80it/s] 74%|███████▎  | 294749/400000 [00:40<00:13, 7944.38it/s] 74%|███████▍  | 295593/400000 [00:41<00:12, 8086.53it/s] 74%|███████▍  | 296404/400000 [00:41<00:12, 8069.27it/s] 74%|███████▍  | 297293/400000 [00:41<00:12, 8296.83it/s] 75%|███████▍  | 298177/400000 [00:41<00:12, 8450.25it/s] 75%|███████▍  | 299025/400000 [00:41<00:12, 8213.05it/s] 75%|███████▍  | 299882/400000 [00:41<00:12, 8316.95it/s] 75%|███████▌  | 300773/400000 [00:41<00:11, 8485.88it/s] 75%|███████▌  | 301625/400000 [00:41<00:11, 8246.06it/s] 76%|███████▌  | 302494/400000 [00:41<00:11, 8372.80it/s] 76%|███████▌  | 303366/400000 [00:41<00:11, 8472.80it/s] 76%|███████▌  | 304246/400000 [00:42<00:11, 8566.56it/s] 76%|███████▋  | 305105/400000 [00:42<00:11, 8268.70it/s] 76%|███████▋  | 305985/400000 [00:42<00:11, 8418.96it/s] 77%|███████▋  | 306844/400000 [00:42<00:11, 8467.82it/s] 77%|███████▋  | 307725/400000 [00:42<00:10, 8567.60it/s] 77%|███████▋  | 308602/400000 [00:42<00:10, 8624.89it/s] 77%|███████▋  | 309483/400000 [00:42<00:10, 8679.36it/s] 78%|███████▊  | 310352/400000 [00:42<00:10, 8259.82it/s] 78%|███████▊  | 311214/400000 [00:42<00:10, 8361.84it/s] 78%|███████▊  | 312094/400000 [00:42<00:10, 8487.44it/s] 78%|███████▊  | 312969/400000 [00:43<00:10, 8563.59it/s] 78%|███████▊  | 313840/400000 [00:43<00:10, 8604.89it/s] 79%|███████▊  | 314732/400000 [00:43<00:09, 8694.20it/s] 79%|███████▉  | 315603/400000 [00:43<00:09, 8576.48it/s] 79%|███████▉  | 316481/400000 [00:43<00:09, 8634.04it/s] 79%|███████▉  | 317365/400000 [00:43<00:09, 8693.43it/s] 80%|███████▉  | 318246/400000 [00:43<00:09, 8726.19it/s] 80%|███████▉  | 319123/400000 [00:43<00:09, 8737.78it/s] 80%|███████▉  | 319998/400000 [00:43<00:09, 8724.05it/s] 80%|████████  | 320871/400000 [00:43<00:09, 8527.70it/s] 80%|████████  | 321725/400000 [00:44<00:09, 8425.87it/s] 81%|████████  | 322569/400000 [00:44<00:09, 8381.80it/s] 81%|████████  | 323409/400000 [00:44<00:09, 8343.81it/s] 81%|████████  | 324245/400000 [00:44<00:09, 8253.35it/s] 81%|████████▏ | 325071/400000 [00:44<00:09, 8185.36it/s] 81%|████████▏ | 325899/400000 [00:44<00:09, 8211.26it/s] 82%|████████▏ | 326765/400000 [00:44<00:08, 8339.79it/s] 82%|████████▏ | 327636/400000 [00:44<00:08, 8447.26it/s] 82%|████████▏ | 328510/400000 [00:44<00:08, 8532.49it/s] 82%|████████▏ | 329402/400000 [00:45<00:08, 8643.46it/s] 83%|████████▎ | 330288/400000 [00:45<00:08, 8704.80it/s] 83%|████████▎ | 331175/400000 [00:45<00:07, 8751.09it/s] 83%|████████▎ | 332051/400000 [00:45<00:07, 8523.46it/s] 83%|████████▎ | 332906/400000 [00:45<00:07, 8422.54it/s] 83%|████████▎ | 333750/400000 [00:45<00:07, 8374.04it/s] 84%|████████▎ | 334589/400000 [00:45<00:07, 8324.61it/s] 84%|████████▍ | 335471/400000 [00:45<00:07, 8467.21it/s] 84%|████████▍ | 336357/400000 [00:45<00:07, 8578.61it/s] 84%|████████▍ | 337222/400000 [00:45<00:07, 8598.69it/s] 85%|████████▍ | 338096/400000 [00:46<00:07, 8638.98it/s] 85%|████████▍ | 338961/400000 [00:46<00:07, 8435.89it/s] 85%|████████▍ | 339807/400000 [00:46<00:07, 8332.65it/s] 85%|████████▌ | 340642/400000 [00:46<00:07, 8210.36it/s] 85%|████████▌ | 341475/400000 [00:46<00:07, 8243.56it/s] 86%|████████▌ | 342355/400000 [00:46<00:06, 8400.20it/s] 86%|████████▌ | 343243/400000 [00:46<00:06, 8535.83it/s] 86%|████████▌ | 344128/400000 [00:46<00:06, 8627.44it/s] 86%|████████▋ | 345014/400000 [00:46<00:06, 8694.11it/s] 86%|████████▋ | 345895/400000 [00:46<00:06, 8727.60it/s] 87%|████████▋ | 346769/400000 [00:47<00:06, 8537.29it/s] 87%|████████▋ | 347625/400000 [00:47<00:06, 8405.45it/s] 87%|████████▋ | 348497/400000 [00:47<00:06, 8495.17it/s] 87%|████████▋ | 349350/400000 [00:47<00:05, 8504.26it/s] 88%|████████▊ | 350219/400000 [00:47<00:05, 8557.87it/s] 88%|████████▊ | 351090/400000 [00:47<00:05, 8602.02it/s] 88%|████████▊ | 351972/400000 [00:47<00:05, 8663.36it/s] 88%|████████▊ | 352851/400000 [00:47<00:05, 8698.67it/s] 88%|████████▊ | 353722/400000 [00:47<00:05, 8617.82it/s] 89%|████████▊ | 354610/400000 [00:47<00:05, 8692.56it/s] 89%|████████▉ | 355490/400000 [00:48<00:05, 8724.47it/s] 89%|████████▉ | 356363/400000 [00:48<00:05, 8003.36it/s] 89%|████████▉ | 357176/400000 [00:48<00:05, 7524.92it/s] 90%|████████▉ | 358027/400000 [00:48<00:05, 7793.32it/s] 90%|████████▉ | 358898/400000 [00:48<00:05, 8045.99it/s] 90%|████████▉ | 359754/400000 [00:48<00:04, 8191.70it/s] 90%|█████████ | 360623/400000 [00:48<00:04, 8334.22it/s] 90%|█████████ | 361498/400000 [00:48<00:04, 8450.38it/s] 91%|█████████ | 362350/400000 [00:48<00:04, 8469.85it/s] 91%|█████████ | 363201/400000 [00:49<00:04, 7731.40it/s] 91%|█████████ | 363989/400000 [00:49<00:04, 7236.16it/s] 91%|█████████ | 364730/400000 [00:49<00:05, 7015.55it/s] 91%|█████████▏| 365445/400000 [00:49<00:04, 7000.15it/s] 92%|█████████▏| 366155/400000 [00:49<00:04, 6857.17it/s] 92%|█████████▏| 366848/400000 [00:49<00:04, 6636.23it/s] 92%|█████████▏| 367519/400000 [00:49<00:04, 6631.47it/s] 92%|█████████▏| 368191/400000 [00:49<00:04, 6655.77it/s] 92%|█████████▏| 368860/400000 [00:49<00:04, 6557.26it/s] 92%|█████████▏| 369564/400000 [00:50<00:04, 6693.39it/s] 93%|█████████▎| 370370/400000 [00:50<00:04, 7051.41it/s] 93%|█████████▎| 371159/400000 [00:50<00:03, 7281.48it/s] 93%|█████████▎| 371898/400000 [00:50<00:03, 7313.62it/s] 93%|█████████▎| 372635/400000 [00:50<00:03, 7197.94it/s] 93%|█████████▎| 373359/400000 [00:50<00:03, 7005.12it/s] 94%|█████████▎| 374134/400000 [00:50<00:03, 7210.70it/s] 94%|█████████▎| 374950/400000 [00:50<00:03, 7470.72it/s] 94%|█████████▍| 375779/400000 [00:50<00:03, 7696.44it/s] 94%|█████████▍| 376606/400000 [00:50<00:02, 7858.44it/s] 94%|█████████▍| 377441/400000 [00:51<00:02, 7994.09it/s] 95%|█████████▍| 378268/400000 [00:51<00:02, 8073.65it/s] 95%|█████████▍| 379088/400000 [00:51<00:02, 8109.86it/s] 95%|█████████▍| 379902/400000 [00:51<00:02, 7523.41it/s] 95%|█████████▌| 380665/400000 [00:51<00:02, 7083.52it/s] 95%|█████████▌| 381386/400000 [00:51<00:02, 6914.63it/s] 96%|█████████▌| 382087/400000 [00:51<00:02, 6822.45it/s] 96%|█████████▌| 382777/400000 [00:51<00:02, 6743.43it/s] 96%|█████████▌| 383457/400000 [00:51<00:02, 6692.09it/s] 96%|█████████▌| 384136/400000 [00:52<00:02, 6719.21it/s] 96%|█████████▌| 384917/400000 [00:52<00:02, 7011.05it/s] 96%|█████████▋| 385684/400000 [00:52<00:01, 7194.67it/s] 97%|█████████▋| 386497/400000 [00:52<00:01, 7451.70it/s] 97%|█████████▋| 387323/400000 [00:52<00:01, 7675.09it/s] 97%|█████████▋| 388124/400000 [00:52<00:01, 7770.86it/s] 97%|█████████▋| 388954/400000 [00:52<00:01, 7920.66it/s] 97%|█████████▋| 389782/400000 [00:52<00:01, 8024.87it/s] 98%|█████████▊| 390613/400000 [00:52<00:01, 8107.26it/s] 98%|█████████▊| 391452/400000 [00:52<00:01, 8188.73it/s] 98%|█████████▊| 392273/400000 [00:53<00:00, 8183.50it/s] 98%|█████████▊| 393093/400000 [00:53<00:00, 8173.12it/s] 98%|█████████▊| 393915/400000 [00:53<00:00, 8186.74it/s] 99%|█████████▊| 394748/400000 [00:53<00:00, 8227.34it/s] 99%|█████████▉| 395577/400000 [00:53<00:00, 8245.25it/s] 99%|█████████▉| 396402/400000 [00:53<00:00, 8181.28it/s] 99%|█████████▉| 397234/400000 [00:53<00:00, 8220.71it/s]100%|█████████▉| 398057/400000 [00:53<00:00, 8147.75it/s]100%|█████████▉| 398873/400000 [00:53<00:00, 7656.42it/s]100%|█████████▉| 399647/400000 [00:53<00:00, 7681.13it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7408.75it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f0eb7627710>, <torchtext.data.dataset.TabularDataset object at 0x7f0eb7627860>, <torchtext.vocab.Vocab object at 0x7f0eb7627780>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.09 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.09 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.09 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.77 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.77 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.33 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.33 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.92 file/s]2020-06-22 01:02:40.697960: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-22 01:02:40.702440: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-22 01:02:40.702733: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fc8322f9c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-22 01:02:40.702990: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:22, 120360.52it/s] 67%|██████▋   | 6643712/9912422 [00:00<00:19, 171809.59it/s]9920512it [00:00, 36208273.90it/s]                           
0it [00:00, ?it/s]32768it [00:00, 616699.81it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161112.50it/s]1654784it [00:00, 10031286.44it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186098.50it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
