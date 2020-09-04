
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/57fcd971b3e97a7b25384251ac846601a7030760', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '57fcd971b3e97a7b25384251ac846601a7030760', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/57fcd971b3e97a7b25384251ac846601a7030760

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/57fcd971b3e97a7b25384251ac846601a7030760

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/57fcd971b3e97a7b25384251ac846601a7030760

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f841d62e6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f841d62e6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f8488357488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f8488357488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f84a7576e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f84a7576e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f843568e510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f843568e510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f843568e510> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3629056/9912422 [00:00<00:00, 36285403.94it/s]9920512it [00:00, 34617324.42it/s]                             
0it [00:00, ?it/s]32768it [00:00, 610029.13it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162303.91it/s]1654784it [00:00, 11451469.82it/s]                         
0it [00:00, ?it/s]8192it [00:00, 179864.73it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f841d562ef0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f841c8acb00>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f843568e1e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f843568e1e0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f843568e1e0> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:19,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:18,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:55,  2.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:55,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:38,  3.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:38,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:37,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:37,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:12, 10.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:12, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:12, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:12, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:12, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:12, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:08, 13.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:08, 13.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:08, 13.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 13.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 13.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 18.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 18.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 18.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 18.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 18.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 18.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 23.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 29.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 36.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 42.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 42.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 42.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 42.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 48.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 48.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 48.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 48.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 48.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 48.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 48.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 48.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 48.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 51.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 51.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 51.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 55.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 55.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 55.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 55.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 55.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 55.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 55.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 55.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 58.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 58.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 58.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 61.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 62.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 62.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 63.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 63.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 63.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 63.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 63.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 63.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 63.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 63.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.73 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.17s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.73 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.17s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.18s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.30 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.18s/ url]
0 examples [00:00, ? examples/s]2020-09-04 06:07:03.214514: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-04 06:07:03.227385: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-09-04 06:07:03.227619: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55598500a730 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-04 06:07:03.227644: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
22 examples [00:00, 218.68 examples/s]116 examples [00:00, 284.06 examples/s]211 examples [00:00, 359.62 examples/s]306 examples [00:00, 441.35 examples/s]399 examples [00:00, 523.55 examples/s]494 examples [00:00, 604.79 examples/s]585 examples [00:00, 671.68 examples/s]668 examples [00:00, 710.48 examples/s]754 examples [00:00, 748.43 examples/s]847 examples [00:01, 793.46 examples/s]939 examples [00:01, 825.74 examples/s]1035 examples [00:01, 859.96 examples/s]1125 examples [00:01, 866.77 examples/s]1219 examples [00:01, 885.85 examples/s]1310 examples [00:01, 887.78 examples/s]1401 examples [00:01, 889.36 examples/s]1492 examples [00:01, 895.09 examples/s]1590 examples [00:01, 917.84 examples/s]1683 examples [00:01, 916.45 examples/s]1777 examples [00:02, 921.37 examples/s]1874 examples [00:02, 932.63 examples/s]1968 examples [00:02, 923.49 examples/s]2061 examples [00:02, 923.30 examples/s]2157 examples [00:02, 932.40 examples/s]2251 examples [00:02, 934.57 examples/s]2350 examples [00:02, 948.75 examples/s]2448 examples [00:02, 957.52 examples/s]2548 examples [00:02, 968.32 examples/s]2645 examples [00:02, 960.15 examples/s]2742 examples [00:03, 954.87 examples/s]2838 examples [00:03, 951.96 examples/s]2934 examples [00:03, 946.33 examples/s]3033 examples [00:03, 957.22 examples/s]3130 examples [00:03, 960.08 examples/s]3227 examples [00:03, 950.98 examples/s]3325 examples [00:03, 957.98 examples/s]3422 examples [00:03, 961.02 examples/s]3521 examples [00:03, 968.55 examples/s]3618 examples [00:03, 952.57 examples/s]3714 examples [00:04, 941.72 examples/s]3814 examples [00:04, 958.43 examples/s]3915 examples [00:04, 971.47 examples/s]4017 examples [00:04, 982.95 examples/s]4116 examples [00:04, 975.29 examples/s]4214 examples [00:04, 972.31 examples/s]4312 examples [00:04, 965.58 examples/s]4409 examples [00:04, 964.61 examples/s]4511 examples [00:04, 978.57 examples/s]4609 examples [00:04, 976.31 examples/s]4707 examples [00:05, 957.90 examples/s]4806 examples [00:05, 965.99 examples/s]4903 examples [00:05, 955.85 examples/s]5001 examples [00:05, 960.21 examples/s]5098 examples [00:05, 935.56 examples/s]5196 examples [00:05, 946.52 examples/s]5291 examples [00:05, 946.33 examples/s]5386 examples [00:05, 944.97 examples/s]5481 examples [00:05, 917.22 examples/s]5573 examples [00:06, 902.33 examples/s]5664 examples [00:06, 896.26 examples/s]5759 examples [00:06, 910.42 examples/s]5851 examples [00:06, 909.54 examples/s]5947 examples [00:06, 923.49 examples/s]6040 examples [00:06, 921.10 examples/s]6137 examples [00:06, 933.60 examples/s]6231 examples [00:06, 909.44 examples/s]6323 examples [00:06, 893.01 examples/s]6419 examples [00:06, 908.55 examples/s]6512 examples [00:07, 913.83 examples/s]6608 examples [00:07, 926.33 examples/s]6701 examples [00:07, 913.08 examples/s]6793 examples [00:07, 864.33 examples/s]6883 examples [00:07, 874.40 examples/s]6972 examples [00:07, 877.81 examples/s]7068 examples [00:07, 898.79 examples/s]7167 examples [00:07, 923.15 examples/s]7264 examples [00:07, 934.73 examples/s]7361 examples [00:07, 942.79 examples/s]7456 examples [00:08, 941.90 examples/s]7555 examples [00:08, 955.78 examples/s]7656 examples [00:08, 968.88 examples/s]7758 examples [00:08, 982.31 examples/s]7857 examples [00:08, 968.54 examples/s]7955 examples [00:08, 957.55 examples/s]8051 examples [00:08, 943.64 examples/s]8146 examples [00:08, 941.21 examples/s]8241 examples [00:08, 940.32 examples/s]8339 examples [00:08, 949.87 examples/s]8435 examples [00:09, 950.67 examples/s]8533 examples [00:09, 958.12 examples/s]8629 examples [00:09, 952.25 examples/s]8726 examples [00:09, 954.83 examples/s]8824 examples [00:09, 959.85 examples/s]8921 examples [00:09, 943.09 examples/s]9016 examples [00:09, 936.18 examples/s]9110 examples [00:09, 937.29 examples/s]9204 examples [00:09, 936.48 examples/s]9299 examples [00:09, 938.34 examples/s]9393 examples [00:10, 902.25 examples/s]9485 examples [00:10, 905.38 examples/s]9580 examples [00:10, 917.58 examples/s]9672 examples [00:10, 915.42 examples/s]9764 examples [00:10, 905.93 examples/s]9858 examples [00:10, 913.02 examples/s]9954 examples [00:10, 923.99 examples/s]10047 examples [00:10, 892.61 examples/s]10137 examples [00:10, 887.86 examples/s]10227 examples [00:11, 887.17 examples/s]10319 examples [00:11, 896.57 examples/s]10416 examples [00:11, 917.38 examples/s]10512 examples [00:11, 927.66 examples/s]10607 examples [00:11, 931.60 examples/s]10701 examples [00:11, 924.84 examples/s]10797 examples [00:11, 934.61 examples/s]10893 examples [00:11, 941.50 examples/s]10992 examples [00:11, 955.01 examples/s]11088 examples [00:11, 950.45 examples/s]11184 examples [00:12, 944.09 examples/s]11279 examples [00:12, 943.19 examples/s]11374 examples [00:12, 935.68 examples/s]11471 examples [00:12, 944.86 examples/s]11571 examples [00:12, 960.60 examples/s]11668 examples [00:12, 929.94 examples/s]11762 examples [00:12, 918.44 examples/s]11856 examples [00:12, 922.33 examples/s]11949 examples [00:12, 923.17 examples/s]12043 examples [00:12, 927.76 examples/s]12142 examples [00:13, 943.31 examples/s]12237 examples [00:13, 897.20 examples/s]12328 examples [00:13, 900.15 examples/s]12423 examples [00:13, 913.27 examples/s]12520 examples [00:13, 927.34 examples/s]12614 examples [00:13, 925.55 examples/s]12707 examples [00:13, 925.65 examples/s]12800 examples [00:13, 887.83 examples/s]12890 examples [00:13, 882.19 examples/s]12980 examples [00:14, 887.04 examples/s]13079 examples [00:14, 914.22 examples/s]13171 examples [00:14, 915.23 examples/s]13263 examples [00:14, 912.26 examples/s]13355 examples [00:14, 884.14 examples/s]13452 examples [00:14, 907.39 examples/s]13551 examples [00:14, 929.57 examples/s]13648 examples [00:14, 938.95 examples/s]13747 examples [00:14, 953.38 examples/s]13843 examples [00:14, 949.60 examples/s]13944 examples [00:15, 964.46 examples/s]14046 examples [00:15, 977.92 examples/s]14144 examples [00:15, 946.27 examples/s]14240 examples [00:15, 949.59 examples/s]14338 examples [00:15, 957.38 examples/s]14434 examples [00:15, 950.93 examples/s]14530 examples [00:15, 923.65 examples/s]14625 examples [00:15, 930.87 examples/s]14725 examples [00:15, 950.19 examples/s]14821 examples [00:15, 936.14 examples/s]14920 examples [00:16, 951.04 examples/s]15016 examples [00:16, 944.99 examples/s]15114 examples [00:16, 955.07 examples/s]15216 examples [00:16, 972.40 examples/s]15319 examples [00:16, 986.71 examples/s]15421 examples [00:16, 993.38 examples/s]15521 examples [00:16, 992.78 examples/s]15621 examples [00:16, 984.53 examples/s]15720 examples [00:16, 981.42 examples/s]15819 examples [00:16, 976.03 examples/s]15917 examples [00:17, 967.95 examples/s]16018 examples [00:17, 979.03 examples/s]16116 examples [00:17, 975.40 examples/s]16214 examples [00:17, 974.32 examples/s]16315 examples [00:17, 982.47 examples/s]16414 examples [00:17, 969.27 examples/s]16513 examples [00:17, 973.24 examples/s]16611 examples [00:17, 969.45 examples/s]16712 examples [00:17, 979.33 examples/s]16810 examples [00:17, 972.36 examples/s]16908 examples [00:18, 969.88 examples/s]17006 examples [00:18, 965.70 examples/s]17103 examples [00:18, 955.08 examples/s]17199 examples [00:18, 942.86 examples/s]17294 examples [00:18, 937.78 examples/s]17400 examples [00:18, 970.72 examples/s]17499 examples [00:18, 973.54 examples/s]17597 examples [00:18, 970.84 examples/s]17695 examples [00:18, 972.08 examples/s]17794 examples [00:19, 975.93 examples/s]17895 examples [00:19, 985.83 examples/s]17995 examples [00:19, 988.12 examples/s]18095 examples [00:19, 988.76 examples/s]18194 examples [00:19, 985.12 examples/s]18295 examples [00:19, 990.01 examples/s]18396 examples [00:19, 993.80 examples/s]18499 examples [00:19, 1003.04 examples/s]18600 examples [00:19, 996.02 examples/s] 18700 examples [00:19, 982.84 examples/s]18800 examples [00:20, 986.68 examples/s]18899 examples [00:20, 969.13 examples/s]18998 examples [00:20, 975.11 examples/s]19096 examples [00:20, 974.62 examples/s]19194 examples [00:20, 975.74 examples/s]19292 examples [00:20, 969.52 examples/s]19391 examples [00:20, 973.12 examples/s]19489 examples [00:20, 955.29 examples/s]19585 examples [00:20, 956.14 examples/s]19685 examples [00:20, 967.11 examples/s]19784 examples [00:21, 971.38 examples/s]19882 examples [00:21, 960.90 examples/s]19979 examples [00:21, 963.20 examples/s]20076 examples [00:21, 924.29 examples/s]20169 examples [00:21, 922.13 examples/s]20262 examples [00:21, 914.42 examples/s]20358 examples [00:21, 926.17 examples/s]20451 examples [00:21, 924.82 examples/s]20544 examples [00:21, 921.41 examples/s]20646 examples [00:21, 946.53 examples/s]20741 examples [00:22, 919.26 examples/s]20842 examples [00:22, 942.36 examples/s]20940 examples [00:22, 949.17 examples/s]21036 examples [00:22, 940.58 examples/s]21132 examples [00:22, 945.57 examples/s]21227 examples [00:22, 939.68 examples/s]21323 examples [00:22, 944.88 examples/s]21418 examples [00:22, 934.11 examples/s]21513 examples [00:22, 937.23 examples/s]21607 examples [00:23, 937.68 examples/s]21701 examples [00:23, 920.27 examples/s]21794 examples [00:23, 917.86 examples/s]21886 examples [00:23, 899.22 examples/s]21978 examples [00:23, 903.92 examples/s]22072 examples [00:23, 913.14 examples/s]22164 examples [00:23, 912.08 examples/s]22259 examples [00:23, 920.71 examples/s]22352 examples [00:23, 913.85 examples/s]22444 examples [00:23, 883.80 examples/s]22537 examples [00:24, 895.70 examples/s]22632 examples [00:24, 909.91 examples/s]22727 examples [00:24, 920.56 examples/s]22821 examples [00:24, 923.69 examples/s]22915 examples [00:24, 928.11 examples/s]23012 examples [00:24, 939.70 examples/s]23109 examples [00:24, 946.62 examples/s]23206 examples [00:24, 951.41 examples/s]23302 examples [00:24, 933.19 examples/s]23396 examples [00:24, 888.60 examples/s]23487 examples [00:25, 892.47 examples/s]23581 examples [00:25, 905.25 examples/s]23681 examples [00:25, 931.36 examples/s]23775 examples [00:25, 930.46 examples/s]23869 examples [00:25, 925.14 examples/s]23962 examples [00:25, 906.80 examples/s]24055 examples [00:25, 910.97 examples/s]24150 examples [00:25, 920.54 examples/s]24243 examples [00:25, 911.43 examples/s]24336 examples [00:25, 915.23 examples/s]24428 examples [00:26, 902.18 examples/s]24521 examples [00:26, 909.63 examples/s]24613 examples [00:26, 911.14 examples/s]24705 examples [00:26, 899.46 examples/s]24796 examples [00:26, 891.89 examples/s]24890 examples [00:26, 905.34 examples/s]24981 examples [00:26, 890.44 examples/s]25075 examples [00:26, 904.03 examples/s]25166 examples [00:26, 897.69 examples/s]25259 examples [00:27, 904.69 examples/s]25351 examples [00:27, 906.59 examples/s]25446 examples [00:27, 917.45 examples/s]25540 examples [00:27, 922.54 examples/s]25634 examples [00:27, 927.04 examples/s]25727 examples [00:27, 927.55 examples/s]25820 examples [00:27, 916.91 examples/s]25915 examples [00:27, 924.09 examples/s]26011 examples [00:27, 933.74 examples/s]26105 examples [00:27, 915.15 examples/s]26200 examples [00:28, 923.87 examples/s]26293 examples [00:28, 919.57 examples/s]26387 examples [00:28, 924.27 examples/s]26481 examples [00:28, 928.45 examples/s]26574 examples [00:28, 920.63 examples/s]26667 examples [00:28, 902.78 examples/s]26758 examples [00:28, 903.77 examples/s]26849 examples [00:28, 887.74 examples/s]26940 examples [00:28, 892.13 examples/s]27030 examples [00:28, 866.87 examples/s]27120 examples [00:29, 874.10 examples/s]27208 examples [00:29, 855.56 examples/s]27301 examples [00:29, 874.64 examples/s]27394 examples [00:29, 888.32 examples/s]27486 examples [00:29, 895.59 examples/s]27576 examples [00:29, 893.69 examples/s]27668 examples [00:29, 900.80 examples/s]27760 examples [00:29, 903.95 examples/s]27851 examples [00:29, 898.16 examples/s]27942 examples [00:29, 900.62 examples/s]28033 examples [00:30, 886.92 examples/s]28123 examples [00:30, 890.71 examples/s]28215 examples [00:30, 899.07 examples/s]28306 examples [00:30, 901.23 examples/s]28397 examples [00:30, 880.44 examples/s]28489 examples [00:30, 889.84 examples/s]28582 examples [00:30, 900.25 examples/s]28674 examples [00:30, 905.84 examples/s]28765 examples [00:30, 881.25 examples/s]28857 examples [00:31, 890.31 examples/s]28952 examples [00:31, 905.14 examples/s]29043 examples [00:31, 901.64 examples/s]29136 examples [00:31, 908.78 examples/s]29231 examples [00:31, 920.75 examples/s]29325 examples [00:31, 923.81 examples/s]29419 examples [00:31, 928.55 examples/s]29515 examples [00:31, 935.94 examples/s]29612 examples [00:31, 943.09 examples/s]29707 examples [00:31, 934.77 examples/s]29801 examples [00:32, 925.89 examples/s]29894 examples [00:32, 921.76 examples/s]29989 examples [00:32, 927.36 examples/s]30082 examples [00:32, 884.24 examples/s]30179 examples [00:32, 908.21 examples/s]30272 examples [00:32, 913.89 examples/s]30364 examples [00:32, 908.17 examples/s]30457 examples [00:32, 913.84 examples/s]30549 examples [00:32, 913.86 examples/s]30641 examples [00:32, 910.06 examples/s]30733 examples [00:33, 908.43 examples/s]30824 examples [00:33, 900.74 examples/s]30917 examples [00:33, 899.85 examples/s]31008 examples [00:33, 896.54 examples/s]31101 examples [00:33, 903.53 examples/s]31192 examples [00:33, 898.70 examples/s]31284 examples [00:33, 902.90 examples/s]31377 examples [00:33, 910.28 examples/s]31471 examples [00:33, 916.56 examples/s]31563 examples [00:33, 891.44 examples/s]31655 examples [00:34, 897.37 examples/s]31747 examples [00:34, 901.77 examples/s]31840 examples [00:34, 907.78 examples/s]31934 examples [00:34, 915.88 examples/s]32026 examples [00:34, 911.18 examples/s]32118 examples [00:34, 901.57 examples/s]32211 examples [00:34, 909.59 examples/s]32303 examples [00:34, 904.73 examples/s]32394 examples [00:34, 895.60 examples/s]32484 examples [00:34, 880.77 examples/s]32573 examples [00:35, 882.84 examples/s]32662 examples [00:35, 880.40 examples/s]32751 examples [00:35, 867.42 examples/s]32840 examples [00:35, 873.35 examples/s]32932 examples [00:35, 884.20 examples/s]33021 examples [00:35, 880.51 examples/s]33115 examples [00:35, 895.51 examples/s]33208 examples [00:35, 904.16 examples/s]33300 examples [00:35, 907.95 examples/s]33395 examples [00:36, 919.82 examples/s]33489 examples [00:36, 923.58 examples/s]33585 examples [00:36, 931.73 examples/s]33679 examples [00:36, 929.60 examples/s]33777 examples [00:36, 943.02 examples/s]33875 examples [00:36, 953.04 examples/s]33974 examples [00:36, 962.06 examples/s]34074 examples [00:36, 972.85 examples/s]34174 examples [00:36, 980.25 examples/s]34280 examples [00:36, 1001.42 examples/s]34381 examples [00:37, 985.70 examples/s] 34480 examples [00:37, 973.90 examples/s]34579 examples [00:37, 977.29 examples/s]34679 examples [00:37, 982.73 examples/s]34778 examples [00:37, 962.57 examples/s]34877 examples [00:37, 970.05 examples/s]34977 examples [00:37, 978.23 examples/s]35075 examples [00:37, 962.47 examples/s]35173 examples [00:37, 965.80 examples/s]35273 examples [00:37, 973.06 examples/s]35371 examples [00:38, 973.87 examples/s]35469 examples [00:38, 959.24 examples/s]35566 examples [00:38, 936.55 examples/s]35662 examples [00:38, 940.88 examples/s]35757 examples [00:38, 938.90 examples/s]35851 examples [00:38, 934.56 examples/s]35945 examples [00:38, 927.92 examples/s]36042 examples [00:38, 938.89 examples/s]36136 examples [00:38, 933.78 examples/s]36230 examples [00:38, 934.83 examples/s]36326 examples [00:39, 941.10 examples/s]36421 examples [00:39, 936.84 examples/s]36515 examples [00:39, 936.97 examples/s]36609 examples [00:39, 937.43 examples/s]36704 examples [00:39, 940.34 examples/s]36801 examples [00:39, 946.64 examples/s]36896 examples [00:39, 943.05 examples/s]36993 examples [00:39, 949.92 examples/s]37089 examples [00:39, 931.37 examples/s]37184 examples [00:39, 934.04 examples/s]37279 examples [00:40, 935.42 examples/s]37373 examples [00:40, 926.87 examples/s]37470 examples [00:40, 938.13 examples/s]37567 examples [00:40, 946.63 examples/s]37662 examples [00:40, 936.64 examples/s]37756 examples [00:40, 926.66 examples/s]37850 examples [00:40, 929.01 examples/s]37943 examples [00:40, 903.52 examples/s]38036 examples [00:40, 910.84 examples/s]38132 examples [00:41, 924.57 examples/s]38226 examples [00:41, 927.64 examples/s]38322 examples [00:41, 936.50 examples/s]38416 examples [00:41, 927.87 examples/s]38509 examples [00:41, 925.23 examples/s]38602 examples [00:41, 921.77 examples/s]38695 examples [00:41, 913.04 examples/s]38788 examples [00:41, 917.37 examples/s]38884 examples [00:41, 927.21 examples/s]38979 examples [00:41, 931.66 examples/s]39076 examples [00:42, 941.73 examples/s]39171 examples [00:42, 936.43 examples/s]39267 examples [00:42, 941.86 examples/s]39364 examples [00:42, 949.67 examples/s]39460 examples [00:42, 945.96 examples/s]39555 examples [00:42, 938.60 examples/s]39649 examples [00:42, 928.14 examples/s]39744 examples [00:42, 932.80 examples/s]39838 examples [00:42, 906.59 examples/s]39933 examples [00:42, 915.81 examples/s]40025 examples [00:43, 877.82 examples/s]40117 examples [00:43, 889.02 examples/s]40211 examples [00:43, 901.33 examples/s]40303 examples [00:43, 906.77 examples/s]40396 examples [00:43, 912.42 examples/s]40490 examples [00:43, 920.00 examples/s]40583 examples [00:43, 922.67 examples/s]40676 examples [00:43, 920.54 examples/s]40769 examples [00:43, 901.61 examples/s]40860 examples [00:43, 903.15 examples/s]40954 examples [00:44, 911.79 examples/s]41046 examples [00:44, 909.80 examples/s]41141 examples [00:44, 921.46 examples/s]41234 examples [00:44, 885.48 examples/s]41328 examples [00:44, 899.64 examples/s]41420 examples [00:44, 905.62 examples/s]41512 examples [00:44, 907.29 examples/s]41606 examples [00:44, 915.69 examples/s]41701 examples [00:44, 923.21 examples/s]41796 examples [00:44, 928.56 examples/s]41892 examples [00:45, 936.04 examples/s]41986 examples [00:45, 932.06 examples/s]42081 examples [00:45, 935.65 examples/s]42175 examples [00:45, 936.32 examples/s]42269 examples [00:45, 937.29 examples/s]42363 examples [00:45, 933.55 examples/s]42457 examples [00:45, 924.65 examples/s]42550 examples [00:45, 920.30 examples/s]42643 examples [00:45, 920.46 examples/s]42737 examples [00:46, 923.59 examples/s]42830 examples [00:46, 923.74 examples/s]42923 examples [00:46, 918.78 examples/s]43019 examples [00:46, 928.23 examples/s]43112 examples [00:46, 919.81 examples/s]43210 examples [00:46, 936.85 examples/s]43306 examples [00:46, 942.30 examples/s]43401 examples [00:46, 934.05 examples/s]43498 examples [00:46, 943.73 examples/s]43593 examples [00:46, 926.54 examples/s]43688 examples [00:47, 931.35 examples/s]43782 examples [00:47, 932.76 examples/s]43876 examples [00:47, 930.28 examples/s]43970 examples [00:47, 926.13 examples/s]44064 examples [00:47, 928.15 examples/s]44157 examples [00:47, 921.41 examples/s]44250 examples [00:47, 922.01 examples/s]44343 examples [00:47, 908.83 examples/s]44435 examples [00:47, 911.54 examples/s]44527 examples [00:47, 909.57 examples/s]44618 examples [00:48, 884.85 examples/s]44710 examples [00:48, 892.72 examples/s]44803 examples [00:48, 901.09 examples/s]44898 examples [00:48, 911.95 examples/s]44992 examples [00:48, 917.87 examples/s]45088 examples [00:48, 927.67 examples/s]45183 examples [00:48, 932.47 examples/s]45277 examples [00:48, 928.15 examples/s]45370 examples [00:48, 925.62 examples/s]45464 examples [00:48, 929.66 examples/s]45558 examples [00:49, 932.68 examples/s]45652 examples [00:49, 932.80 examples/s]45746 examples [00:49, 932.63 examples/s]45841 examples [00:49, 935.38 examples/s]45935 examples [00:49, 933.63 examples/s]46029 examples [00:49, 930.61 examples/s]46123 examples [00:49, 932.82 examples/s]46217 examples [00:49, 929.14 examples/s]46312 examples [00:49, 933.33 examples/s]46406 examples [00:49, 933.26 examples/s]46500 examples [00:50, 933.79 examples/s]46594 examples [00:50, 930.53 examples/s]46688 examples [00:50, 918.36 examples/s]46782 examples [00:50, 922.32 examples/s]46875 examples [00:50, 918.78 examples/s]46970 examples [00:50, 925.57 examples/s]47063 examples [00:50, 922.66 examples/s]47156 examples [00:50, 919.94 examples/s]47250 examples [00:50, 918.26 examples/s]47342 examples [00:50, 902.43 examples/s]47436 examples [00:51, 912.95 examples/s]47530 examples [00:51, 920.63 examples/s]47623 examples [00:51, 917.77 examples/s]47717 examples [00:51, 922.86 examples/s]47810 examples [00:51, 923.97 examples/s]47906 examples [00:51, 933.87 examples/s]48004 examples [00:51, 945.68 examples/s]48101 examples [00:51, 950.60 examples/s]48197 examples [00:51, 936.75 examples/s]48296 examples [00:52, 950.51 examples/s]48392 examples [00:52, 941.28 examples/s]48487 examples [00:52, 926.12 examples/s]48580 examples [00:52, 925.26 examples/s]48673 examples [00:52, 922.91 examples/s]48766 examples [00:52, 916.72 examples/s]48860 examples [00:52, 921.73 examples/s]48955 examples [00:52, 928.70 examples/s]49054 examples [00:52, 945.74 examples/s]49152 examples [00:52, 953.89 examples/s]49253 examples [00:53, 969.22 examples/s]49354 examples [00:53, 980.66 examples/s]49453 examples [00:53, 978.13 examples/s]49551 examples [00:53, 940.73 examples/s]49651 examples [00:53, 955.73 examples/s]49747 examples [00:53, 955.40 examples/s]49847 examples [00:53, 967.12 examples/s]49946 examples [00:53, 971.38 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7167/50000 [00:00<00:00, 71663.92 examples/s] 38%|███▊      | 18916/50000 [00:00<00:00, 81160.25 examples/s] 62%|██████▏   | 30788/50000 [00:00<00:00, 89670.92 examples/s] 86%|████████▋ | 43219/50000 [00:00<00:00, 97850.14 examples/s]                                                               0 examples [00:00, ? examples/s]86 examples [00:00, 851.96 examples/s]188 examples [00:00, 894.67 examples/s]291 examples [00:00, 929.60 examples/s]392 examples [00:00, 950.44 examples/s]491 examples [00:00, 958.37 examples/s]590 examples [00:00, 966.69 examples/s]681 examples [00:00, 946.46 examples/s]780 examples [00:00, 958.20 examples/s]880 examples [00:00, 968.26 examples/s]981 examples [00:01, 980.23 examples/s]1080 examples [00:01, 980.74 examples/s]1178 examples [00:01, 979.98 examples/s]1282 examples [00:01, 994.82 examples/s]1384 examples [00:01, 1001.95 examples/s]1484 examples [00:01, 996.15 examples/s] 1584 examples [00:01, 991.48 examples/s]1687 examples [00:01, 1000.76 examples/s]1789 examples [00:01, 1004.87 examples/s]1890 examples [00:01, 1004.98 examples/s]1992 examples [00:02, 1008.76 examples/s]2093 examples [00:02, 995.51 examples/s] 2194 examples [00:02, 999.71 examples/s]2296 examples [00:02, 1005.47 examples/s]2397 examples [00:02, 995.72 examples/s] 2497 examples [00:02, 988.65 examples/s]2596 examples [00:02, 978.70 examples/s]2699 examples [00:02, 991.35 examples/s]2800 examples [00:02, 994.21 examples/s]2900 examples [00:02, 993.49 examples/s]3001 examples [00:03, 996.03 examples/s]3101 examples [00:03, 984.06 examples/s]3202 examples [00:03, 990.81 examples/s]3302 examples [00:03, 975.31 examples/s]3403 examples [00:03, 983.33 examples/s]3504 examples [00:03, 990.29 examples/s]3604 examples [00:03, 980.03 examples/s]3703 examples [00:03, 978.67 examples/s]3804 examples [00:03, 984.97 examples/s]3904 examples [00:03, 986.62 examples/s]4005 examples [00:04, 992.96 examples/s]4105 examples [00:04, 979.72 examples/s]4204 examples [00:04, 974.31 examples/s]4305 examples [00:04, 981.78 examples/s]4406 examples [00:04, 988.84 examples/s]4507 examples [00:04, 992.83 examples/s]4607 examples [00:04, 981.66 examples/s]4706 examples [00:04, 983.03 examples/s]4805 examples [00:04, 799.02 examples/s]4903 examples [00:05, 844.61 examples/s]4999 examples [00:05, 875.28 examples/s]5096 examples [00:05, 900.58 examples/s]5195 examples [00:05, 924.73 examples/s]5297 examples [00:05, 949.81 examples/s]5394 examples [00:05, 951.79 examples/s]5491 examples [00:05, 930.06 examples/s]5586 examples [00:05, 933.76 examples/s]5680 examples [00:05, 916.70 examples/s]5773 examples [00:05, 901.94 examples/s]5865 examples [00:06, 904.99 examples/s]5957 examples [00:06, 908.51 examples/s]6049 examples [00:06, 905.50 examples/s]6142 examples [00:06, 912.12 examples/s]6238 examples [00:06, 924.19 examples/s]6334 examples [00:06, 933.86 examples/s]6428 examples [00:06, 919.18 examples/s]6521 examples [00:06, 910.85 examples/s]6613 examples [00:06, 861.16 examples/s]6703 examples [00:07, 872.25 examples/s]6797 examples [00:07, 889.79 examples/s]6891 examples [00:07, 903.80 examples/s]6988 examples [00:07, 919.98 examples/s]7081 examples [00:07, 922.41 examples/s]7174 examples [00:07, 922.92 examples/s]7267 examples [00:07, 918.55 examples/s]7360 examples [00:07, 920.81 examples/s]7453 examples [00:07, 917.12 examples/s]7545 examples [00:07, 913.41 examples/s]7637 examples [00:08, 911.27 examples/s]7732 examples [00:08, 919.99 examples/s]7825 examples [00:08, 915.54 examples/s]7917 examples [00:08, 904.63 examples/s]8012 examples [00:08, 916.30 examples/s]8106 examples [00:08, 922.17 examples/s]8200 examples [00:08, 924.52 examples/s]8293 examples [00:08, 919.93 examples/s]8386 examples [00:08, 922.67 examples/s]8479 examples [00:08, 917.60 examples/s]8573 examples [00:09, 923.48 examples/s]8666 examples [00:09, 896.25 examples/s]8756 examples [00:09, 894.40 examples/s]8848 examples [00:09, 899.88 examples/s]8939 examples [00:09, 878.72 examples/s]9034 examples [00:09, 896.60 examples/s]9124 examples [00:09, 881.81 examples/s]9216 examples [00:09, 891.81 examples/s]9311 examples [00:09, 907.10 examples/s]9403 examples [00:09, 910.62 examples/s]9497 examples [00:10, 916.66 examples/s]9590 examples [00:10, 919.03 examples/s]9685 examples [00:10, 922.79 examples/s]9780 examples [00:10, 929.13 examples/s]9877 examples [00:10, 939.13 examples/s]9973 examples [00:10, 944.08 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQO0111/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQO0111/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f843568e510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f843568e510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f843568e510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f84336f4b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83d06725f8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f843568e510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f843568e510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f843568e510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83d0672fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f841c8cd198>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.10 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.71 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.71 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.47 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.47 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.26 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.26 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.62 file/s]2020-09-04 06:08:15.615705: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-04 06:08:15.619629: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-09-04 06:08:15.619802: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c358bb76f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-04 06:08:15.619822: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'mnist2', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy', 'train', 'test'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:22, 440620.56it/s] 98%|█████████▊| 9732096/9912422 [00:00<00:00, 628229.15it/s]9920512it [00:00, 46561089.06it/s]                           
0it [00:00, ?it/s]32768it [00:00, 609674.64it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 448180.24it/s]1654784it [00:00, 11655936.32it/s]                         
0it [00:00, ?it/s]8192it [00:00, 196806.95it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
