
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f80589aaae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f80589aaae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f80c361a510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f80c361a510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f80e2847ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f80e2847ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f80709c2a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f80709c2a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f80709c2a60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 162028.38it/s] 51%|█████     | 5062656/9912422 [00:00<00:20, 231140.39it/s]9920512it [00:00, 34868747.21it/s]                           
0it [00:00, ?it/s]32768it [00:00, 709014.69it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1023368.01it/s]1654784it [00:00, 11582082.46it/s]                           
0it [00:00, ?it/s]8192it [00:00, 225005.82it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f807094d4a8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8058875978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f80709c26a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f80709c26a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f80709c26a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:55,  1.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:55,  1.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:54,  1.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:53,  1.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:53,  1.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:52,  1.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:51,  1.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:50,  1.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:50,  1.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:17,  1.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:17,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:16,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:16,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:15,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:15,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:14,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:14,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:13,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:13,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<01:12,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:50,  2.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:50,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:50,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:50,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:49,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:49,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:49,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:48,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:48,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:48,  2.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:47,  2.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:33,  3.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:33,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:33,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:33,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:32,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:32,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:32,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:32,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:31,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:31,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:31,  3.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:22,  5.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:22,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:21,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:21,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:21,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:21,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:21,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:21,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:20,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:20,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:20,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:20,  5.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:14,  7.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:14,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:14,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:14,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:14,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:13,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:13,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:13,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:13,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:13,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:13,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:13,  7.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:09, 10.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:09, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:09, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:09, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:09, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:09, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:08, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:08, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:08, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:08, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:08, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:06, 14.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:06, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:06, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:06, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:06, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:05, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:05, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:05, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:05, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:05, 14.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 19.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:04, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 19.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 25.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 25.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 32.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 32.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 40.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:01, 40.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 48.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 48.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 56.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 56.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 63.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 69.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.43 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.48s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.43 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.48s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.48s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.17 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.48s/ url]
0 examples [00:00, ? examples/s]2020-07-30 06:08:43.461312: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-30 06:08:43.475874: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-30 06:08:43.476202: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b031f48a70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-30 06:08:43.476240: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
21 examples [00:00, 208.21 examples/s]127 examples [00:00, 274.21 examples/s]238 examples [00:00, 353.96 examples/s]349 examples [00:00, 444.46 examples/s]459 examples [00:00, 541.12 examples/s]570 examples [00:00, 638.58 examples/s]682 examples [00:00, 732.68 examples/s]794 examples [00:00, 816.83 examples/s]903 examples [00:00, 882.17 examples/s]1014 examples [00:01, 939.15 examples/s]1125 examples [00:01, 983.72 examples/s]1234 examples [00:01, 1012.28 examples/s]1342 examples [00:01, 1031.56 examples/s]1454 examples [00:01, 1054.32 examples/s]1563 examples [00:01, 1049.53 examples/s]1671 examples [00:01, 1019.01 examples/s]1779 examples [00:01, 1033.74 examples/s]1884 examples [00:01, 1029.12 examples/s]1996 examples [00:01, 1052.79 examples/s]2106 examples [00:02, 1065.16 examples/s]2219 examples [00:02, 1082.61 examples/s]2331 examples [00:02, 1092.74 examples/s]2441 examples [00:02, 1093.99 examples/s]2552 examples [00:02, 1096.34 examples/s]2664 examples [00:02, 1101.39 examples/s]2776 examples [00:02, 1105.27 examples/s]2887 examples [00:02, 1095.66 examples/s]2998 examples [00:02, 1098.96 examples/s]3108 examples [00:02, 1098.54 examples/s]3218 examples [00:03, 1068.44 examples/s]3330 examples [00:03, 1081.52 examples/s]3443 examples [00:03, 1093.70 examples/s]3553 examples [00:03, 1086.90 examples/s]3666 examples [00:03, 1098.29 examples/s]3777 examples [00:03, 1100.68 examples/s]3888 examples [00:03, 1103.39 examples/s]3999 examples [00:03, 1090.82 examples/s]4109 examples [00:03, 1076.48 examples/s]4217 examples [00:03, 1068.98 examples/s]4326 examples [00:04, 1072.71 examples/s]4434 examples [00:04, 1044.74 examples/s]4544 examples [00:04, 1059.77 examples/s]4653 examples [00:04, 1066.32 examples/s]4767 examples [00:04, 1085.29 examples/s]4878 examples [00:04, 1090.08 examples/s]4991 examples [00:04, 1100.14 examples/s]5102 examples [00:04, 1100.54 examples/s]5214 examples [00:04, 1105.96 examples/s]5325 examples [00:04, 1098.65 examples/s]5439 examples [00:05, 1108.91 examples/s]5552 examples [00:05, 1113.75 examples/s]5664 examples [00:05, 1098.92 examples/s]5777 examples [00:05, 1107.40 examples/s]5891 examples [00:05, 1114.82 examples/s]6003 examples [00:05, 1110.61 examples/s]6115 examples [00:05, 1112.84 examples/s]6227 examples [00:05, 1108.32 examples/s]6342 examples [00:05, 1118.17 examples/s]6454 examples [00:05, 1109.23 examples/s]6568 examples [00:06, 1116.58 examples/s]6680 examples [00:06, 1117.22 examples/s]6792 examples [00:06, 1109.04 examples/s]6905 examples [00:06, 1113.53 examples/s]7018 examples [00:06, 1117.86 examples/s]7132 examples [00:06, 1122.52 examples/s]7245 examples [00:06, 1111.59 examples/s]7357 examples [00:06, 1101.83 examples/s]7468 examples [00:06, 1102.02 examples/s]7579 examples [00:07, 1098.13 examples/s]7689 examples [00:07, 1091.02 examples/s]7799 examples [00:07, 1086.16 examples/s]7908 examples [00:07, 1081.10 examples/s]8017 examples [00:07, 1073.00 examples/s]8125 examples [00:07, 1074.50 examples/s]8233 examples [00:07, 1074.03 examples/s]8341 examples [00:07, 1071.10 examples/s]8449 examples [00:07, 1066.05 examples/s]8556 examples [00:07, 1062.63 examples/s]8666 examples [00:08, 1073.16 examples/s]8774 examples [00:08, 1055.69 examples/s]8880 examples [00:08, 1055.79 examples/s]8986 examples [00:08, 1049.86 examples/s]9094 examples [00:08, 1056.06 examples/s]9202 examples [00:08, 1060.45 examples/s]9309 examples [00:08, 1062.90 examples/s]9416 examples [00:08, 1043.03 examples/s]9521 examples [00:08, 1026.98 examples/s]9624 examples [00:08, 1022.85 examples/s]9733 examples [00:09, 1041.77 examples/s]9845 examples [00:09, 1062.19 examples/s]9959 examples [00:09, 1082.76 examples/s]10068 examples [00:09, 1026.36 examples/s]10177 examples [00:09, 1044.35 examples/s]10290 examples [00:09, 1066.45 examples/s]10403 examples [00:09, 1083.09 examples/s]10515 examples [00:09, 1092.51 examples/s]10625 examples [00:09, 1084.81 examples/s]10734 examples [00:09, 1085.52 examples/s]10844 examples [00:10, 1089.39 examples/s]10954 examples [00:10, 1091.90 examples/s]11064 examples [00:10, 1080.05 examples/s]11174 examples [00:10, 1083.30 examples/s]11285 examples [00:10, 1089.86 examples/s]11395 examples [00:10, 1089.45 examples/s]11506 examples [00:10, 1095.00 examples/s]11618 examples [00:10, 1101.36 examples/s]11729 examples [00:10, 1096.40 examples/s]11840 examples [00:10, 1099.62 examples/s]11950 examples [00:11, 1091.20 examples/s]12063 examples [00:11, 1099.54 examples/s]12173 examples [00:11, 1067.44 examples/s]12280 examples [00:11, 1066.81 examples/s]12393 examples [00:11, 1083.75 examples/s]12505 examples [00:11, 1092.59 examples/s]12617 examples [00:11, 1099.74 examples/s]12731 examples [00:11, 1110.78 examples/s]12843 examples [00:11, 1099.99 examples/s]12956 examples [00:12, 1108.64 examples/s]13068 examples [00:12, 1109.41 examples/s]13179 examples [00:12, 1106.86 examples/s]13290 examples [00:12, 1094.26 examples/s]13400 examples [00:12, 1094.78 examples/s]13510 examples [00:12, 1091.05 examples/s]13620 examples [00:12, 1072.21 examples/s]13728 examples [00:12, 1064.08 examples/s]13836 examples [00:12, 1068.73 examples/s]13944 examples [00:12, 1071.10 examples/s]14055 examples [00:13, 1080.66 examples/s]14166 examples [00:13, 1086.92 examples/s]14276 examples [00:13, 1090.37 examples/s]14386 examples [00:13, 1086.48 examples/s]14495 examples [00:13, 1078.63 examples/s]14606 examples [00:13, 1086.03 examples/s]14715 examples [00:13, 1073.89 examples/s]14823 examples [00:13, 1068.55 examples/s]14933 examples [00:13, 1076.60 examples/s]15042 examples [00:13, 1077.89 examples/s]15151 examples [00:14, 1078.75 examples/s]15261 examples [00:14, 1082.24 examples/s]15374 examples [00:14, 1094.02 examples/s]15487 examples [00:14, 1102.18 examples/s]15598 examples [00:14, 1098.27 examples/s]15710 examples [00:14, 1101.66 examples/s]15821 examples [00:14, 1102.31 examples/s]15934 examples [00:14, 1108.06 examples/s]16045 examples [00:14, 1107.63 examples/s]16156 examples [00:14, 1099.48 examples/s]16269 examples [00:15, 1106.21 examples/s]16380 examples [00:15, 1088.72 examples/s]16491 examples [00:15, 1092.64 examples/s]16601 examples [00:15, 1094.19 examples/s]16711 examples [00:15, 1093.41 examples/s]16824 examples [00:15, 1103.45 examples/s]16936 examples [00:15, 1107.28 examples/s]17049 examples [00:15, 1112.90 examples/s]17161 examples [00:15, 1114.67 examples/s]17273 examples [00:15, 1107.25 examples/s]17385 examples [00:16, 1110.69 examples/s]17499 examples [00:16, 1116.84 examples/s]17611 examples [00:16, 1103.09 examples/s]17722 examples [00:16, 1104.54 examples/s]17833 examples [00:16, 1086.27 examples/s]17942 examples [00:16, 1085.97 examples/s]18054 examples [00:16, 1093.27 examples/s]18164 examples [00:16, 1065.30 examples/s]18274 examples [00:16, 1072.58 examples/s]18382 examples [00:16, 1071.61 examples/s]18490 examples [00:17, 1068.13 examples/s]18597 examples [00:17, 1063.38 examples/s]18705 examples [00:17, 1068.05 examples/s]18815 examples [00:17, 1074.54 examples/s]18923 examples [00:17, 1064.78 examples/s]19030 examples [00:17, 1050.09 examples/s]19137 examples [00:17, 1055.39 examples/s]19247 examples [00:17, 1066.71 examples/s]19357 examples [00:17, 1074.78 examples/s]19465 examples [00:18, 1065.97 examples/s]19574 examples [00:18, 1071.50 examples/s]19683 examples [00:18, 1074.80 examples/s]19793 examples [00:18, 1080.78 examples/s]19902 examples [00:18, 1074.45 examples/s]20010 examples [00:18, 1014.23 examples/s]20120 examples [00:18, 1037.08 examples/s]20230 examples [00:18, 1054.37 examples/s]20340 examples [00:18, 1065.49 examples/s]20448 examples [00:18, 1066.82 examples/s]20555 examples [00:19, 1062.94 examples/s]20664 examples [00:19, 1070.23 examples/s]20775 examples [00:19, 1079.54 examples/s]20885 examples [00:19, 1084.03 examples/s]20995 examples [00:19, 1085.96 examples/s]21104 examples [00:19, 1081.94 examples/s]21213 examples [00:19, 1071.75 examples/s]21323 examples [00:19, 1078.40 examples/s]21432 examples [00:19, 1080.51 examples/s]21543 examples [00:19, 1086.40 examples/s]21652 examples [00:20, 1079.97 examples/s]21761 examples [00:20, 1074.72 examples/s]21869 examples [00:20, 1066.73 examples/s]21976 examples [00:20, 1065.81 examples/s]22087 examples [00:20, 1078.62 examples/s]22196 examples [00:20, 1079.37 examples/s]22307 examples [00:20, 1087.93 examples/s]22418 examples [00:20, 1093.49 examples/s]22531 examples [00:20, 1102.52 examples/s]22644 examples [00:20, 1108.69 examples/s]22755 examples [00:21, 1097.45 examples/s]22865 examples [00:21, 1028.44 examples/s]22969 examples [00:21, 993.15 examples/s] 23070 examples [00:21, 990.99 examples/s]23170 examples [00:21, 927.50 examples/s]23265 examples [00:21, 912.87 examples/s]23362 examples [00:21, 928.13 examples/s]23456 examples [00:21, 908.23 examples/s]23548 examples [00:21, 911.54 examples/s]23640 examples [00:22, 897.22 examples/s]23731 examples [00:22, 896.56 examples/s]23834 examples [00:22, 928.31 examples/s]23932 examples [00:22, 941.55 examples/s]24039 examples [00:22, 974.31 examples/s]24146 examples [00:22, 1000.55 examples/s]24248 examples [00:22, 1004.77 examples/s]24354 examples [00:22, 1019.61 examples/s]24460 examples [00:22, 1030.78 examples/s]24567 examples [00:22, 1042.05 examples/s]24675 examples [00:23, 1050.89 examples/s]24782 examples [00:23, 1054.49 examples/s]24889 examples [00:23, 1058.23 examples/s]24995 examples [00:23, 1058.33 examples/s]25101 examples [00:23, 1052.44 examples/s]25209 examples [00:23, 1059.86 examples/s]25316 examples [00:23, 1041.18 examples/s]25424 examples [00:23, 1049.80 examples/s]25536 examples [00:23, 1067.34 examples/s]25646 examples [00:23, 1076.57 examples/s]25754 examples [00:24, 1075.85 examples/s]25863 examples [00:24, 1078.29 examples/s]25973 examples [00:24, 1082.18 examples/s]26082 examples [00:24, 1082.84 examples/s]26191 examples [00:24, 1082.52 examples/s]26300 examples [00:24, 1074.29 examples/s]26408 examples [00:24, 1069.10 examples/s]26517 examples [00:24, 1074.41 examples/s]26625 examples [00:24, 1069.57 examples/s]26736 examples [00:24, 1079.10 examples/s]26844 examples [00:25, 1074.18 examples/s]26953 examples [00:25, 1078.05 examples/s]27062 examples [00:25, 1078.65 examples/s]27170 examples [00:25, 1046.21 examples/s]27281 examples [00:25, 1061.63 examples/s]27389 examples [00:25, 1066.37 examples/s]27499 examples [00:25, 1075.55 examples/s]27611 examples [00:25, 1088.45 examples/s]27721 examples [00:25, 1091.18 examples/s]27831 examples [00:25, 1090.61 examples/s]27941 examples [00:26, 1076.66 examples/s]28049 examples [00:26, 1046.96 examples/s]28158 examples [00:26, 1058.64 examples/s]28266 examples [00:26, 1063.34 examples/s]28373 examples [00:26, 1053.86 examples/s]28483 examples [00:26, 1066.31 examples/s]28592 examples [00:26, 1070.62 examples/s]28705 examples [00:26, 1085.44 examples/s]28816 examples [00:26, 1090.71 examples/s]28926 examples [00:27, 1085.74 examples/s]29038 examples [00:27, 1093.33 examples/s]29148 examples [00:27, 1093.97 examples/s]29261 examples [00:27, 1101.80 examples/s]29372 examples [00:27, 1101.76 examples/s]29483 examples [00:27, 1103.13 examples/s]29595 examples [00:27, 1105.48 examples/s]29706 examples [00:27, 1093.86 examples/s]29816 examples [00:27, 1068.77 examples/s]29929 examples [00:27, 1084.82 examples/s]30038 examples [00:28, 1036.24 examples/s]30145 examples [00:28, 1043.93 examples/s]30253 examples [00:28, 1052.33 examples/s]30363 examples [00:28, 1065.27 examples/s]30470 examples [00:28, 1059.01 examples/s]30578 examples [00:28, 1063.39 examples/s]30685 examples [00:28, 1063.96 examples/s]30792 examples [00:28, 1058.31 examples/s]30898 examples [00:28, 1040.76 examples/s]31003 examples [00:28, 1042.46 examples/s]31108 examples [00:29, 1044.31 examples/s]31218 examples [00:29, 1060.12 examples/s]31327 examples [00:29, 1067.28 examples/s]31437 examples [00:29, 1075.56 examples/s]31548 examples [00:29, 1085.61 examples/s]31657 examples [00:29, 1080.53 examples/s]31768 examples [00:29, 1086.46 examples/s]31877 examples [00:29, 1083.95 examples/s]31986 examples [00:29, 1081.92 examples/s]32095 examples [00:29, 1079.49 examples/s]32206 examples [00:30, 1086.57 examples/s]32315 examples [00:30, 1083.04 examples/s]32424 examples [00:30, 1072.09 examples/s]32532 examples [00:30, 1055.17 examples/s]32638 examples [00:30, 1054.33 examples/s]32744 examples [00:30, 1034.97 examples/s]32852 examples [00:30, 1045.27 examples/s]32958 examples [00:30, 1048.38 examples/s]33067 examples [00:30, 1058.69 examples/s]33175 examples [00:30, 1064.46 examples/s]33283 examples [00:31, 1067.32 examples/s]33391 examples [00:31, 1069.20 examples/s]33498 examples [00:31, 1064.78 examples/s]33606 examples [00:31, 1067.05 examples/s]33717 examples [00:31, 1077.15 examples/s]33827 examples [00:31, 1080.92 examples/s]33938 examples [00:31, 1087.17 examples/s]34047 examples [00:31, 1084.01 examples/s]34156 examples [00:31, 1080.42 examples/s]34265 examples [00:31, 1077.56 examples/s]34373 examples [00:32, 1065.74 examples/s]34480 examples [00:32, 1064.18 examples/s]34587 examples [00:32, 1061.75 examples/s]34694 examples [00:32, 1047.86 examples/s]34801 examples [00:32, 1054.14 examples/s]34907 examples [00:32, 1055.72 examples/s]35013 examples [00:32, 1026.06 examples/s]35119 examples [00:32, 1033.97 examples/s]35226 examples [00:32, 1043.61 examples/s]35334 examples [00:33, 1053.89 examples/s]35440 examples [00:33, 1054.69 examples/s]35550 examples [00:33, 1067.04 examples/s]35657 examples [00:33, 1064.43 examples/s]35764 examples [00:33, 1061.69 examples/s]35871 examples [00:33, 1036.41 examples/s]35982 examples [00:33, 1056.73 examples/s]36094 examples [00:33, 1073.16 examples/s]36202 examples [00:33, 1068.73 examples/s]36313 examples [00:33, 1078.69 examples/s]36421 examples [00:34, 1069.39 examples/s]36529 examples [00:34, 1062.38 examples/s]36637 examples [00:34, 1066.76 examples/s]36744 examples [00:34, 1052.16 examples/s]36850 examples [00:34, 1035.99 examples/s]36955 examples [00:34, 1038.73 examples/s]37064 examples [00:34, 1052.92 examples/s]37173 examples [00:34, 1062.78 examples/s]37282 examples [00:34, 1069.16 examples/s]37392 examples [00:34, 1077.03 examples/s]37504 examples [00:35, 1087.48 examples/s]37615 examples [00:35, 1091.52 examples/s]37725 examples [00:35, 1082.77 examples/s]37834 examples [00:35, 1084.00 examples/s]37945 examples [00:35, 1090.99 examples/s]38055 examples [00:35, 1091.93 examples/s]38165 examples [00:35, 1093.88 examples/s]38275 examples [00:35, 1095.66 examples/s]38385 examples [00:35, 1093.97 examples/s]38496 examples [00:35, 1097.65 examples/s]38606 examples [00:36, 1094.21 examples/s]38716 examples [00:36, 1092.03 examples/s]38828 examples [00:36, 1097.93 examples/s]38938 examples [00:36, 1097.47 examples/s]39048 examples [00:36, 1094.50 examples/s]39160 examples [00:36, 1099.58 examples/s]39270 examples [00:36, 1098.08 examples/s]39380 examples [00:36, 1088.98 examples/s]39489 examples [00:36, 1083.47 examples/s]39599 examples [00:36, 1088.34 examples/s]39708 examples [00:37, 1082.80 examples/s]39817 examples [00:37, 1081.51 examples/s]39926 examples [00:37, 1076.74 examples/s]40034 examples [00:37, 1017.93 examples/s]40142 examples [00:37, 1034.23 examples/s]40251 examples [00:37, 1047.83 examples/s]40357 examples [00:37, 1051.31 examples/s]40465 examples [00:37, 1059.01 examples/s]40572 examples [00:37, 1058.88 examples/s]40681 examples [00:38, 1067.84 examples/s]40789 examples [00:38, 1068.50 examples/s]40896 examples [00:38, 1058.27 examples/s]41002 examples [00:38, 1037.42 examples/s]41110 examples [00:38, 1048.21 examples/s]41218 examples [00:38, 1057.09 examples/s]41326 examples [00:38, 1063.14 examples/s]41433 examples [00:38, 1060.75 examples/s]41540 examples [00:38, 1049.32 examples/s]41647 examples [00:38, 1054.41 examples/s]41753 examples [00:39, 1046.27 examples/s]41865 examples [00:39, 1064.73 examples/s]41972 examples [00:39, 1044.30 examples/s]42081 examples [00:39, 1056.16 examples/s]42189 examples [00:39, 1061.01 examples/s]42298 examples [00:39, 1067.86 examples/s]42405 examples [00:39, 1064.87 examples/s]42514 examples [00:39, 1069.41 examples/s]42622 examples [00:39, 1070.54 examples/s]42730 examples [00:39, 1065.89 examples/s]42838 examples [00:40, 1068.06 examples/s]42952 examples [00:40, 1086.47 examples/s]43063 examples [00:40, 1090.96 examples/s]43173 examples [00:40, 1074.29 examples/s]43281 examples [00:40, 1070.35 examples/s]43389 examples [00:40, 1070.40 examples/s]43500 examples [00:40, 1081.49 examples/s]43612 examples [00:40, 1091.11 examples/s]43722 examples [00:40, 1090.53 examples/s]43832 examples [00:40, 1059.77 examples/s]43944 examples [00:41, 1075.99 examples/s]44052 examples [00:41, 1075.21 examples/s]44160 examples [00:41, 1074.88 examples/s]44272 examples [00:41, 1085.54 examples/s]44381 examples [00:41, 1078.83 examples/s]44494 examples [00:41, 1091.65 examples/s]44604 examples [00:41, 1093.33 examples/s]44716 examples [00:41, 1098.56 examples/s]44828 examples [00:41, 1104.28 examples/s]44939 examples [00:41, 1101.15 examples/s]45050 examples [00:42, 1097.68 examples/s]45161 examples [00:42, 1100.56 examples/s]45275 examples [00:42, 1110.62 examples/s]45387 examples [00:42, 1109.32 examples/s]45498 examples [00:42, 1099.47 examples/s]45611 examples [00:42, 1107.40 examples/s]45722 examples [00:42, 1105.81 examples/s]45833 examples [00:42, 1105.85 examples/s]45944 examples [00:42, 1095.61 examples/s]46054 examples [00:42, 1090.93 examples/s]46164 examples [00:43, 1090.13 examples/s]46277 examples [00:43, 1100.88 examples/s]46390 examples [00:43, 1107.20 examples/s]46501 examples [00:43, 1103.16 examples/s]46612 examples [00:43, 1097.05 examples/s]46723 examples [00:43, 1099.73 examples/s]46833 examples [00:43, 1089.12 examples/s]46944 examples [00:43, 1093.15 examples/s]47054 examples [00:43, 1091.53 examples/s]47164 examples [00:43, 1090.01 examples/s]47276 examples [00:44, 1097.28 examples/s]47388 examples [00:44, 1102.20 examples/s]47499 examples [00:44, 1086.91 examples/s]47608 examples [00:44, 1077.24 examples/s]47716 examples [00:44, 1070.79 examples/s]47826 examples [00:44, 1077.73 examples/s]47939 examples [00:44, 1091.12 examples/s]48050 examples [00:44, 1096.01 examples/s]48160 examples [00:44, 1090.32 examples/s]48270 examples [00:45, 1083.57 examples/s]48379 examples [00:45, 1077.50 examples/s]48490 examples [00:45, 1087.04 examples/s]48601 examples [00:45, 1092.60 examples/s]48711 examples [00:45, 1083.22 examples/s]48821 examples [00:45, 1086.37 examples/s]48931 examples [00:45, 1088.08 examples/s]49042 examples [00:45, 1091.96 examples/s]49155 examples [00:45, 1100.39 examples/s]49266 examples [00:45, 1097.78 examples/s]49376 examples [00:46, 1086.90 examples/s]49488 examples [00:46, 1094.97 examples/s]49598 examples [00:46, 1096.39 examples/s]49709 examples [00:46, 1099.40 examples/s]49819 examples [00:46, 1098.69 examples/s]49929 examples [00:46, 1090.88 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6592/50000 [00:00<00:00, 65918.49 examples/s] 39%|███▉      | 19512/50000 [00:00<00:00, 77272.53 examples/s] 65%|██████▌   | 32565/50000 [00:00<00:00, 88049.15 examples/s] 92%|█████████▏| 45833/50000 [00:00<00:00, 97931.75 examples/s]                                                               0 examples [00:00, ? examples/s]89 examples [00:00, 885.79 examples/s]199 examples [00:00, 940.45 examples/s]305 examples [00:00, 973.03 examples/s]411 examples [00:00, 997.04 examples/s]525 examples [00:00, 1034.29 examples/s]636 examples [00:00, 1053.39 examples/s]749 examples [00:00, 1072.69 examples/s]857 examples [00:00, 1074.06 examples/s]970 examples [00:00, 1090.23 examples/s]1081 examples [00:01, 1094.53 examples/s]1189 examples [00:01, 1088.56 examples/s]1302 examples [00:01, 1099.70 examples/s]1417 examples [00:01, 1111.53 examples/s]1531 examples [00:01, 1118.42 examples/s]1643 examples [00:01, 1118.49 examples/s]1755 examples [00:01, 1114.18 examples/s]1867 examples [00:01, 1115.39 examples/s]1980 examples [00:01, 1118.57 examples/s]2094 examples [00:01, 1122.52 examples/s]2207 examples [00:02, 1123.38 examples/s]2320 examples [00:02, 1118.88 examples/s]2434 examples [00:02, 1124.95 examples/s]2548 examples [00:02, 1127.17 examples/s]2663 examples [00:02, 1131.57 examples/s]2777 examples [00:02, 1116.37 examples/s]2889 examples [00:02, 1080.35 examples/s]2998 examples [00:02, 1051.89 examples/s]3110 examples [00:02, 1070.32 examples/s]3226 examples [00:02, 1093.40 examples/s]3338 examples [00:03, 1100.49 examples/s]3449 examples [00:03, 1060.54 examples/s]3563 examples [00:03, 1082.11 examples/s]3676 examples [00:03, 1094.50 examples/s]3790 examples [00:03, 1106.92 examples/s]3902 examples [00:03, 1108.96 examples/s]4014 examples [00:03, 1098.23 examples/s]4128 examples [00:03, 1108.97 examples/s]4240 examples [00:03, 1109.62 examples/s]4353 examples [00:03, 1115.33 examples/s]4465 examples [00:04, 1096.98 examples/s]4575 examples [00:04, 1097.19 examples/s]4689 examples [00:04, 1107.72 examples/s]4800 examples [00:04, 1102.97 examples/s]4911 examples [00:04, 1104.83 examples/s]5024 examples [00:04, 1111.45 examples/s]5136 examples [00:04, 1107.60 examples/s]5249 examples [00:04, 1112.91 examples/s]5364 examples [00:04, 1121.33 examples/s]5477 examples [00:04, 1111.03 examples/s]5589 examples [00:05, 1082.56 examples/s]5698 examples [00:05, 1057.84 examples/s]5806 examples [00:05, 1062.99 examples/s]5913 examples [00:05, 1048.67 examples/s]6019 examples [00:05, 1033.43 examples/s]6123 examples [00:05, 1024.04 examples/s]6226 examples [00:05, 1021.87 examples/s]6329 examples [00:05, 1015.10 examples/s]6437 examples [00:05, 1031.48 examples/s]6549 examples [00:06, 1055.55 examples/s]6655 examples [00:06, 1041.99 examples/s]6763 examples [00:06, 1051.85 examples/s]6871 examples [00:06, 1059.30 examples/s]6980 examples [00:06, 1066.57 examples/s]7090 examples [00:06, 1074.70 examples/s]7198 examples [00:06, 1070.42 examples/s]7307 examples [00:06, 1075.19 examples/s]7421 examples [00:06, 1092.56 examples/s]7531 examples [00:06, 1077.06 examples/s]7645 examples [00:07, 1094.09 examples/s]7755 examples [00:07, 1085.85 examples/s]7867 examples [00:07, 1094.13 examples/s]7980 examples [00:07, 1103.04 examples/s]8094 examples [00:07, 1110.97 examples/s]8206 examples [00:07, 1111.25 examples/s]8318 examples [00:07, 1096.74 examples/s]8428 examples [00:07, 1050.53 examples/s]8534 examples [00:07, 1050.26 examples/s]8645 examples [00:07, 1064.73 examples/s]8759 examples [00:08, 1084.07 examples/s]8868 examples [00:08, 1083.05 examples/s]8980 examples [00:08, 1093.82 examples/s]9094 examples [00:08, 1105.45 examples/s]9208 examples [00:08, 1115.26 examples/s]9324 examples [00:08, 1126.45 examples/s]9437 examples [00:08, 1125.22 examples/s]9550 examples [00:08, 1108.78 examples/s]9663 examples [00:08, 1114.66 examples/s]9777 examples [00:08, 1120.15 examples/s]9892 examples [00:09, 1128.71 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4DGNWF/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4DGNWF/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f80709c2a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f80709c2a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f80709c2a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f80bc32f0f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7ffac9d3c8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f80709c2a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f80709c2a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f80709c2a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7ffbd7fd30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f80588750b8>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.85 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.14 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.14 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.79 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.79 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.04 file/s]2020-07-30 06:09:47.077303: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-30 06:09:47.081398: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-30 06:09:47.082182: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5613df5c6910 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-30 06:09:47.082199: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  1%|          | 98304/9912422 [00:00<00:10, 978733.33it/s] 78%|███████▊  | 7692288/9912422 [00:00<00:01, 1390399.96it/s]9920512it [00:00, 44656991.61it/s]                            
0it [00:00, ?it/s]32768it [00:00, 535905.86it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 489439.86it/s]1654784it [00:00, 12109581.44it/s]                         
0it [00:00, ?it/s]8192it [00:00, 231005.37it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
