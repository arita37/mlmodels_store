
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fbd7c143ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fbd7c143ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fbde6db3510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fbde6db3510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fbe05fe0ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fbe05fe0ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbd940e3a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbd940e3a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbd940e3a60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████      | 4087808/9912422 [00:00<00:00, 40191632.90it/s]9920512it [00:00, 32326404.83it/s]                             
0it [00:00, ?it/s]32768it [00:00, 684232.04it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 139289.84it/s]1654784it [00:00, 10326990.66it/s]                         
0it [00:00, ?it/s]8192it [00:00, 164101.51it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbd921c0278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbd7c00f9b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fbd940e36a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fbd940e36a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fbd940e36a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:23,  1.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:22,  1.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:22,  1.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:21,  1.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:21,  1.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:20,  1.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:20,  1.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:56,  2.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:56,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:55,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:55,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:55,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:54,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:54,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:54,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:53,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:53,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:52,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:36,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:36,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:36,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:35,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:35,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:35,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:35,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:34,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:34,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:23,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:23,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:23,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:23,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:22,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:22,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:22,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:00<00:16,  7.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:16,  7.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:15,  7.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:15,  7.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:15,  7.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:15,  7.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:15,  7.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:14,  7.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:14,  7.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:10, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:09, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:09, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:09, 10.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:06, 14.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 25.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 40.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 40.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 49.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 58.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 58.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 66.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 66.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 72.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 79.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.16s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.16s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.16s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.57 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.97s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.16s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 79.57 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.97s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.97s/ file]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 40.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.97s/ url]
0 examples [00:00, ? examples/s]2020-08-05 00:08:43.913516: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-05 00:08:43.929404: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-08-05 00:08:43.929612: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560444515fb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-05 00:08:43.929632: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.19 examples/s]109 examples [00:00,  3.13 examples/s]235 examples [00:00,  4.46 examples/s]354 examples [00:00,  6.36 examples/s]470 examples [00:00,  9.07 examples/s]592 examples [00:00, 12.91 examples/s]716 examples [00:01, 18.36 examples/s]833 examples [00:01, 26.06 examples/s]955 examples [00:01, 36.89 examples/s]1071 examples [00:01, 51.99 examples/s]1188 examples [00:01, 72.88 examples/s]1303 examples [00:01, 101.20 examples/s]1416 examples [00:01, 139.19 examples/s]1530 examples [00:01, 188.94 examples/s]1643 examples [00:01, 251.81 examples/s]1756 examples [00:01, 327.88 examples/s]1869 examples [00:02, 416.48 examples/s]1990 examples [00:02, 518.01 examples/s]2112 examples [00:02, 626.03 examples/s]2237 examples [00:02, 736.21 examples/s]2357 examples [00:02, 827.90 examples/s]2476 examples [00:02, 907.18 examples/s]2595 examples [00:02, 976.53 examples/s]2714 examples [00:02, 1017.35 examples/s]2831 examples [00:02, 1053.18 examples/s]2948 examples [00:02, 1056.19 examples/s]3063 examples [00:03, 1080.98 examples/s]3191 examples [00:03, 1131.27 examples/s]3317 examples [00:03, 1166.10 examples/s]3437 examples [00:03, 1173.93 examples/s]3557 examples [00:03, 1170.66 examples/s]3676 examples [00:03, 1153.45 examples/s]3793 examples [00:03, 1156.05 examples/s]3921 examples [00:03, 1188.36 examples/s]4041 examples [00:03, 1176.12 examples/s]4160 examples [00:04, 1179.69 examples/s]4279 examples [00:04, 1182.18 examples/s]4406 examples [00:04, 1204.50 examples/s]4528 examples [00:04, 1208.81 examples/s]4654 examples [00:04, 1221.43 examples/s]4777 examples [00:04, 1222.87 examples/s]4900 examples [00:04, 1203.29 examples/s]5024 examples [00:04, 1213.11 examples/s]5151 examples [00:04, 1227.33 examples/s]5274 examples [00:04, 1217.68 examples/s]5396 examples [00:05, 1210.12 examples/s]5518 examples [00:05, 1176.86 examples/s]5645 examples [00:05, 1200.28 examples/s]5769 examples [00:05, 1211.46 examples/s]5891 examples [00:05, 1176.48 examples/s]6010 examples [00:05, 1114.18 examples/s]6123 examples [00:05, 1096.15 examples/s]6238 examples [00:05, 1111.35 examples/s]6350 examples [00:05, 1067.77 examples/s]6467 examples [00:05, 1094.53 examples/s]6587 examples [00:06, 1121.87 examples/s]6702 examples [00:06, 1128.77 examples/s]6824 examples [00:06, 1152.02 examples/s]6940 examples [00:06, 1122.67 examples/s]7064 examples [00:06, 1154.43 examples/s]7182 examples [00:06, 1159.95 examples/s]7299 examples [00:06, 1156.68 examples/s]7423 examples [00:06, 1177.45 examples/s]7542 examples [00:06, 1152.77 examples/s]7661 examples [00:07, 1162.43 examples/s]7784 examples [00:07, 1179.81 examples/s]7903 examples [00:07, 1181.05 examples/s]8028 examples [00:07, 1199.95 examples/s]8149 examples [00:07, 1202.90 examples/s]8270 examples [00:07, 1169.95 examples/s]8388 examples [00:07, 1141.38 examples/s]8503 examples [00:07, 1070.77 examples/s]8617 examples [00:07, 1089.52 examples/s]8727 examples [00:07, 1062.47 examples/s]8835 examples [00:08, 1054.10 examples/s]8945 examples [00:08, 1066.37 examples/s]9053 examples [00:08, 1061.66 examples/s]9170 examples [00:08, 1090.47 examples/s]9290 examples [00:08, 1121.04 examples/s]9415 examples [00:08, 1155.25 examples/s]9532 examples [00:08, 1153.42 examples/s]9648 examples [00:08, 1154.61 examples/s]9764 examples [00:08, 1143.69 examples/s]9884 examples [00:08, 1159.21 examples/s]10001 examples [00:09, 1088.19 examples/s]10113 examples [00:09, 1095.96 examples/s]10225 examples [00:09, 1101.70 examples/s]10348 examples [00:09, 1137.16 examples/s]10470 examples [00:09, 1158.94 examples/s]10587 examples [00:09, 1149.91 examples/s]10705 examples [00:09, 1155.75 examples/s]10828 examples [00:09, 1176.79 examples/s]10951 examples [00:09, 1191.15 examples/s]11071 examples [00:10, 1188.96 examples/s]11191 examples [00:10, 1178.35 examples/s]11309 examples [00:10, 1148.60 examples/s]11426 examples [00:10, 1154.23 examples/s]11544 examples [00:10, 1158.84 examples/s]11661 examples [00:10, 1160.80 examples/s]11786 examples [00:10, 1185.44 examples/s]11905 examples [00:10, 1172.00 examples/s]12025 examples [00:10, 1180.26 examples/s]12150 examples [00:10, 1199.34 examples/s]12271 examples [00:11, 1198.26 examples/s]12391 examples [00:11, 1190.12 examples/s]12512 examples [00:11, 1192.59 examples/s]12632 examples [00:11, 1182.24 examples/s]12751 examples [00:11, 1165.31 examples/s]12868 examples [00:11, 1160.25 examples/s]12994 examples [00:11, 1187.84 examples/s]13113 examples [00:11, 1182.92 examples/s]13232 examples [00:11, 1171.84 examples/s]13350 examples [00:11, 1167.56 examples/s]13473 examples [00:12, 1183.92 examples/s]13594 examples [00:12, 1189.00 examples/s]13713 examples [00:12, 1110.38 examples/s]13835 examples [00:12, 1140.05 examples/s]13950 examples [00:12, 1141.27 examples/s]14072 examples [00:12, 1163.37 examples/s]14189 examples [00:12, 1165.11 examples/s]14306 examples [00:12, 1146.69 examples/s]14423 examples [00:12, 1150.95 examples/s]14539 examples [00:12, 1152.35 examples/s]14656 examples [00:13, 1155.37 examples/s]14780 examples [00:13, 1177.58 examples/s]14898 examples [00:13, 1164.31 examples/s]15020 examples [00:13, 1178.37 examples/s]15138 examples [00:13, 1171.35 examples/s]15256 examples [00:13, 1168.74 examples/s]15373 examples [00:13, 1158.67 examples/s]15490 examples [00:13, 1160.06 examples/s]15607 examples [00:13, 1159.90 examples/s]15724 examples [00:14, 1156.19 examples/s]15840 examples [00:14, 1144.69 examples/s]15956 examples [00:14, 1149.17 examples/s]16071 examples [00:14, 1115.72 examples/s]16186 examples [00:14, 1124.03 examples/s]16304 examples [00:14, 1138.41 examples/s]16419 examples [00:14, 1141.30 examples/s]16534 examples [00:14, 1143.22 examples/s]16649 examples [00:14, 1140.47 examples/s]16764 examples [00:14, 1140.46 examples/s]16882 examples [00:15, 1150.68 examples/s]16998 examples [00:15, 1142.51 examples/s]17113 examples [00:15, 1141.09 examples/s]17228 examples [00:15, 1139.63 examples/s]17343 examples [00:15, 1139.97 examples/s]17458 examples [00:15, 1119.35 examples/s]17571 examples [00:15, 1107.95 examples/s]17682 examples [00:15, 1104.69 examples/s]17793 examples [00:15, 1093.32 examples/s]17903 examples [00:15, 1091.50 examples/s]18020 examples [00:16, 1112.82 examples/s]18134 examples [00:16, 1119.95 examples/s]18248 examples [00:16, 1124.56 examples/s]18361 examples [00:16, 1115.92 examples/s]18473 examples [00:16, 1068.00 examples/s]18581 examples [00:16, 1055.92 examples/s]18698 examples [00:16, 1086.21 examples/s]18811 examples [00:16, 1097.87 examples/s]18925 examples [00:16, 1107.53 examples/s]19040 examples [00:16, 1117.46 examples/s]19162 examples [00:17, 1144.43 examples/s]19280 examples [00:17, 1154.73 examples/s]19403 examples [00:17, 1175.63 examples/s]19524 examples [00:17, 1182.68 examples/s]19643 examples [00:17, 1153.20 examples/s]19761 examples [00:17, 1158.37 examples/s]19878 examples [00:17, 1144.14 examples/s]19993 examples [00:17, 1138.77 examples/s]20108 examples [00:17, 1078.84 examples/s]20217 examples [00:18, 1069.24 examples/s]20331 examples [00:18, 1087.40 examples/s]20446 examples [00:18, 1105.42 examples/s]20561 examples [00:18, 1118.07 examples/s]20675 examples [00:18, 1123.33 examples/s]20794 examples [00:18, 1140.64 examples/s]20912 examples [00:18, 1150.77 examples/s]21029 examples [00:18, 1154.52 examples/s]21145 examples [00:18, 1151.34 examples/s]21261 examples [00:18, 1140.46 examples/s]21376 examples [00:19, 1141.44 examples/s]21491 examples [00:19, 1135.41 examples/s]21605 examples [00:19, 1110.46 examples/s]21717 examples [00:19, 1099.78 examples/s]21829 examples [00:19, 1105.00 examples/s]21949 examples [00:19, 1129.35 examples/s]22063 examples [00:19, 1119.66 examples/s]22176 examples [00:19, 1119.30 examples/s]22297 examples [00:19, 1144.05 examples/s]22412 examples [00:19, 1133.38 examples/s]22532 examples [00:20, 1151.17 examples/s]22648 examples [00:20, 1140.75 examples/s]22763 examples [00:20, 1120.49 examples/s]22882 examples [00:20, 1138.62 examples/s]22997 examples [00:20, 1061.42 examples/s]23105 examples [00:20, 993.35 examples/s] 23214 examples [00:20, 1018.33 examples/s]23318 examples [00:20, 945.31 examples/s] 23415 examples [00:20, 921.17 examples/s]23513 examples [00:21, 937.34 examples/s]23608 examples [00:21, 926.98 examples/s]23702 examples [00:21, 928.67 examples/s]23796 examples [00:21, 899.00 examples/s]23893 examples [00:21, 917.41 examples/s]23993 examples [00:21, 938.63 examples/s]24110 examples [00:21, 995.95 examples/s]24222 examples [00:21, 1029.51 examples/s]24333 examples [00:21, 1052.24 examples/s]24445 examples [00:21, 1069.79 examples/s]24563 examples [00:22, 1099.35 examples/s]24681 examples [00:22, 1121.93 examples/s]24794 examples [00:22, 1122.92 examples/s]24910 examples [00:22, 1131.13 examples/s]25024 examples [00:22, 1130.11 examples/s]25138 examples [00:22, 1124.21 examples/s]25251 examples [00:22, 1116.81 examples/s]25363 examples [00:22, 1088.73 examples/s]25477 examples [00:22, 1101.61 examples/s]25588 examples [00:22, 1103.06 examples/s]25702 examples [00:23, 1111.92 examples/s]25818 examples [00:23, 1124.31 examples/s]25932 examples [00:23, 1128.01 examples/s]26051 examples [00:23, 1145.64 examples/s]26166 examples [00:23, 1133.43 examples/s]26282 examples [00:23, 1138.63 examples/s]26396 examples [00:23, 1119.92 examples/s]26510 examples [00:23, 1123.70 examples/s]26625 examples [00:23, 1131.46 examples/s]26739 examples [00:24, 1122.04 examples/s]26855 examples [00:24, 1131.81 examples/s]26971 examples [00:24, 1138.18 examples/s]27085 examples [00:24, 1125.20 examples/s]27203 examples [00:24, 1138.62 examples/s]27321 examples [00:24, 1147.81 examples/s]27436 examples [00:24, 1135.13 examples/s]27554 examples [00:24, 1146.13 examples/s]27669 examples [00:24, 1130.72 examples/s]27788 examples [00:24, 1146.09 examples/s]27903 examples [00:25, 1141.37 examples/s]28018 examples [00:25, 1137.66 examples/s]28132 examples [00:25, 1129.88 examples/s]28246 examples [00:25, 1107.89 examples/s]28361 examples [00:25, 1119.67 examples/s]28474 examples [00:25, 1082.29 examples/s]28583 examples [00:25, 1082.88 examples/s]28692 examples [00:25, 1070.60 examples/s]28800 examples [00:25, 1011.93 examples/s]28916 examples [00:25, 1050.64 examples/s]29030 examples [00:26, 1075.17 examples/s]29153 examples [00:26, 1115.81 examples/s]29276 examples [00:26, 1147.40 examples/s]29392 examples [00:26, 1124.71 examples/s]29511 examples [00:26, 1142.28 examples/s]29626 examples [00:26, 1132.93 examples/s]29749 examples [00:26, 1159.21 examples/s]29869 examples [00:26, 1170.70 examples/s]29988 examples [00:26, 1175.17 examples/s]30106 examples [00:27, 1095.34 examples/s]30221 examples [00:27, 1109.58 examples/s]30338 examples [00:27, 1126.93 examples/s]30461 examples [00:27, 1153.50 examples/s]30578 examples [00:27, 1146.61 examples/s]30694 examples [00:27, 1121.90 examples/s]30815 examples [00:27, 1144.14 examples/s]30933 examples [00:27, 1152.31 examples/s]31049 examples [00:27, 1144.41 examples/s]31165 examples [00:27, 1147.87 examples/s]31280 examples [00:28, 1136.46 examples/s]31394 examples [00:28, 1126.40 examples/s]31507 examples [00:28, 1126.49 examples/s]31623 examples [00:28, 1135.46 examples/s]31743 examples [00:28, 1154.08 examples/s]31860 examples [00:28, 1157.15 examples/s]31976 examples [00:28, 1143.06 examples/s]32098 examples [00:28, 1163.72 examples/s]32215 examples [00:28, 1161.55 examples/s]32332 examples [00:28, 1156.19 examples/s]32448 examples [00:29, 1140.17 examples/s]32563 examples [00:29, 1142.68 examples/s]32685 examples [00:29, 1162.18 examples/s]32802 examples [00:29, 1154.34 examples/s]32918 examples [00:29, 1153.95 examples/s]33034 examples [00:29, 1146.16 examples/s]33149 examples [00:29, 1131.54 examples/s]33263 examples [00:29, 1133.19 examples/s]33377 examples [00:29, 1132.55 examples/s]33492 examples [00:29, 1136.78 examples/s]33608 examples [00:30, 1143.01 examples/s]33724 examples [00:30, 1133.87 examples/s]33844 examples [00:30, 1150.29 examples/s]33963 examples [00:30, 1161.62 examples/s]34080 examples [00:30, 1154.85 examples/s]34196 examples [00:30, 1135.41 examples/s]34317 examples [00:30, 1154.92 examples/s]34438 examples [00:30, 1170.24 examples/s]34556 examples [00:30, 1163.99 examples/s]34678 examples [00:30, 1179.44 examples/s]34797 examples [00:31, 1181.12 examples/s]34916 examples [00:31, 1178.44 examples/s]35034 examples [00:31, 1175.76 examples/s]35152 examples [00:31, 1126.64 examples/s]35270 examples [00:31, 1140.67 examples/s]35385 examples [00:31, 1118.80 examples/s]35498 examples [00:31, 1122.13 examples/s]35621 examples [00:31, 1151.23 examples/s]35737 examples [00:31, 1138.28 examples/s]35852 examples [00:32, 1130.41 examples/s]35966 examples [00:32, 1127.02 examples/s]36082 examples [00:32, 1136.63 examples/s]36199 examples [00:32, 1144.98 examples/s]36316 examples [00:32, 1149.68 examples/s]36432 examples [00:32, 1145.22 examples/s]36547 examples [00:32, 1136.63 examples/s]36661 examples [00:32, 1124.31 examples/s]36775 examples [00:32, 1126.84 examples/s]36888 examples [00:32, 1126.90 examples/s]37001 examples [00:33, 1122.39 examples/s]37114 examples [00:33, 1112.22 examples/s]37229 examples [00:33, 1123.09 examples/s]37348 examples [00:33, 1139.19 examples/s]37468 examples [00:33, 1156.15 examples/s]37584 examples [00:33, 1153.54 examples/s]37700 examples [00:33, 1134.98 examples/s]37817 examples [00:33, 1144.05 examples/s]37934 examples [00:33, 1148.71 examples/s]38049 examples [00:33, 1132.27 examples/s]38164 examples [00:34, 1136.31 examples/s]38278 examples [00:34, 1118.55 examples/s]38390 examples [00:34, 1117.30 examples/s]38502 examples [00:34, 1108.66 examples/s]38618 examples [00:34, 1120.71 examples/s]38733 examples [00:34, 1128.37 examples/s]38846 examples [00:34, 1122.02 examples/s]38969 examples [00:34, 1151.32 examples/s]39091 examples [00:34, 1168.46 examples/s]39209 examples [00:34, 1149.03 examples/s]39325 examples [00:35, 1138.15 examples/s]39439 examples [00:35, 1113.40 examples/s]39554 examples [00:35, 1123.38 examples/s]39673 examples [00:35, 1141.39 examples/s]39788 examples [00:35, 1125.51 examples/s]39901 examples [00:35, 1111.55 examples/s]40013 examples [00:35, 1043.62 examples/s]40127 examples [00:35, 1070.59 examples/s]40240 examples [00:35, 1085.56 examples/s]40354 examples [00:36, 1100.96 examples/s]40466 examples [00:36, 1104.78 examples/s]40577 examples [00:36, 1105.58 examples/s]40688 examples [00:36, 1087.41 examples/s]40797 examples [00:36, 1050.17 examples/s]40908 examples [00:36, 1067.27 examples/s]41016 examples [00:36, 1037.99 examples/s]41121 examples [00:36, 1038.26 examples/s]41226 examples [00:36, 1024.64 examples/s]41334 examples [00:36, 1039.41 examples/s]41444 examples [00:37, 1056.26 examples/s]41551 examples [00:37, 1059.70 examples/s]41658 examples [00:37, 1049.04 examples/s]41769 examples [00:37, 1063.33 examples/s]41876 examples [00:37, 1059.99 examples/s]41989 examples [00:37, 1077.95 examples/s]42105 examples [00:37, 1098.76 examples/s]42216 examples [00:37, 1087.45 examples/s]42326 examples [00:37, 1087.02 examples/s]42438 examples [00:37, 1096.21 examples/s]42548 examples [00:38, 1096.65 examples/s]42658 examples [00:38, 1092.33 examples/s]42771 examples [00:38, 1102.07 examples/s]42885 examples [00:38, 1112.59 examples/s]43007 examples [00:38, 1141.52 examples/s]43122 examples [00:38, 1141.64 examples/s]43243 examples [00:38, 1160.32 examples/s]43360 examples [00:38, 1147.74 examples/s]43475 examples [00:38, 1130.51 examples/s]43589 examples [00:38, 1112.76 examples/s]43701 examples [00:39, 1101.38 examples/s]43822 examples [00:39, 1131.44 examples/s]43936 examples [00:39, 1120.24 examples/s]44053 examples [00:39, 1133.92 examples/s]44169 examples [00:39, 1140.89 examples/s]44286 examples [00:39, 1146.70 examples/s]44401 examples [00:39, 1134.54 examples/s]44520 examples [00:39, 1148.49 examples/s]44635 examples [00:39, 1119.49 examples/s]44761 examples [00:40, 1157.90 examples/s]44878 examples [00:40, 1136.48 examples/s]44993 examples [00:40, 1112.62 examples/s]45105 examples [00:40, 1102.44 examples/s]45225 examples [00:40, 1129.09 examples/s]45340 examples [00:40, 1133.23 examples/s]45456 examples [00:40, 1139.55 examples/s]45571 examples [00:40, 1137.94 examples/s]45695 examples [00:40, 1165.99 examples/s]45813 examples [00:40, 1168.72 examples/s]45931 examples [00:41, 1158.98 examples/s]46053 examples [00:41, 1175.78 examples/s]46178 examples [00:41, 1197.09 examples/s]46298 examples [00:41, 1192.95 examples/s]46418 examples [00:41, 1183.01 examples/s]46537 examples [00:41, 1166.71 examples/s]46655 examples [00:41, 1168.13 examples/s]46772 examples [00:41, 1130.24 examples/s]46886 examples [00:41, 1095.94 examples/s]46997 examples [00:41, 1064.79 examples/s]47104 examples [00:42, 1062.72 examples/s]47216 examples [00:42, 1075.82 examples/s]47324 examples [00:42, 1042.14 examples/s]47429 examples [00:42, 991.61 examples/s] 47537 examples [00:42, 1015.32 examples/s]47641 examples [00:42, 1022.18 examples/s]47744 examples [00:42, 1014.48 examples/s]47846 examples [00:42, 999.88 examples/s] 47952 examples [00:42, 1015.12 examples/s]48054 examples [00:43, 1015.43 examples/s]48161 examples [00:43, 1029.37 examples/s]48265 examples [00:43, 964.75 examples/s] 48371 examples [00:43, 989.17 examples/s]48471 examples [00:43, 983.20 examples/s]48582 examples [00:43, 1017.13 examples/s]48687 examples [00:43, 1026.55 examples/s]48791 examples [00:43, 1026.74 examples/s]48900 examples [00:43, 1043.90 examples/s]49009 examples [00:43, 1056.22 examples/s]49120 examples [00:44, 1071.75 examples/s]49233 examples [00:44, 1087.87 examples/s]49353 examples [00:44, 1118.17 examples/s]49470 examples [00:44, 1132.36 examples/s]49585 examples [00:44, 1136.23 examples/s]49714 examples [00:44, 1175.65 examples/s]49836 examples [00:44, 1187.77 examples/s]49956 examples [00:44, 1188.27 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7340/50000 [00:00<00:00, 73397.62 examples/s] 41%|████      | 20416/50000 [00:00<00:00, 84520.39 examples/s] 67%|██████▋   | 33726/50000 [00:00<00:00, 94912.30 examples/s] 96%|█████████▌| 47855/50000 [00:00<00:00, 105278.15 examples/s]                                                                0 examples [00:00, ? examples/s]92 examples [00:00, 912.89 examples/s]208 examples [00:00, 973.88 examples/s]317 examples [00:00, 1005.11 examples/s]429 examples [00:00, 1035.52 examples/s]541 examples [00:00, 1057.14 examples/s]654 examples [00:00, 1076.55 examples/s]775 examples [00:00, 1110.45 examples/s]890 examples [00:00, 1121.81 examples/s]1005 examples [00:00, 1128.67 examples/s]1121 examples [00:01, 1135.61 examples/s]1233 examples [00:01, 1095.77 examples/s]1342 examples [00:01, 1080.15 examples/s]1450 examples [00:01, 1067.92 examples/s]1557 examples [00:01, 1066.33 examples/s]1665 examples [00:01, 1070.18 examples/s]1774 examples [00:01, 1074.09 examples/s]1890 examples [00:01, 1095.71 examples/s]2006 examples [00:01, 1112.39 examples/s]2118 examples [00:01, 1109.88 examples/s]2230 examples [00:02, 1112.52 examples/s]2346 examples [00:02, 1126.00 examples/s]2463 examples [00:02, 1137.92 examples/s]2585 examples [00:02, 1159.02 examples/s]2702 examples [00:02, 1152.44 examples/s]2818 examples [00:02, 1132.92 examples/s]2932 examples [00:02, 1122.69 examples/s]3045 examples [00:02, 1116.17 examples/s]3158 examples [00:02, 1118.75 examples/s]3274 examples [00:02, 1130.38 examples/s]3393 examples [00:03, 1145.78 examples/s]3508 examples [00:03, 1146.37 examples/s]3628 examples [00:03, 1159.10 examples/s]3754 examples [00:03, 1186.77 examples/s]3873 examples [00:03, 1168.71 examples/s]3993 examples [00:03, 1175.69 examples/s]4111 examples [00:03, 1122.06 examples/s]4228 examples [00:03, 1134.18 examples/s]4342 examples [00:03, 1118.20 examples/s]4468 examples [00:03, 1155.57 examples/s]4595 examples [00:04, 1185.33 examples/s]4715 examples [00:04, 1188.07 examples/s]4835 examples [00:04, 1187.71 examples/s]4958 examples [00:04, 1196.89 examples/s]5078 examples [00:04, 1173.66 examples/s]5199 examples [00:04, 1181.43 examples/s]5318 examples [00:04, 1155.09 examples/s]5437 examples [00:04, 1163.62 examples/s]5554 examples [00:04, 1148.84 examples/s]5670 examples [00:05, 1151.46 examples/s]5801 examples [00:05, 1193.44 examples/s]5923 examples [00:05, 1198.78 examples/s]6044 examples [00:05, 1162.39 examples/s]6164 examples [00:05, 1171.19 examples/s]6287 examples [00:05, 1187.80 examples/s]6407 examples [00:05, 1183.14 examples/s]6526 examples [00:05, 1151.32 examples/s]6651 examples [00:05, 1177.33 examples/s]6775 examples [00:05, 1192.44 examples/s]6903 examples [00:06, 1215.00 examples/s]7027 examples [00:06, 1220.47 examples/s]7150 examples [00:06, 1201.26 examples/s]7271 examples [00:06, 1182.67 examples/s]7392 examples [00:06, 1189.36 examples/s]7514 examples [00:06, 1197.45 examples/s]7634 examples [00:06, 1086.73 examples/s]7752 examples [00:06, 1111.98 examples/s]7876 examples [00:06, 1145.53 examples/s]8001 examples [00:06, 1172.75 examples/s]8125 examples [00:07, 1191.82 examples/s]8251 examples [00:07, 1211.40 examples/s]8374 examples [00:07, 1216.07 examples/s]8497 examples [00:07, 1219.98 examples/s]8620 examples [00:07, 1193.39 examples/s]8740 examples [00:07, 1194.03 examples/s]8860 examples [00:07, 1189.13 examples/s]8983 examples [00:07, 1200.84 examples/s]9104 examples [00:07, 1197.56 examples/s]9224 examples [00:08, 1146.34 examples/s]9353 examples [00:08, 1184.75 examples/s]9475 examples [00:08, 1192.21 examples/s]9595 examples [00:08, 1183.77 examples/s]9725 examples [00:08, 1215.31 examples/s]9853 examples [00:08, 1233.15 examples/s]9977 examples [00:08, 1211.54 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWXYVH3/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWXYVH3/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbd940e3a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbd940e3a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbd940e3a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbddfac70b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbd2b9be400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbd940e3a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbd940e3a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbd940e3a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbd7c00f588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbd7c00f470>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.84 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.74 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.74 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.72 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.72 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.52 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.52 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.08 file/s]2020-08-05 00:09:44.390815: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-05 00:09:44.394779: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-08-05 00:09:44.394957: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5574ca4dae10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-05 00:09:44.394972: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  1%|          | 98304/9912422 [00:00<00:10, 976707.00it/s] 65%|██████▍   | 6406144/9912422 [00:00<00:02, 1386097.53it/s]9920512it [00:00, 42307161.33it/s]                            
0it [00:00, ?it/s]32768it [00:00, 688103.06it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 478591.21it/s]1654784it [00:00, 12046151.33it/s]                         
0it [00:00, ?it/s]8192it [00:00, 192209.41it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
