
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/474a593f862813bf55ed98b2a6b35150c729973c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '474a593f862813bf55ed98b2a6b35150c729973c', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/474a593f862813bf55ed98b2a6b35150c729973c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/474a593f862813bf55ed98b2a6b35150c729973c

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/474a593f862813bf55ed98b2a6b35150c729973c

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fa2662731e0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fa2662731e0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fa2d0f99400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fa2d0f99400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fa2f01b5ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fa2f01b5ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa27e2d40d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa27e2d40d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa27e2d40d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:15, 131257.51it/s] 43%|████▎     | 4276224/9912422 [00:00<00:30, 187261.31it/s]9920512it [00:00, 37269598.43it/s]                           
0it [00:00, ?it/s]32768it [00:00, 628619.96it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1010287.97it/s]1654784it [00:00, 12546783.16it/s]                           
0it [00:00, ?it/s]8192it [00:00, 384388.71it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa27c481b70>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa27c496198>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fa27e2ccd08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fa27e2ccd08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fa27e2ccd08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:13,  2.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:13,  2.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:12,  2.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:12,  2.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:11,  2.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:11,  2.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:10,  2.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:49,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:49,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:49,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:48,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:48,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:48,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:47,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:47,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:47,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:32,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:32,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:32,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:31,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:31,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:31,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:31,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  6.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:20,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:20,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:20,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  8.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:13,  8.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 11.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 11.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 26.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 49.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 57.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 57.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 57.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 57.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 57.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 57.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 57.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 57.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 63.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 63.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 70.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 70.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 70.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 70.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 70.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 70.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 70.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 74.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.93 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.51s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.93 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.51s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.51s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.89 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.51s/ url]
0 examples [00:00, ? examples/s]2020-08-20 06:07:55.164498: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-20 06:07:55.181212: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-08-20 06:07:55.181495: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56318b9b4850 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-20 06:07:55.181515: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
42 examples [00:00, 419.88 examples/s]143 examples [00:00, 508.83 examples/s]249 examples [00:00, 602.05 examples/s]355 examples [00:00, 690.81 examples/s]461 examples [00:00, 771.01 examples/s]566 examples [00:00, 836.15 examples/s]672 examples [00:00, 891.61 examples/s]777 examples [00:00, 932.96 examples/s]882 examples [00:00, 963.65 examples/s]988 examples [00:01, 988.89 examples/s]1091 examples [00:01, 999.63 examples/s]1196 examples [00:01, 1012.08 examples/s]1299 examples [00:01, 984.33 examples/s] 1401 examples [00:01, 992.93 examples/s]1507 examples [00:01, 1011.09 examples/s]1611 examples [00:01, 1019.50 examples/s]1718 examples [00:01, 1031.33 examples/s]1822 examples [00:01, 1010.81 examples/s]1927 examples [00:01, 1020.89 examples/s]2031 examples [00:02, 1024.94 examples/s]2134 examples [00:02, 1010.42 examples/s]2236 examples [00:02, 1011.82 examples/s]2341 examples [00:02, 1020.27 examples/s]2446 examples [00:02, 1027.03 examples/s]2549 examples [00:02, 1025.62 examples/s]2654 examples [00:02, 1032.64 examples/s]2760 examples [00:02, 1038.71 examples/s]2864 examples [00:02, 1037.14 examples/s]2968 examples [00:02, 1029.27 examples/s]3073 examples [00:03, 1035.19 examples/s]3179 examples [00:03, 1040.79 examples/s]3284 examples [00:03, 1041.13 examples/s]3389 examples [00:03, 1024.90 examples/s]3494 examples [00:03, 1032.04 examples/s]3598 examples [00:03, 1032.25 examples/s]3703 examples [00:03, 1035.38 examples/s]3809 examples [00:03, 1041.64 examples/s]3914 examples [00:03, 1040.71 examples/s]4020 examples [00:03, 1044.38 examples/s]4125 examples [00:04, 1041.32 examples/s]4230 examples [00:04, 1043.84 examples/s]4336 examples [00:04, 1046.60 examples/s]4441 examples [00:04, 1043.75 examples/s]4546 examples [00:04, 1036.75 examples/s]4652 examples [00:04, 1040.65 examples/s]4759 examples [00:04, 1046.14 examples/s]4865 examples [00:04, 1047.81 examples/s]4971 examples [00:04, 1049.28 examples/s]5076 examples [00:04, 1035.94 examples/s]5180 examples [00:05, 1025.56 examples/s]5283 examples [00:05, 1026.17 examples/s]5388 examples [00:05, 1032.31 examples/s]5492 examples [00:05, 1014.11 examples/s]5595 examples [00:05, 1018.66 examples/s]5701 examples [00:05, 1030.38 examples/s]5807 examples [00:05, 1038.63 examples/s]5911 examples [00:05, 1039.03 examples/s]6015 examples [00:05, 1038.67 examples/s]6119 examples [00:05, 1021.31 examples/s]6224 examples [00:06, 1029.67 examples/s]6330 examples [00:06, 1036.23 examples/s]6434 examples [00:06, 1019.13 examples/s]6537 examples [00:06, 1008.34 examples/s]6638 examples [00:06, 982.41 examples/s] 6741 examples [00:06, 986.16 examples/s]6846 examples [00:06, 1003.40 examples/s]6948 examples [00:06, 1007.81 examples/s]7052 examples [00:06, 1016.79 examples/s]7159 examples [00:07, 1030.93 examples/s]7263 examples [00:07, 1030.10 examples/s]7370 examples [00:07, 1038.95 examples/s]7474 examples [00:07, 1012.83 examples/s]7576 examples [00:07, 1009.86 examples/s]7682 examples [00:07, 1024.03 examples/s]7789 examples [00:07, 1035.32 examples/s]7896 examples [00:07, 1044.34 examples/s]8002 examples [00:07, 1046.21 examples/s]8107 examples [00:07, 1042.27 examples/s]8214 examples [00:08, 1050.14 examples/s]8320 examples [00:08, 1049.46 examples/s]8425 examples [00:08, 1034.00 examples/s]8530 examples [00:08, 1038.43 examples/s]8634 examples [00:08, 999.19 examples/s] 8735 examples [00:08, 985.44 examples/s]8840 examples [00:08, 1002.11 examples/s]8946 examples [00:08, 1017.32 examples/s]9048 examples [00:08, 1014.43 examples/s]9152 examples [00:08, 1020.88 examples/s]9259 examples [00:09, 1032.99 examples/s]9363 examples [00:09, 1026.90 examples/s]9469 examples [00:09, 1036.08 examples/s]9573 examples [00:09, 1011.25 examples/s]9675 examples [00:09, 1010.76 examples/s]9779 examples [00:09, 1017.50 examples/s]9885 examples [00:09, 1029.48 examples/s]9991 examples [00:09, 1035.86 examples/s]10095 examples [00:09, 969.84 examples/s]10195 examples [00:09, 977.66 examples/s]10300 examples [00:10, 995.98 examples/s]10404 examples [00:10, 1008.37 examples/s]10509 examples [00:10, 1018.19 examples/s]10613 examples [00:10, 1024.52 examples/s]10717 examples [00:10, 1026.84 examples/s]10823 examples [00:10, 1036.32 examples/s]10929 examples [00:10, 1042.90 examples/s]11035 examples [00:10, 1045.29 examples/s]11140 examples [00:10, 1040.42 examples/s]11245 examples [00:10, 1041.11 examples/s]11351 examples [00:11, 1045.76 examples/s]11456 examples [00:11, 1037.83 examples/s]11562 examples [00:11, 1043.43 examples/s]11667 examples [00:11, 1012.16 examples/s]11769 examples [00:11, 980.92 examples/s] 11873 examples [00:11, 996.71 examples/s]11978 examples [00:11, 1010.53 examples/s]12083 examples [00:11, 1019.83 examples/s]12186 examples [00:11, 1006.80 examples/s]12290 examples [00:12, 1015.33 examples/s]12392 examples [00:12, 1005.97 examples/s]12496 examples [00:12, 1015.71 examples/s]12598 examples [00:12, 1006.12 examples/s]12703 examples [00:12, 1017.08 examples/s]12809 examples [00:12, 1029.14 examples/s]12913 examples [00:12, 1032.07 examples/s]13019 examples [00:12, 1039.62 examples/s]13125 examples [00:12, 1043.91 examples/s]13230 examples [00:12, 1038.94 examples/s]13334 examples [00:13, 1034.65 examples/s]13438 examples [00:13, 1032.50 examples/s]13542 examples [00:13, 1027.40 examples/s]13648 examples [00:13, 1035.80 examples/s]13753 examples [00:13, 1038.34 examples/s]13857 examples [00:13, 1025.00 examples/s]13962 examples [00:13, 1031.89 examples/s]14066 examples [00:13, 1030.13 examples/s]14170 examples [00:13, 1028.63 examples/s]14275 examples [00:13, 1032.39 examples/s]14379 examples [00:14, 1013.73 examples/s]14484 examples [00:14, 1022.92 examples/s]14588 examples [00:14, 1025.42 examples/s]14693 examples [00:14, 1032.37 examples/s]14799 examples [00:14, 1039.40 examples/s]14903 examples [00:14, 1019.39 examples/s]15010 examples [00:14, 1032.77 examples/s]15114 examples [00:14, 1028.81 examples/s]15220 examples [00:14, 1037.60 examples/s]15327 examples [00:14, 1044.97 examples/s]15432 examples [00:15, 1023.83 examples/s]15538 examples [00:15, 1032.95 examples/s]15643 examples [00:15, 1037.79 examples/s]15748 examples [00:15, 1041.07 examples/s]15853 examples [00:15, 1040.23 examples/s]15958 examples [00:15, 1024.13 examples/s]16064 examples [00:15, 1032.65 examples/s]16170 examples [00:15, 1040.51 examples/s]16275 examples [00:15, 1023.43 examples/s]16381 examples [00:15, 1034.06 examples/s]16485 examples [00:16, 1027.74 examples/s]16591 examples [00:16, 1036.09 examples/s]16695 examples [00:16, 1033.28 examples/s]16801 examples [00:16, 1038.36 examples/s]16907 examples [00:16, 1043.44 examples/s]17012 examples [00:16, 1038.96 examples/s]17116 examples [00:16, 1037.35 examples/s]17221 examples [00:16, 1040.93 examples/s]17326 examples [00:16, 1036.89 examples/s]17431 examples [00:17, 1040.65 examples/s]17536 examples [00:17, 1030.32 examples/s]17640 examples [00:17, 1014.99 examples/s]17742 examples [00:17, 1004.96 examples/s]17843 examples [00:17, 1003.13 examples/s]17944 examples [00:17, 987.69 examples/s] 18047 examples [00:17, 999.82 examples/s]18152 examples [00:17, 1013.23 examples/s]18257 examples [00:17, 1022.96 examples/s]18362 examples [00:17, 1030.65 examples/s]18467 examples [00:18, 1033.59 examples/s]18571 examples [00:18, 1031.22 examples/s]18677 examples [00:18, 1038.77 examples/s]18781 examples [00:18, 1021.93 examples/s]18887 examples [00:18, 1030.48 examples/s]18993 examples [00:18, 1036.92 examples/s]19097 examples [00:18, 1030.06 examples/s]19203 examples [00:18, 1037.91 examples/s]19307 examples [00:18, 1034.28 examples/s]19413 examples [00:18, 1039.87 examples/s]19519 examples [00:19, 1044.23 examples/s]19624 examples [00:19, 1025.97 examples/s]19727 examples [00:19, 1022.42 examples/s]19830 examples [00:19, 1024.28 examples/s]19933 examples [00:19, 1017.70 examples/s]20035 examples [00:19, 975.26 examples/s] 20139 examples [00:19, 993.77 examples/s]20245 examples [00:19, 1012.10 examples/s]20352 examples [00:19, 1026.20 examples/s]20458 examples [00:19, 1035.27 examples/s]20562 examples [00:20, 1034.46 examples/s]20666 examples [00:20, 1033.73 examples/s]20770 examples [00:20, 1033.62 examples/s]20874 examples [00:20, 1033.86 examples/s]20978 examples [00:20, 1032.41 examples/s]21082 examples [00:20, 1032.02 examples/s]21187 examples [00:20, 1036.46 examples/s]21293 examples [00:20, 1041.26 examples/s]21399 examples [00:20, 1043.76 examples/s]21504 examples [00:20, 1035.09 examples/s]21609 examples [00:21, 1038.51 examples/s]21713 examples [00:21, 1019.18 examples/s]21819 examples [00:21, 1029.97 examples/s]21926 examples [00:21, 1039.04 examples/s]22030 examples [00:21, 1031.30 examples/s]22134 examples [00:21, 1029.94 examples/s]22238 examples [00:21, 1028.06 examples/s]22341 examples [00:21, 1022.55 examples/s]22447 examples [00:21, 1032.65 examples/s]22552 examples [00:21, 1037.70 examples/s]22657 examples [00:22, 1039.85 examples/s]22762 examples [00:22, 1040.21 examples/s]22867 examples [00:22, 1032.39 examples/s]22973 examples [00:22, 1039.77 examples/s]23079 examples [00:22, 1043.49 examples/s]23184 examples [00:22, 1037.80 examples/s]23288 examples [00:22, 1030.79 examples/s]23393 examples [00:22, 1034.38 examples/s]23497 examples [00:22, 1029.19 examples/s]23600 examples [00:23, 1027.57 examples/s]23703 examples [00:23, 1010.30 examples/s]23805 examples [00:23, 993.33 examples/s] 23910 examples [00:23, 1008.26 examples/s]24011 examples [00:23, 1007.05 examples/s]24114 examples [00:23, 1011.63 examples/s]24218 examples [00:23, 1019.89 examples/s]24321 examples [00:23, 1022.32 examples/s]24424 examples [00:23, 1017.34 examples/s]24526 examples [00:23, 1017.35 examples/s]24628 examples [00:24, 1002.62 examples/s]24731 examples [00:24, 1010.36 examples/s]24835 examples [00:24, 1016.73 examples/s]24937 examples [00:24, 1001.50 examples/s]25038 examples [00:24, 998.40 examples/s] 25142 examples [00:24, 1008.02 examples/s]25243 examples [00:24, 1006.30 examples/s]25348 examples [00:24, 1018.61 examples/s]25453 examples [00:24, 1025.61 examples/s]25556 examples [00:24, 1002.64 examples/s]25658 examples [00:25, 1006.07 examples/s]25759 examples [00:25, 989.23 examples/s] 25863 examples [00:25, 1003.36 examples/s]25964 examples [00:25, 982.96 examples/s] 26063 examples [00:25, 976.39 examples/s]26167 examples [00:25, 993.54 examples/s]26267 examples [00:25, 994.42 examples/s]26371 examples [00:25, 1006.66 examples/s]26475 examples [00:25, 1013.86 examples/s]26578 examples [00:25, 1018.01 examples/s]26680 examples [00:26, 1017.70 examples/s]26783 examples [00:26, 1018.92 examples/s]26885 examples [00:26, 1015.36 examples/s]26990 examples [00:26, 1024.41 examples/s]27095 examples [00:26, 1029.91 examples/s]27201 examples [00:26, 1036.30 examples/s]27305 examples [00:26, 1035.84 examples/s]27409 examples [00:26, 1036.21 examples/s]27513 examples [00:26, 1034.69 examples/s]27619 examples [00:26, 1040.43 examples/s]27724 examples [00:27, 1041.77 examples/s]27829 examples [00:27, 1036.39 examples/s]27933 examples [00:27, 1034.28 examples/s]28037 examples [00:27, 1021.94 examples/s]28141 examples [00:27, 1026.89 examples/s]28246 examples [00:27, 1032.81 examples/s]28350 examples [00:27, 1024.17 examples/s]28453 examples [00:27, 1010.05 examples/s]28556 examples [00:27, 1015.44 examples/s]28660 examples [00:27, 1022.33 examples/s]28763 examples [00:28, 1019.75 examples/s]28869 examples [00:28, 1029.62 examples/s]28973 examples [00:28, 1021.64 examples/s]29077 examples [00:28, 1025.21 examples/s]29181 examples [00:28, 1029.57 examples/s]29284 examples [00:28, 1024.51 examples/s]29389 examples [00:28, 1029.12 examples/s]29492 examples [00:28, 1029.05 examples/s]29595 examples [00:28, 1025.42 examples/s]29701 examples [00:29, 1033.14 examples/s]29805 examples [00:29, 1029.93 examples/s]29909 examples [00:29, 1032.09 examples/s]30013 examples [00:29, 981.38 examples/s] 30112 examples [00:29, 972.67 examples/s]30216 examples [00:29, 991.81 examples/s]30321 examples [00:29, 1006.33 examples/s]30424 examples [00:29, 1011.10 examples/s]30526 examples [00:29, 1010.13 examples/s]30628 examples [00:29, 1007.09 examples/s]30734 examples [00:30, 1020.71 examples/s]30839 examples [00:30, 1027.23 examples/s]30944 examples [00:30, 1032.15 examples/s]31050 examples [00:30, 1040.04 examples/s]31155 examples [00:30, 972.29 examples/s] 31259 examples [00:30, 990.00 examples/s]31364 examples [00:30, 1007.25 examples/s]31467 examples [00:30, 1013.48 examples/s]31570 examples [00:30, 1016.42 examples/s]31672 examples [00:30, 1012.49 examples/s]31778 examples [00:31, 1025.00 examples/s]31884 examples [00:31, 1032.51 examples/s]31988 examples [00:31, 984.12 examples/s] 32092 examples [00:31, 997.96 examples/s]32193 examples [00:31, 997.70 examples/s]32299 examples [00:31, 1014.02 examples/s]32403 examples [00:31, 1019.40 examples/s]32507 examples [00:31, 1023.17 examples/s]32613 examples [00:31, 1030.31 examples/s]32717 examples [00:31, 1020.63 examples/s]32820 examples [00:32, 1018.02 examples/s]32926 examples [00:32, 1028.76 examples/s]33029 examples [00:32, 1012.58 examples/s]33131 examples [00:32, 1014.34 examples/s]33234 examples [00:32, 1017.54 examples/s]33336 examples [00:32, 1009.63 examples/s]33442 examples [00:32, 1021.87 examples/s]33546 examples [00:32, 1025.66 examples/s]33649 examples [00:32, 1020.68 examples/s]33752 examples [00:33, 1018.73 examples/s]33857 examples [00:33, 1027.74 examples/s]33962 examples [00:33, 1032.51 examples/s]34066 examples [00:33, 1024.54 examples/s]34171 examples [00:33, 1029.26 examples/s]34275 examples [00:33, 1031.92 examples/s]34380 examples [00:33, 1036.88 examples/s]34485 examples [00:33, 1039.18 examples/s]34589 examples [00:33, 1036.56 examples/s]34693 examples [00:33, 1030.83 examples/s]34797 examples [00:34, 1033.52 examples/s]34901 examples [00:34, 1027.30 examples/s]35008 examples [00:34, 1037.95 examples/s]35112 examples [00:34, 1035.36 examples/s]35216 examples [00:34, 1036.47 examples/s]35320 examples [00:34, 1034.47 examples/s]35424 examples [00:34, 1017.46 examples/s]35526 examples [00:34, 990.08 examples/s] 35627 examples [00:34, 994.46 examples/s]35728 examples [00:34, 997.09 examples/s]35833 examples [00:35, 1011.10 examples/s]35937 examples [00:35, 1016.59 examples/s]36039 examples [00:35, 1013.39 examples/s]36141 examples [00:35, 1009.17 examples/s]36245 examples [00:35, 1017.84 examples/s]36347 examples [00:35, 1015.83 examples/s]36452 examples [00:35, 1024.61 examples/s]36558 examples [00:35, 1034.44 examples/s]36664 examples [00:35, 1039.44 examples/s]36769 examples [00:35, 1041.10 examples/s]36874 examples [00:36, 1042.84 examples/s]36981 examples [00:36, 1048.72 examples/s]37087 examples [00:36, 1049.29 examples/s]37192 examples [00:36, 1047.98 examples/s]37297 examples [00:36, 1026.02 examples/s]37403 examples [00:36, 1033.93 examples/s]37507 examples [00:36, 1033.67 examples/s]37611 examples [00:36, 1034.42 examples/s]37716 examples [00:36, 1036.59 examples/s]37820 examples [00:36, 997.37 examples/s] 37925 examples [00:37, 1011.74 examples/s]38031 examples [00:37, 1024.11 examples/s]38136 examples [00:37, 1029.93 examples/s]38240 examples [00:37, 1025.83 examples/s]38343 examples [00:37, 996.06 examples/s] 38448 examples [00:37, 1010.64 examples/s]38551 examples [00:37, 1016.17 examples/s]38656 examples [00:37, 1025.61 examples/s]38759 examples [00:37, 1023.31 examples/s]38864 examples [00:37, 1030.91 examples/s]38969 examples [00:38, 1035.35 examples/s]39074 examples [00:38, 1039.45 examples/s]39178 examples [00:38, 1033.78 examples/s]39282 examples [00:38, 1030.31 examples/s]39387 examples [00:38, 1034.01 examples/s]39493 examples [00:38, 1039.72 examples/s]39599 examples [00:38, 1044.22 examples/s]39705 examples [00:38, 1046.32 examples/s]39810 examples [00:38, 1026.48 examples/s]39914 examples [00:39, 1029.67 examples/s]40018 examples [00:39, 980.83 examples/s] 40120 examples [00:39, 991.53 examples/s]40225 examples [00:39, 1008.31 examples/s]40327 examples [00:39, 1011.65 examples/s]40429 examples [00:39, 1011.22 examples/s]40532 examples [00:39, 1014.42 examples/s]40636 examples [00:39, 1020.12 examples/s]40739 examples [00:39, 1020.99 examples/s]40842 examples [00:39, 999.85 examples/s] 40945 examples [00:40, 1008.33 examples/s]41046 examples [00:40, 1000.95 examples/s]41152 examples [00:40, 1017.49 examples/s]41257 examples [00:40, 1024.83 examples/s]41361 examples [00:40, 1027.41 examples/s]41465 examples [00:40, 1030.33 examples/s]41570 examples [00:40, 1036.09 examples/s]41676 examples [00:40, 1042.83 examples/s]41782 examples [00:40, 1046.10 examples/s]41887 examples [00:40, 1024.77 examples/s]41990 examples [00:41, 1025.32 examples/s]42095 examples [00:41, 1030.70 examples/s]42199 examples [00:41, 1011.83 examples/s]42303 examples [00:41, 1017.85 examples/s]42405 examples [00:41, 997.96 examples/s] 42506 examples [00:41, 1000.75 examples/s]42611 examples [00:41, 1014.85 examples/s]42713 examples [00:41, 1016.30 examples/s]42815 examples [00:41, 999.69 examples/s] 42916 examples [00:41, 999.35 examples/s]43020 examples [00:42, 1010.29 examples/s]43122 examples [00:42, 975.81 examples/s] 43220 examples [00:42, 919.53 examples/s]43323 examples [00:42, 947.65 examples/s]43420 examples [00:42, 952.64 examples/s]43525 examples [00:42, 978.81 examples/s]43629 examples [00:42, 995.38 examples/s]43734 examples [00:42, 1010.40 examples/s]43837 examples [00:42, 1016.07 examples/s]43940 examples [00:43, 1017.94 examples/s]44042 examples [00:43, 1007.83 examples/s]44144 examples [00:43, 1009.84 examples/s]44250 examples [00:43, 1021.72 examples/s]44356 examples [00:43, 1030.47 examples/s]44460 examples [00:43, 1016.14 examples/s]44562 examples [00:43, 990.83 examples/s] 44667 examples [00:43, 1007.63 examples/s]44773 examples [00:43, 1021.01 examples/s]44877 examples [00:43, 1025.10 examples/s]44980 examples [00:44, 988.04 examples/s] 45084 examples [00:44, 1000.82 examples/s]45187 examples [00:44, 1006.58 examples/s]45293 examples [00:44, 1021.08 examples/s]45398 examples [00:44, 1027.29 examples/s]45501 examples [00:44, 1022.35 examples/s]45604 examples [00:44, 1022.49 examples/s]45708 examples [00:44, 1024.80 examples/s]45815 examples [00:44, 1037.55 examples/s]45919 examples [00:44, 1033.02 examples/s]46023 examples [00:45, 1031.16 examples/s]46127 examples [00:45, 1031.56 examples/s]46232 examples [00:45, 1035.15 examples/s]46338 examples [00:45, 1041.53 examples/s]46443 examples [00:45, 1040.91 examples/s]46548 examples [00:45, 1035.89 examples/s]46652 examples [00:45, 1033.36 examples/s]46757 examples [00:45, 1036.08 examples/s]46861 examples [00:45, 1031.94 examples/s]46965 examples [00:45, 1028.09 examples/s]47068 examples [00:46, 1028.53 examples/s]47173 examples [00:46, 1032.28 examples/s]47277 examples [00:46, 1034.26 examples/s]47382 examples [00:46, 1036.04 examples/s]47487 examples [00:46, 1038.29 examples/s]47591 examples [00:46, 1036.75 examples/s]47697 examples [00:46, 1041.58 examples/s]47802 examples [00:46, 1037.71 examples/s]47908 examples [00:46, 1042.36 examples/s]48013 examples [00:46, 1040.79 examples/s]48118 examples [00:47, 1037.48 examples/s]48224 examples [00:47, 1042.17 examples/s]48329 examples [00:47, 1040.59 examples/s]48434 examples [00:47, 1042.69 examples/s]48539 examples [00:47, 1039.56 examples/s]48643 examples [00:47, 1022.12 examples/s]48747 examples [00:47, 1025.64 examples/s]48851 examples [00:47, 1027.11 examples/s]48954 examples [00:47, 1027.67 examples/s]49059 examples [00:47, 1032.52 examples/s]49163 examples [00:48, 1032.75 examples/s]49267 examples [00:48, 1027.07 examples/s]49372 examples [00:48, 1032.16 examples/s]49478 examples [00:48, 1037.18 examples/s]49583 examples [00:48, 1040.61 examples/s]49688 examples [00:48, 1022.87 examples/s]49792 examples [00:48, 1027.91 examples/s]49896 examples [00:48, 1030.63 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6094/50000 [00:00<00:00, 60936.28 examples/s] 37%|███▋      | 18535/50000 [00:00<00:00, 71948.12 examples/s] 62%|██████▏   | 31169/50000 [00:00<00:00, 82617.34 examples/s] 88%|████████▊ | 43976/50000 [00:00<00:00, 92460.29 examples/s]                                                               0 examples [00:00, ? examples/s]82 examples [00:00, 812.60 examples/s]181 examples [00:00, 855.59 examples/s]287 examples [00:00, 907.50 examples/s]393 examples [00:00, 947.29 examples/s]500 examples [00:00, 980.09 examples/s]607 examples [00:00, 1004.60 examples/s]714 examples [00:00, 1021.08 examples/s]818 examples [00:00, 1025.72 examples/s]925 examples [00:00, 1038.41 examples/s]1029 examples [00:01, 1036.34 examples/s]1136 examples [00:01, 1043.69 examples/s]1239 examples [00:01, 1008.10 examples/s]1343 examples [00:01, 1016.61 examples/s]1449 examples [00:01, 1028.42 examples/s]1552 examples [00:01, 1017.03 examples/s]1654 examples [00:01, 1004.52 examples/s]1759 examples [00:01, 1015.38 examples/s]1865 examples [00:01, 1026.03 examples/s]1971 examples [00:01, 1035.17 examples/s]2078 examples [00:02, 1044.79 examples/s]2183 examples [00:02, 1035.75 examples/s]2287 examples [00:02, 1030.69 examples/s]2391 examples [00:02, 1028.05 examples/s]2494 examples [00:02, 1021.39 examples/s]2600 examples [00:02, 1030.17 examples/s]2706 examples [00:02, 1038.49 examples/s]2813 examples [00:02, 1045.80 examples/s]2919 examples [00:02, 1048.12 examples/s]3026 examples [00:02, 1053.69 examples/s]3132 examples [00:03, 1050.19 examples/s]3238 examples [00:03, 1051.18 examples/s]3344 examples [00:03, 1053.67 examples/s]3450 examples [00:03, 1043.20 examples/s]3556 examples [00:03, 1047.86 examples/s]3662 examples [00:03, 1049.02 examples/s]3769 examples [00:03, 1052.82 examples/s]3875 examples [00:03, 1051.23 examples/s]3981 examples [00:03, 1036.27 examples/s]4088 examples [00:03, 1044.31 examples/s]4193 examples [00:04, 1044.26 examples/s]4298 examples [00:04, 1044.88 examples/s]4403 examples [00:04, 1044.22 examples/s]4508 examples [00:04, 1036.67 examples/s]4615 examples [00:04, 1044.76 examples/s]4720 examples [00:04, 1043.00 examples/s]4825 examples [00:04, 835.12 examples/s] 4918 examples [00:04, 859.78 examples/s]5024 examples [00:04, 910.07 examples/s]5129 examples [00:05, 945.71 examples/s]5234 examples [00:05, 974.47 examples/s]5334 examples [00:05, 974.20 examples/s]5437 examples [00:05, 989.89 examples/s]5538 examples [00:05, 984.55 examples/s]5644 examples [00:05, 1005.50 examples/s]5749 examples [00:05, 1017.59 examples/s]5854 examples [00:05, 1025.51 examples/s]5957 examples [00:05, 1016.10 examples/s]6059 examples [00:05, 1015.48 examples/s]6165 examples [00:06, 1025.98 examples/s]6268 examples [00:06, 1022.86 examples/s]6371 examples [00:06, 1018.05 examples/s]6474 examples [00:06, 1019.16 examples/s]6580 examples [00:06, 1030.56 examples/s]6684 examples [00:06, 1032.08 examples/s]6788 examples [00:06, 1033.81 examples/s]6894 examples [00:06, 1040.73 examples/s]6999 examples [00:06, 1033.06 examples/s]7104 examples [00:06, 1037.37 examples/s]7209 examples [00:07, 1039.98 examples/s]7315 examples [00:07, 1042.11 examples/s]7421 examples [00:07, 1045.01 examples/s]7526 examples [00:07, 1038.77 examples/s]7632 examples [00:07, 1044.99 examples/s]7737 examples [00:07, 1043.18 examples/s]7842 examples [00:07, 1044.13 examples/s]7947 examples [00:07, 1043.99 examples/s]8052 examples [00:07, 1039.88 examples/s]8156 examples [00:07, 1038.07 examples/s]8262 examples [00:08, 1042.85 examples/s]8367 examples [00:08, 1031.89 examples/s]8471 examples [00:08, 1017.47 examples/s]8574 examples [00:08, 1020.75 examples/s]8680 examples [00:08, 1031.64 examples/s]8784 examples [00:08, 1024.21 examples/s]8887 examples [00:08, 1020.79 examples/s]8993 examples [00:08, 1031.47 examples/s]9097 examples [00:08, 1032.83 examples/s]9201 examples [00:08, 1030.23 examples/s]9305 examples [00:09, 1030.72 examples/s]9409 examples [00:09, 1027.46 examples/s]9512 examples [00:09, 1002.92 examples/s]9616 examples [00:09, 1011.95 examples/s]9722 examples [00:09, 1024.31 examples/s]9825 examples [00:09, 1023.44 examples/s]9929 examples [00:09, 1026.06 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteDZDVMK/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteDZDVMK/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa27e2d40d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa27e2d40d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa27e2d40d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa2c9cb3828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa2045656a0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa27e2d40d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa27e2d40d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa27e2d40d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa27c4a90b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa27c4a9128>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.42 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.76 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.17 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.17 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.25 file/s]2020-08-20 06:09:01.989145: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-20 06:09:01.993364: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-08-20 06:09:01.993545: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557389917080 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-20 06:09:01.993561: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist2', 'train', 'test', 'cifar10', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 480498.66it/s] 92%|█████████▏| 9166848/9912422 [00:00<00:01, 684878.66it/s]9920512it [00:00, 47762833.47it/s]                           
0it [00:00, ?it/s]32768it [00:00, 656857.79it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1003474.49it/s]1654784it [00:00, 12625776.57it/s]                           
0it [00:00, ?it/s]8192it [00:00, 192720.42it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
