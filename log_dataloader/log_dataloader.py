
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'ff21906d63b46d89670bd3ed148b01355ca5ff0e', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/ff21906d63b46d89670bd3ed148b01355ca5ff0e

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7496a35730> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7496a35730> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f750175e488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f750175e488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f752097de18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f752097de18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f74aea95510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f74aea95510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f74aea95510> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 142291.67it/s] 33%|███▎      | 3309568/9912422 [00:00<00:32, 202897.36it/s]9920512it [00:00, 32000254.68it/s]                           
0it [00:00, ?it/s]32768it [00:00, 572858.03it/s]
0it [00:00, ?it/s]  2%|▏         | 40960/1648877 [00:00<00:03, 409531.07it/s]1654784it [00:00, 11088638.51it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191257.20it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f749696bf60>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7495cb2ba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f74aea951e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f74aea951e0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f74aea951e0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:39,  1.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:39,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:38,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:38,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:37,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:37,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:36,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:07,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:07,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:06,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:06,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:05,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:05,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:04,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:45,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:45,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:44,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:44,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:44,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:43,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:43,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:29,  4.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:29,  4.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:29,  4.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:20,  6.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:19,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:14,  8.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:14,  8.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:14,  8.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:13,  8.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:13,  8.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:13,  8.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:13,  8.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:13,  8.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 11.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 11.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 11.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 15.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 15.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 32.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 32.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 32.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 32.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 45.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 45.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 45.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 50.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 50.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 50.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 50.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 50.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 50.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 50.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 50.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 50.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 55.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 68.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 68.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 69.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.85s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.85s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.85s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.10s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.85s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 69.76 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.10s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.10s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.10s/ url]
0 examples [00:00, ? examples/s]2020-09-05 06:07:25.727938: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-05 06:07:25.741671: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-09-05 06:07:25.741844: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564b55d8f540 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-05 06:07:25.741863: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
35 examples [00:00, 347.67 examples/s]130 examples [00:00, 429.29 examples/s]228 examples [00:00, 515.57 examples/s]326 examples [00:00, 601.01 examples/s]414 examples [00:00, 663.35 examples/s]509 examples [00:00, 727.59 examples/s]607 examples [00:00, 788.41 examples/s]705 examples [00:00, 836.77 examples/s]799 examples [00:00, 865.09 examples/s]890 examples [00:01, 875.87 examples/s]981 examples [00:01, 861.41 examples/s]1073 examples [00:01, 877.19 examples/s]1170 examples [00:01, 901.66 examples/s]1266 examples [00:01, 917.77 examples/s]1364 examples [00:01, 933.22 examples/s]1459 examples [00:01, 937.00 examples/s]1562 examples [00:01, 960.61 examples/s]1661 examples [00:01, 969.12 examples/s]1762 examples [00:01, 979.17 examples/s]1861 examples [00:02, 979.51 examples/s]1965 examples [00:02, 995.78 examples/s]2066 examples [00:02, 997.72 examples/s]2166 examples [00:02, 985.13 examples/s]2268 examples [00:02, 994.30 examples/s]2368 examples [00:02, 995.83 examples/s]2474 examples [00:02, 1011.56 examples/s]2576 examples [00:02, 995.54 examples/s] 2676 examples [00:02, 985.11 examples/s]2776 examples [00:02, 988.69 examples/s]2877 examples [00:03, 994.92 examples/s]2982 examples [00:03, 1007.67 examples/s]3083 examples [00:03, 990.43 examples/s] 3187 examples [00:03, 1004.06 examples/s]3288 examples [00:03, 968.71 examples/s] 3393 examples [00:03, 991.36 examples/s]3495 examples [00:03, 999.00 examples/s]3601 examples [00:03, 1014.87 examples/s]3703 examples [00:03, 967.89 examples/s] 3802 examples [00:03, 971.37 examples/s]3907 examples [00:04, 991.38 examples/s]4007 examples [00:04, 985.81 examples/s]4109 examples [00:04, 994.27 examples/s]4209 examples [00:04, 971.83 examples/s]4311 examples [00:04, 985.02 examples/s]4413 examples [00:04, 995.11 examples/s]4515 examples [00:04, 1000.95 examples/s]4618 examples [00:04, 1008.92 examples/s]4721 examples [00:04, 1012.19 examples/s]4823 examples [00:04, 1008.52 examples/s]4924 examples [00:05, 1002.09 examples/s]5025 examples [00:05, 999.50 examples/s] 5127 examples [00:05, 1005.50 examples/s]5228 examples [00:05, 994.85 examples/s] 5328 examples [00:05, 988.86 examples/s]5428 examples [00:05, 992.09 examples/s]5528 examples [00:05, 981.46 examples/s]5627 examples [00:05, 934.90 examples/s]5723 examples [00:05, 941.53 examples/s]5821 examples [00:06, 950.31 examples/s]5923 examples [00:06, 969.48 examples/s]6021 examples [00:06, 963.63 examples/s]6122 examples [00:06, 975.28 examples/s]6220 examples [00:06, 961.79 examples/s]6317 examples [00:06, 955.08 examples/s]6413 examples [00:06, 932.27 examples/s]6507 examples [00:06, 930.47 examples/s]6606 examples [00:06, 945.83 examples/s]6701 examples [00:06, 908.76 examples/s]6800 examples [00:07, 929.31 examples/s]6902 examples [00:07, 953.33 examples/s]6998 examples [00:07, 940.05 examples/s]7093 examples [00:07, 941.70 examples/s]7190 examples [00:07, 948.39 examples/s]7286 examples [00:07, 930.69 examples/s]7384 examples [00:07, 944.20 examples/s]7479 examples [00:07, 902.27 examples/s]7570 examples [00:07, 873.62 examples/s]7658 examples [00:08, 869.46 examples/s]7748 examples [00:08, 875.63 examples/s]7837 examples [00:08, 878.57 examples/s]7926 examples [00:08, 868.89 examples/s]8019 examples [00:08, 885.76 examples/s]8108 examples [00:08, 844.11 examples/s]8200 examples [00:08, 863.46 examples/s]8287 examples [00:08, 858.11 examples/s]8374 examples [00:08, 843.16 examples/s]8465 examples [00:08, 861.67 examples/s]8555 examples [00:09, 870.64 examples/s]8651 examples [00:09, 895.47 examples/s]8745 examples [00:09, 907.56 examples/s]8837 examples [00:09, 894.60 examples/s]8930 examples [00:09, 904.22 examples/s]9024 examples [00:09, 913.63 examples/s]9121 examples [00:09, 928.94 examples/s]9223 examples [00:09, 953.13 examples/s]9319 examples [00:09, 932.65 examples/s]9418 examples [00:09, 946.75 examples/s]9521 examples [00:10, 968.55 examples/s]9623 examples [00:10, 983.41 examples/s]9722 examples [00:10, 975.24 examples/s]9820 examples [00:10, 957.38 examples/s]9917 examples [00:10, 959.72 examples/s]10014 examples [00:10, 896.09 examples/s]10116 examples [00:10, 926.65 examples/s]10210 examples [00:10, 853.01 examples/s]10306 examples [00:10, 877.23 examples/s]10407 examples [00:11, 912.06 examples/s]10503 examples [00:11, 925.14 examples/s]10597 examples [00:11, 918.11 examples/s]10690 examples [00:11, 892.28 examples/s]10797 examples [00:11, 936.72 examples/s]10896 examples [00:11, 950.41 examples/s]10996 examples [00:11, 964.06 examples/s]11095 examples [00:11, 971.21 examples/s]11193 examples [00:11, 966.65 examples/s]11290 examples [00:11, 960.10 examples/s]11387 examples [00:12, 961.20 examples/s]11484 examples [00:12, 951.56 examples/s]11581 examples [00:12, 955.57 examples/s]11677 examples [00:12, 908.31 examples/s]11770 examples [00:12, 914.01 examples/s]11862 examples [00:12, 887.29 examples/s]11960 examples [00:12, 911.94 examples/s]12052 examples [00:12, 889.89 examples/s]12142 examples [00:12, 870.54 examples/s]12238 examples [00:13, 894.42 examples/s]12331 examples [00:13, 902.20 examples/s]12422 examples [00:13, 892.95 examples/s]12512 examples [00:13, 880.30 examples/s]12601 examples [00:13, 875.07 examples/s]12693 examples [00:13, 886.47 examples/s]12785 examples [00:13, 896.14 examples/s]12892 examples [00:13, 941.95 examples/s]12989 examples [00:13, 949.42 examples/s]13085 examples [00:13, 944.90 examples/s]13184 examples [00:14, 955.62 examples/s]13280 examples [00:14, 944.54 examples/s]13375 examples [00:14, 858.84 examples/s]13463 examples [00:14, 857.07 examples/s]13550 examples [00:14, 847.42 examples/s]13645 examples [00:14, 874.82 examples/s]13741 examples [00:14, 895.90 examples/s]13838 examples [00:14, 914.49 examples/s]13931 examples [00:14, 917.59 examples/s]14027 examples [00:14, 929.69 examples/s]14122 examples [00:15, 935.04 examples/s]14216 examples [00:15, 904.76 examples/s]14315 examples [00:15, 926.58 examples/s]14409 examples [00:15, 924.37 examples/s]14502 examples [00:15, 884.09 examples/s]14596 examples [00:15, 898.70 examples/s]14687 examples [00:15, 859.05 examples/s]14774 examples [00:15, 831.92 examples/s]14863 examples [00:15, 848.25 examples/s]14959 examples [00:16, 876.25 examples/s]15053 examples [00:16, 893.03 examples/s]15149 examples [00:16, 909.56 examples/s]15241 examples [00:16, 890.21 examples/s]15334 examples [00:16, 899.12 examples/s]15428 examples [00:16, 910.13 examples/s]15520 examples [00:16, 894.83 examples/s]15613 examples [00:16, 904.25 examples/s]15704 examples [00:16, 904.57 examples/s]15797 examples [00:16, 909.45 examples/s]15890 examples [00:17, 914.14 examples/s]15984 examples [00:17, 919.44 examples/s]16077 examples [00:17, 871.82 examples/s]16171 examples [00:17, 890.31 examples/s]16261 examples [00:17, 869.89 examples/s]16349 examples [00:17, 821.44 examples/s]16447 examples [00:17, 861.71 examples/s]16546 examples [00:17, 894.50 examples/s]16644 examples [00:17, 917.59 examples/s]16738 examples [00:18, 922.03 examples/s]16834 examples [00:18, 930.56 examples/s]16930 examples [00:18, 937.96 examples/s]17028 examples [00:18, 947.79 examples/s]17124 examples [00:18, 917.04 examples/s]17217 examples [00:18, 915.76 examples/s]17309 examples [00:18, 907.36 examples/s]17400 examples [00:18, 904.21 examples/s]17491 examples [00:18, 894.05 examples/s]17581 examples [00:18, 885.13 examples/s]17679 examples [00:19, 908.85 examples/s]17771 examples [00:19, 893.25 examples/s]17861 examples [00:19, 846.70 examples/s]17947 examples [00:19, 836.24 examples/s]18032 examples [00:19, 831.23 examples/s]18125 examples [00:19, 858.07 examples/s]18214 examples [00:19, 865.72 examples/s]18305 examples [00:19, 877.28 examples/s]18394 examples [00:19, 866.45 examples/s]18484 examples [00:19, 874.35 examples/s]18582 examples [00:20, 901.78 examples/s]18680 examples [00:20, 922.51 examples/s]18773 examples [00:20, 914.34 examples/s]18870 examples [00:20, 928.60 examples/s]18968 examples [00:20, 941.98 examples/s]19063 examples [00:20, 940.04 examples/s]19160 examples [00:20, 946.27 examples/s]19255 examples [00:20, 941.26 examples/s]19353 examples [00:20, 950.94 examples/s]19449 examples [00:21, 933.19 examples/s]19549 examples [00:21, 951.91 examples/s]19645 examples [00:21, 952.84 examples/s]19742 examples [00:21, 957.82 examples/s]19839 examples [00:21, 960.83 examples/s]19936 examples [00:21, 958.91 examples/s]20032 examples [00:21, 895.92 examples/s]20127 examples [00:21, 909.55 examples/s]20220 examples [00:21, 914.19 examples/s]20312 examples [00:21, 902.90 examples/s]20406 examples [00:22, 912.39 examples/s]20506 examples [00:22, 935.67 examples/s]20606 examples [00:22, 953.25 examples/s]20702 examples [00:22, 909.94 examples/s]20794 examples [00:22, 902.69 examples/s]20885 examples [00:22, 887.84 examples/s]20975 examples [00:22, 887.59 examples/s]21065 examples [00:22, 873.49 examples/s]21172 examples [00:22, 923.66 examples/s]21276 examples [00:22, 953.36 examples/s]21378 examples [00:23, 971.25 examples/s]21483 examples [00:23, 983.30 examples/s]21584 examples [00:23, 989.31 examples/s]21684 examples [00:23, 975.42 examples/s]21782 examples [00:23, 948.14 examples/s]21882 examples [00:23, 960.59 examples/s]21985 examples [00:23, 979.52 examples/s]22089 examples [00:23, 996.28 examples/s]22192 examples [00:23, 1005.11 examples/s]22293 examples [00:24, 998.81 examples/s] 22397 examples [00:24, 1008.63 examples/s]22500 examples [00:24, 1012.26 examples/s]22604 examples [00:24, 1019.62 examples/s]22707 examples [00:24, 986.88 examples/s] 22806 examples [00:24, 937.17 examples/s]22901 examples [00:24, 921.40 examples/s]22999 examples [00:24, 935.40 examples/s]23093 examples [00:24, 934.46 examples/s]23192 examples [00:24, 949.40 examples/s]23288 examples [00:25, 952.50 examples/s]23384 examples [00:25, 952.09 examples/s]23483 examples [00:25, 962.42 examples/s]23580 examples [00:25, 963.68 examples/s]23677 examples [00:25, 934.23 examples/s]23771 examples [00:25, 906.99 examples/s]23866 examples [00:25, 917.45 examples/s]23960 examples [00:25, 921.47 examples/s]24053 examples [00:25, 916.69 examples/s]24151 examples [00:25, 932.46 examples/s]24247 examples [00:26, 938.79 examples/s]24342 examples [00:26, 939.44 examples/s]24437 examples [00:26, 922.31 examples/s]24530 examples [00:26, 921.24 examples/s]24626 examples [00:26, 930.09 examples/s]24722 examples [00:26, 936.49 examples/s]24816 examples [00:26, 928.99 examples/s]24909 examples [00:26, 927.61 examples/s]25002 examples [00:26, 917.32 examples/s]25094 examples [00:26, 903.00 examples/s]25185 examples [00:27, 890.69 examples/s]25280 examples [00:27, 905.70 examples/s]25372 examples [00:27, 898.19 examples/s]25462 examples [00:27, 890.08 examples/s]25554 examples [00:27, 898.74 examples/s]25646 examples [00:27, 904.30 examples/s]25737 examples [00:27, 892.83 examples/s]25827 examples [00:27, 884.37 examples/s]25916 examples [00:27, 885.72 examples/s]26005 examples [00:28, 842.87 examples/s]26096 examples [00:28, 861.30 examples/s]26183 examples [00:28, 839.04 examples/s]26268 examples [00:28, 829.52 examples/s]26352 examples [00:28, 777.88 examples/s]26445 examples [00:28, 817.84 examples/s]26528 examples [00:28, 811.44 examples/s]26617 examples [00:28, 832.39 examples/s]26701 examples [00:28, 829.12 examples/s]26785 examples [00:28, 824.36 examples/s]26868 examples [00:29, 808.09 examples/s]26950 examples [00:29, 805.97 examples/s]27037 examples [00:29, 823.49 examples/s]27128 examples [00:29, 847.04 examples/s]27214 examples [00:29, 799.38 examples/s]27306 examples [00:29, 831.29 examples/s]27392 examples [00:29, 838.27 examples/s]27479 examples [00:29, 845.56 examples/s]27565 examples [00:29, 836.13 examples/s]27660 examples [00:30, 867.21 examples/s]27753 examples [00:30, 885.01 examples/s]27842 examples [00:30, 878.90 examples/s]27939 examples [00:30, 901.70 examples/s]28036 examples [00:30, 918.75 examples/s]28132 examples [00:30, 930.18 examples/s]28226 examples [00:30, 932.92 examples/s]28320 examples [00:30, 918.97 examples/s]28414 examples [00:30, 924.16 examples/s]28511 examples [00:30, 927.45 examples/s]28604 examples [00:31, 887.03 examples/s]28698 examples [00:31, 901.96 examples/s]28793 examples [00:31, 915.39 examples/s]28895 examples [00:31, 944.28 examples/s]28995 examples [00:31, 960.05 examples/s]29092 examples [00:31, 948.89 examples/s]29188 examples [00:31, 933.23 examples/s]29282 examples [00:31, 904.87 examples/s]29386 examples [00:31, 940.93 examples/s]29481 examples [00:31, 938.87 examples/s]29584 examples [00:32, 962.47 examples/s]29681 examples [00:32, 959.13 examples/s]29778 examples [00:32, 946.59 examples/s]29878 examples [00:32, 961.63 examples/s]29979 examples [00:32, 975.40 examples/s]30077 examples [00:32, 914.66 examples/s]30170 examples [00:32, 910.08 examples/s]30268 examples [00:32, 927.39 examples/s]30362 examples [00:32, 929.62 examples/s]30456 examples [00:33, 908.73 examples/s]30551 examples [00:33, 919.41 examples/s]30644 examples [00:33, 905.32 examples/s]30739 examples [00:33, 916.01 examples/s]30831 examples [00:33, 771.93 examples/s]30919 examples [00:33, 800.81 examples/s]31003 examples [00:33, 789.48 examples/s]31088 examples [00:33, 805.72 examples/s]31187 examples [00:33, 851.29 examples/s]31287 examples [00:34, 888.25 examples/s]31390 examples [00:34, 925.20 examples/s]31488 examples [00:34, 938.36 examples/s]31583 examples [00:34, 931.64 examples/s]31682 examples [00:34, 934.27 examples/s]31780 examples [00:34, 946.39 examples/s]31876 examples [00:34, 945.44 examples/s]31977 examples [00:34, 960.90 examples/s]32074 examples [00:34, 941.16 examples/s]32171 examples [00:34, 947.95 examples/s]32273 examples [00:35, 967.27 examples/s]32374 examples [00:35, 978.64 examples/s]32473 examples [00:35, 981.24 examples/s]32572 examples [00:35, 975.67 examples/s]32670 examples [00:35, 974.54 examples/s]32768 examples [00:35, 973.44 examples/s]32866 examples [00:35, 954.77 examples/s]32962 examples [00:35, 933.43 examples/s]33059 examples [00:35, 943.63 examples/s]33155 examples [00:35, 945.60 examples/s]33250 examples [00:36, 916.56 examples/s]33348 examples [00:36, 934.15 examples/s]33442 examples [00:36, 891.15 examples/s]33532 examples [00:36, 862.15 examples/s]33626 examples [00:36, 884.03 examples/s]33720 examples [00:36, 899.42 examples/s]33817 examples [00:36, 918.77 examples/s]33912 examples [00:36, 927.56 examples/s]34007 examples [00:36, 931.51 examples/s]34101 examples [00:37, 929.24 examples/s]34202 examples [00:37, 951.60 examples/s]34298 examples [00:37, 951.69 examples/s]34397 examples [00:37, 962.69 examples/s]34494 examples [00:37, 942.50 examples/s]34589 examples [00:37, 927.01 examples/s]34695 examples [00:37, 961.43 examples/s]34793 examples [00:37, 965.60 examples/s]34890 examples [00:37, 927.55 examples/s]34984 examples [00:37, 910.69 examples/s]35086 examples [00:38, 940.59 examples/s]35181 examples [00:38, 940.37 examples/s]35281 examples [00:38, 955.59 examples/s]35377 examples [00:38, 930.16 examples/s]35471 examples [00:38, 919.40 examples/s]35564 examples [00:38, 882.96 examples/s]35659 examples [00:38, 901.02 examples/s]35750 examples [00:38, 896.33 examples/s]35840 examples [00:38, 895.17 examples/s]35932 examples [00:38, 900.60 examples/s]36023 examples [00:39, 892.07 examples/s]36113 examples [00:39, 884.68 examples/s]36208 examples [00:39, 902.61 examples/s]36299 examples [00:39, 866.20 examples/s]36388 examples [00:39, 871.65 examples/s]36477 examples [00:39, 876.19 examples/s]36572 examples [00:39, 895.40 examples/s]36665 examples [00:39, 903.74 examples/s]36759 examples [00:39, 912.60 examples/s]36851 examples [00:40, 903.59 examples/s]36943 examples [00:40, 907.77 examples/s]37034 examples [00:40, 884.17 examples/s]37123 examples [00:40, 860.88 examples/s]37210 examples [00:40, 847.40 examples/s]37300 examples [00:40, 861.07 examples/s]37394 examples [00:40, 882.56 examples/s]37492 examples [00:40, 909.04 examples/s]37588 examples [00:40, 923.63 examples/s]37681 examples [00:40, 917.02 examples/s]37773 examples [00:41, 905.14 examples/s]37864 examples [00:41, 836.02 examples/s]37961 examples [00:41, 870.69 examples/s]38057 examples [00:41, 895.42 examples/s]38148 examples [00:41, 874.26 examples/s]38244 examples [00:41, 897.33 examples/s]38335 examples [00:41, 900.62 examples/s]38433 examples [00:41, 920.94 examples/s]38526 examples [00:41, 903.56 examples/s]38617 examples [00:41, 892.88 examples/s]38712 examples [00:42, 907.22 examples/s]38811 examples [00:42, 930.41 examples/s]38912 examples [00:42, 950.40 examples/s]39010 examples [00:42, 956.42 examples/s]39106 examples [00:42, 940.64 examples/s]39203 examples [00:42, 948.33 examples/s]39299 examples [00:42, 941.77 examples/s]39399 examples [00:42, 958.04 examples/s]39495 examples [00:42, 956.92 examples/s]39591 examples [00:43, 934.23 examples/s]39693 examples [00:43, 955.66 examples/s]39789 examples [00:43, 937.54 examples/s]39886 examples [00:43, 946.42 examples/s]39984 examples [00:43, 953.07 examples/s]40080 examples [00:43, 879.39 examples/s]40170 examples [00:43, 883.15 examples/s]40272 examples [00:43, 917.35 examples/s]40373 examples [00:43, 940.89 examples/s]40478 examples [00:43, 969.83 examples/s]40587 examples [00:44, 1002.81 examples/s]40693 examples [00:44, 1017.04 examples/s]40796 examples [00:44, 1003.92 examples/s]40905 examples [00:44, 1027.01 examples/s]41009 examples [00:44, 1022.71 examples/s]41112 examples [00:44, 1023.26 examples/s]41221 examples [00:44, 1041.29 examples/s]41327 examples [00:44, 1044.58 examples/s]41432 examples [00:44, 1035.64 examples/s]41536 examples [00:44, 1010.25 examples/s]41638 examples [00:45, 1000.20 examples/s]41739 examples [00:45, 969.76 examples/s] 41837 examples [00:45, 957.51 examples/s]41944 examples [00:45, 988.10 examples/s]42044 examples [00:45, 972.79 examples/s]42142 examples [00:45, 968.56 examples/s]42240 examples [00:45, 968.87 examples/s]42340 examples [00:45, 977.19 examples/s]42438 examples [00:45, 945.27 examples/s]42533 examples [00:46, 927.41 examples/s]42636 examples [00:46, 955.01 examples/s]42736 examples [00:46, 965.34 examples/s]42836 examples [00:46, 974.93 examples/s]42937 examples [00:46, 984.54 examples/s]43036 examples [00:46, 961.04 examples/s]43133 examples [00:46, 948.19 examples/s]43229 examples [00:46, 944.36 examples/s]43329 examples [00:46, 958.16 examples/s]43425 examples [00:46, 957.82 examples/s]43521 examples [00:47, 941.77 examples/s]43620 examples [00:47, 954.61 examples/s]43716 examples [00:47, 953.34 examples/s]43812 examples [00:47, 955.31 examples/s]43908 examples [00:47, 943.00 examples/s]44003 examples [00:47, 933.74 examples/s]44103 examples [00:47, 950.27 examples/s]44200 examples [00:47, 953.92 examples/s]44296 examples [00:47, 930.81 examples/s]44396 examples [00:47, 948.06 examples/s]44492 examples [00:48, 910.44 examples/s]44593 examples [00:48, 936.31 examples/s]44693 examples [00:48, 953.47 examples/s]44790 examples [00:48, 956.60 examples/s]44895 examples [00:48, 980.58 examples/s]44995 examples [00:48, 986.18 examples/s]45096 examples [00:48, 992.01 examples/s]45196 examples [00:48, 990.69 examples/s]45296 examples [00:48, 984.35 examples/s]45395 examples [00:49, 976.20 examples/s]45493 examples [00:49, 973.63 examples/s]45591 examples [00:49, 942.66 examples/s]45687 examples [00:49, 946.66 examples/s]45788 examples [00:49, 964.06 examples/s]45887 examples [00:49, 969.31 examples/s]45985 examples [00:49, 956.01 examples/s]46085 examples [00:49, 966.78 examples/s]46184 examples [00:49, 973.45 examples/s]46286 examples [00:49, 985.38 examples/s]46385 examples [00:50, 986.14 examples/s]46484 examples [00:50, 964.94 examples/s]46581 examples [00:50, 964.49 examples/s]46678 examples [00:50, 945.17 examples/s]46773 examples [00:50, 926.53 examples/s]46866 examples [00:50, 923.32 examples/s]46959 examples [00:50, 922.44 examples/s]47052 examples [00:50, 884.55 examples/s]47144 examples [00:50, 894.85 examples/s]47237 examples [00:50, 903.89 examples/s]47331 examples [00:51, 914.32 examples/s]47424 examples [00:51, 918.69 examples/s]47517 examples [00:51, 886.52 examples/s]47616 examples [00:51, 913.04 examples/s]47714 examples [00:51, 930.45 examples/s]47814 examples [00:51, 947.73 examples/s]47910 examples [00:51, 948.88 examples/s]48009 examples [00:51, 960.81 examples/s]48107 examples [00:51, 965.69 examples/s]48205 examples [00:51, 967.35 examples/s]48302 examples [00:52, 964.62 examples/s]48399 examples [00:52, 945.66 examples/s]48494 examples [00:52, 939.50 examples/s]48589 examples [00:52, 930.46 examples/s]48688 examples [00:52, 947.55 examples/s]48788 examples [00:52, 961.06 examples/s]48886 examples [00:52, 965.31 examples/s]48987 examples [00:52, 975.65 examples/s]49086 examples [00:52, 977.46 examples/s]49185 examples [00:53, 980.11 examples/s]49284 examples [00:53, 977.24 examples/s]49382 examples [00:53, 929.18 examples/s]49476 examples [00:53, 925.69 examples/s]49569 examples [00:53, 912.16 examples/s]49665 examples [00:53, 924.83 examples/s]49761 examples [00:53, 934.41 examples/s]49856 examples [00:53, 938.01 examples/s]49950 examples [00:53, 935.08 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7044/50000 [00:00<00:00, 70435.20 examples/s] 39%|███▊      | 19251/50000 [00:00<00:00, 80672.11 examples/s] 59%|█████▉    | 29637/50000 [00:00<00:00, 86462.37 examples/s] 84%|████████▍ | 42220/50000 [00:00<00:00, 95418.05 examples/s]                                                               0 examples [00:00, ? examples/s]71 examples [00:00, 708.79 examples/s]171 examples [00:00, 775.38 examples/s]266 examples [00:00, 817.85 examples/s]370 examples [00:00, 871.55 examples/s]468 examples [00:00, 899.01 examples/s]571 examples [00:00, 933.95 examples/s]675 examples [00:00, 963.21 examples/s]771 examples [00:00, 961.57 examples/s]865 examples [00:00, 928.93 examples/s]956 examples [00:01, 914.13 examples/s]1052 examples [00:01, 925.60 examples/s]1144 examples [00:01, 918.15 examples/s]1239 examples [00:01, 925.21 examples/s]1341 examples [00:01, 949.22 examples/s]1446 examples [00:01, 975.25 examples/s]1550 examples [00:01, 991.81 examples/s]1650 examples [00:01, 992.98 examples/s]1750 examples [00:01, 987.21 examples/s]1850 examples [00:01, 989.56 examples/s]1950 examples [00:02, 936.96 examples/s]2047 examples [00:02, 946.25 examples/s]2146 examples [00:02, 958.62 examples/s]2243 examples [00:02, 959.21 examples/s]2342 examples [00:02, 968.22 examples/s]2440 examples [00:02, 970.18 examples/s]2538 examples [00:02, 918.24 examples/s]2641 examples [00:02, 947.71 examples/s]2737 examples [00:02, 948.61 examples/s]2838 examples [00:02, 964.42 examples/s]2937 examples [00:03, 969.94 examples/s]3036 examples [00:03, 974.41 examples/s]3134 examples [00:03, 932.97 examples/s]3235 examples [00:03, 953.32 examples/s]3340 examples [00:03, 980.28 examples/s]3446 examples [00:03, 1002.21 examples/s]3554 examples [00:03, 1022.69 examples/s]3659 examples [00:03, 1028.27 examples/s]3763 examples [00:03, 1027.29 examples/s]3866 examples [00:03, 1027.61 examples/s]3969 examples [00:04, 1024.83 examples/s]4072 examples [00:04, 1009.48 examples/s]4177 examples [00:04, 1019.01 examples/s]4280 examples [00:04, 1019.25 examples/s]4383 examples [00:04, 1013.63 examples/s]4489 examples [00:04, 1024.97 examples/s]4595 examples [00:04, 1033.28 examples/s]4699 examples [00:04, 1034.77 examples/s]4803 examples [00:04, 831.58 examples/s] 4903 examples [00:05, 874.36 examples/s]5008 examples [00:05, 919.66 examples/s]5117 examples [00:05, 962.88 examples/s]5223 examples [00:05, 989.99 examples/s]5331 examples [00:05, 1012.65 examples/s]5435 examples [00:05, 990.33 examples/s] 5536 examples [00:05, 987.72 examples/s]5636 examples [00:05, 960.33 examples/s]5739 examples [00:05, 979.33 examples/s]5840 examples [00:06, 986.53 examples/s]5940 examples [00:06, 899.63 examples/s]6032 examples [00:06, 901.01 examples/s]6127 examples [00:06, 911.56 examples/s]6228 examples [00:06, 938.41 examples/s]6332 examples [00:06, 966.29 examples/s]6430 examples [00:06, 959.61 examples/s]6527 examples [00:06, 936.40 examples/s]6628 examples [00:06, 955.79 examples/s]6729 examples [00:06, 968.58 examples/s]6832 examples [00:07, 984.99 examples/s]6938 examples [00:07, 1005.28 examples/s]7039 examples [00:07, 987.82 examples/s] 7139 examples [00:07, 981.95 examples/s]7238 examples [00:07, 979.37 examples/s]7346 examples [00:07, 1007.41 examples/s]7448 examples [00:07, 1000.49 examples/s]7549 examples [00:07, 992.81 examples/s] 7649 examples [00:07, 984.05 examples/s]7750 examples [00:07, 991.11 examples/s]7851 examples [00:08, 996.40 examples/s]7951 examples [00:08, 994.89 examples/s]8051 examples [00:08, 996.11 examples/s]8151 examples [00:08, 969.60 examples/s]8251 examples [00:08, 976.19 examples/s]8351 examples [00:08, 980.41 examples/s]8450 examples [00:08, 981.82 examples/s]8550 examples [00:08, 985.76 examples/s]8649 examples [00:08, 974.54 examples/s]8747 examples [00:09, 956.15 examples/s]8843 examples [00:09, 953.61 examples/s]8940 examples [00:09, 957.40 examples/s]9038 examples [00:09, 963.67 examples/s]9135 examples [00:09, 952.05 examples/s]9231 examples [00:09, 836.01 examples/s]9318 examples [00:09, 776.97 examples/s]9416 examples [00:09, 827.08 examples/s]9508 examples [00:09, 850.57 examples/s]9610 examples [00:10, 893.20 examples/s]9714 examples [00:10, 931.24 examples/s]9821 examples [00:10, 968.03 examples/s]9923 examples [00:10, 981.69 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFHK672/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFHK672/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f74aea95510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f74aea95510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f74aea95510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f74acaf9278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7438cf4128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f74aea95510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f74aea95510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f74aea95510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7438cf4080>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7495cd41d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.21 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.21 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.21 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.96 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.96 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.44 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.44 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.33 file/s]2020-09-05 06:08:37.730656: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-05 06:08:37.734586: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-09-05 06:08:37.734731: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b469360030 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-05 06:08:37.734764: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 155888.34it/s] 45%|████▌     | 4464640/9912422 [00:00<00:24, 222356.07it/s]9920512it [00:00, 35915865.15it/s]                           
0it [00:00, ?it/s]32768it [00:00, 594492.57it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158419.02it/s]1654784it [00:00, 11332071.50it/s]                         
0it [00:00, ?it/s]8192it [00:00, 209441.64it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
