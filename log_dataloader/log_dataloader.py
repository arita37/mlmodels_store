
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '76c59fd9a4bb974b18235d07ef6a03bf2361dc5c', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7745937620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7745937620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f77b065c400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f77b065c400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f77cf879ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f77cf879ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f775d993400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f775d993400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f775d993400> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:33, 296834.99it/s]  2%|▏         | 212992/9912422 [00:00<00:25, 383364.19it/s]  9%|▉         | 876544/9912422 [00:00<00:17, 530506.38it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 749317.03it/s] 77%|███████▋  | 7585792/9912422 [00:00<00:02, 1059523.24it/s]9920512it [00:01, 9676598.19it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 138841.53it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 303543.32it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 391047.61it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 540806.33it/s]1654784it [00:00, 2656793.00it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 51739.04it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f775b9f71d0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f775bb4db70>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f775d9930d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f775d9930d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f775d9930d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:34,  1.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:34,  1.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:01<02:33,  1.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:01<02:32,  1.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:01<01:47,  1.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<01:47,  1.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<01:46,  1.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<01:46,  1.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<01:45,  1.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<01:44,  1.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<01:44,  1.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:01<01:13,  2.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<01:13,  2.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<01:12,  2.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<01:12,  2.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<01:11,  2.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<01:11,  2.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:50,  2.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:50,  2.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:50,  2.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:49,  2.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:49,  2.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:49,  2.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:35,  4.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:35,  4.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:33,  4.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:01<00:24,  5.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:23,  5.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:23,  5.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:23,  5.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:23,  5.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:17,  7.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:17,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:16,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:16,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:16,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:16,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:16,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 13.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 13.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 13.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 17.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:06, 17.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:06, 17.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:06, 17.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:02<00:04, 21.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:04, 21.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:04, 21.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:04, 21.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:04, 21.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:04, 21.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:04, 21.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 26.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 26.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 26.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 26.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 26.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 26.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 26.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:02<00:03, 30.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:03, 30.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 30.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:03, 30.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 30.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 30.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 30.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 35.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 35.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 35.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 35.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 35.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 35.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 35.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 38.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 38.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 38.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:02, 38.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:02, 38.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:02, 38.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:02, 38.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 41.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 41.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 41.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 41.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 41.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 44.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 44.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 44.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 44.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 44.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 44.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 44.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 46.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 46.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 46.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 46.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 46.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 46.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 46.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 47.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 47.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 47.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 47.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 47.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 47.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 47.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 47.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 47.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 47.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 47.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 47.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 47.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 47.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 48.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 48.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 48.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 48.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 48.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 48.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 48.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 48.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 48.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 48.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 48.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 48.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 48.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 48.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 48.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 48.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 48.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 48.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 48.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 48.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 49.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 49.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 49.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 49.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 49.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 49.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 49.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 50.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 50.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 50.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 50.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 50.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 50.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 50.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 50.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 50.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 50.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 50.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 50.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 50.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 50.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 50.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 51.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 51.08 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 51.08 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 51.08 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 51.08 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 51.08 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 51.08 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 51.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 51.28 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 51.28 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 51.28 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.24s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 51.28 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.24s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.24s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.97 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.24s/ url]
0 examples [00:00, ? examples/s]2020-08-30 12:07:09.405542: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-30 12:07:09.421371: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-30 12:07:09.421837: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559e3c855b00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-30 12:07:09.421859: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.15 examples/s]98 examples [00:00,  3.07 examples/s]197 examples [00:00,  4.39 examples/s]295 examples [00:00,  6.25 examples/s]400 examples [00:00,  8.91 examples/s]506 examples [00:00, 12.68 examples/s]612 examples [00:01, 18.02 examples/s]717 examples [00:01, 25.56 examples/s]819 examples [00:01, 36.12 examples/s]918 examples [00:01, 50.81 examples/s]1024 examples [00:01, 71.12 examples/s]1124 examples [00:01, 98.49 examples/s]1226 examples [00:01, 135.08 examples/s]1335 examples [00:01, 183.20 examples/s]1438 examples [00:01, 242.80 examples/s]1544 examples [00:01, 315.67 examples/s]1647 examples [00:02, 397.05 examples/s]1749 examples [00:02, 486.06 examples/s]1851 examples [00:02, 573.66 examples/s]1953 examples [00:02, 658.77 examples/s]2056 examples [00:02, 738.20 examples/s]2160 examples [00:02, 807.15 examples/s]2265 examples [00:02, 867.25 examples/s]2368 examples [00:02, 906.24 examples/s]2471 examples [00:02, 931.07 examples/s]2573 examples [00:02, 955.65 examples/s]2675 examples [00:03, 973.68 examples/s]2777 examples [00:03, 986.65 examples/s]2879 examples [00:03, 990.68 examples/s]2981 examples [00:03, 986.05 examples/s]3083 examples [00:03, 995.10 examples/s]3192 examples [00:03, 1020.74 examples/s]3300 examples [00:03, 1034.70 examples/s]3405 examples [00:03, 1025.81 examples/s]3509 examples [00:03, 1009.37 examples/s]3614 examples [00:04, 1020.60 examples/s]3717 examples [00:04, 1009.19 examples/s]3819 examples [00:04, 1009.05 examples/s]3921 examples [00:04, 1001.54 examples/s]4022 examples [00:04, 992.29 examples/s] 4122 examples [00:04, 991.36 examples/s]4227 examples [00:04, 1006.21 examples/s]4332 examples [00:04, 1017.15 examples/s]4434 examples [00:04, 1004.78 examples/s]4535 examples [00:04, 1002.63 examples/s]4636 examples [00:05, 994.83 examples/s] 4736 examples [00:05, 980.03 examples/s]4835 examples [00:05, 982.36 examples/s]4934 examples [00:05, 974.40 examples/s]5032 examples [00:05, 965.10 examples/s]5131 examples [00:05, 972.20 examples/s]5232 examples [00:05, 980.90 examples/s]5331 examples [00:05, 977.97 examples/s]5429 examples [00:05, 972.64 examples/s]5527 examples [00:05, 961.55 examples/s]5627 examples [00:06, 969.91 examples/s]5728 examples [00:06, 981.57 examples/s]5830 examples [00:06, 991.50 examples/s]5930 examples [00:06, 985.12 examples/s]6029 examples [00:06, 984.62 examples/s]6128 examples [00:06, 984.87 examples/s]6227 examples [00:06, 986.09 examples/s]6329 examples [00:06, 994.06 examples/s]6432 examples [00:06, 1002.17 examples/s]6534 examples [00:06, 1004.01 examples/s]6638 examples [00:07, 1014.36 examples/s]6742 examples [00:07, 1020.12 examples/s]6845 examples [00:07, 1009.77 examples/s]6954 examples [00:07, 1030.61 examples/s]7058 examples [00:07, 1012.31 examples/s]7160 examples [00:07, 1011.09 examples/s]7264 examples [00:07, 1018.86 examples/s]7366 examples [00:07, 1016.06 examples/s]7468 examples [00:07, 1008.99 examples/s]7569 examples [00:07, 983.49 examples/s] 7668 examples [00:08, 982.45 examples/s]7772 examples [00:08, 997.58 examples/s]7874 examples [00:08, 1001.91 examples/s]7975 examples [00:08, 994.40 examples/s] 8078 examples [00:08, 1002.97 examples/s]8180 examples [00:08, 1005.50 examples/s]8282 examples [00:08, 1009.62 examples/s]8386 examples [00:08, 1016.36 examples/s]8490 examples [00:08, 1021.91 examples/s]8593 examples [00:09, 1012.55 examples/s]8695 examples [00:09, 990.37 examples/s] 8796 examples [00:09, 993.02 examples/s]8898 examples [00:09, 999.72 examples/s]9001 examples [00:09, 1006.87 examples/s]9102 examples [00:09, 973.40 examples/s] 9200 examples [00:09, 951.11 examples/s]9296 examples [00:09, 950.56 examples/s]9400 examples [00:09, 975.27 examples/s]9498 examples [00:09, 976.09 examples/s]9596 examples [00:10, 973.26 examples/s]9698 examples [00:10, 985.44 examples/s]9797 examples [00:10, 984.19 examples/s]9898 examples [00:10, 991.78 examples/s]9998 examples [00:10, 989.68 examples/s]10098 examples [00:10, 914.43 examples/s]10198 examples [00:10, 937.14 examples/s]10298 examples [00:10, 952.59 examples/s]10394 examples [00:10, 952.98 examples/s]10495 examples [00:10, 968.44 examples/s]10594 examples [00:11, 973.14 examples/s]10692 examples [00:11, 967.28 examples/s]10789 examples [00:11, 949.69 examples/s]10889 examples [00:11, 962.57 examples/s]10990 examples [00:11, 974.43 examples/s]11088 examples [00:11, 972.01 examples/s]11191 examples [00:11, 985.87 examples/s]11299 examples [00:11, 1010.31 examples/s]11405 examples [00:11, 1023.87 examples/s]11513 examples [00:11, 1039.09 examples/s]11618 examples [00:12, 1006.92 examples/s]11720 examples [00:12, 1000.58 examples/s]11821 examples [00:12, 986.72 examples/s] 11925 examples [00:12, 999.90 examples/s]12027 examples [00:12, 1005.35 examples/s]12128 examples [00:12, 991.38 examples/s] 12231 examples [00:12, 999.53 examples/s]12332 examples [00:12, 977.24 examples/s]12430 examples [00:12, 975.86 examples/s]12539 examples [00:13, 1005.44 examples/s]12640 examples [00:13, 992.87 examples/s] 12746 examples [00:13, 1011.96 examples/s]12849 examples [00:13, 1015.71 examples/s]12951 examples [00:13, 1011.81 examples/s]13053 examples [00:13, 1012.38 examples/s]13155 examples [00:13, 1006.84 examples/s]13263 examples [00:13, 1024.99 examples/s]13366 examples [00:13, 1024.54 examples/s]13471 examples [00:13, 1031.10 examples/s]13575 examples [00:14, 1027.20 examples/s]13678 examples [00:14, 991.46 examples/s] 13778 examples [00:14, 989.07 examples/s]13878 examples [00:14, 978.44 examples/s]13981 examples [00:14, 992.60 examples/s]14081 examples [00:14, 988.46 examples/s]14183 examples [00:14, 997.54 examples/s]14283 examples [00:14, 985.65 examples/s]14382 examples [00:14, 985.56 examples/s]14486 examples [00:14, 999.84 examples/s]14587 examples [00:15, 967.55 examples/s]14690 examples [00:15, 984.77 examples/s]14792 examples [00:15, 993.81 examples/s]14896 examples [00:15, 1005.31 examples/s]15000 examples [00:15, 1013.31 examples/s]15102 examples [00:15, 1013.79 examples/s]15204 examples [00:15, 1013.21 examples/s]15309 examples [00:15, 1023.36 examples/s]15414 examples [00:15, 1028.90 examples/s]15517 examples [00:15, 1022.75 examples/s]15620 examples [00:16, 1018.79 examples/s]15722 examples [00:16, 1003.40 examples/s]15828 examples [00:16, 1019.52 examples/s]15938 examples [00:16, 1041.89 examples/s]16043 examples [00:16, 1036.72 examples/s]16147 examples [00:16, 1017.75 examples/s]16249 examples [00:16, 1017.05 examples/s]16357 examples [00:16, 1033.64 examples/s]16464 examples [00:16, 1043.23 examples/s]16574 examples [00:17, 1057.36 examples/s]16680 examples [00:17, 1026.20 examples/s]16785 examples [00:17, 1031.93 examples/s]16889 examples [00:17, 1022.62 examples/s]16992 examples [00:17, 1022.70 examples/s]17095 examples [00:17, 1018.53 examples/s]17197 examples [00:17, 1011.34 examples/s]17299 examples [00:17, 1005.47 examples/s]17400 examples [00:17, 1005.17 examples/s]17504 examples [00:17, 1014.84 examples/s]17606 examples [00:18, 998.11 examples/s] 17706 examples [00:18, 979.19 examples/s]17806 examples [00:18, 982.96 examples/s]17917 examples [00:18, 1015.90 examples/s]18019 examples [00:18, 1012.02 examples/s]18121 examples [00:18, 1007.75 examples/s]18224 examples [00:18, 1012.97 examples/s]18326 examples [00:18, 1008.65 examples/s]18434 examples [00:18, 1026.20 examples/s]18538 examples [00:18, 1029.05 examples/s]18642 examples [00:19, 1028.56 examples/s]18745 examples [00:19, 1012.16 examples/s]18849 examples [00:19, 1017.73 examples/s]18952 examples [00:19, 1021.24 examples/s]19057 examples [00:19, 1027.80 examples/s]19162 examples [00:19, 1034.10 examples/s]19266 examples [00:19, 1021.40 examples/s]19369 examples [00:19, 1021.80 examples/s]19472 examples [00:19, 1008.51 examples/s]19573 examples [00:19, 978.84 examples/s] 19672 examples [00:20, 972.90 examples/s]19770 examples [00:20, 971.63 examples/s]19875 examples [00:20, 991.97 examples/s]19976 examples [00:20, 995.63 examples/s]20076 examples [00:20, 954.37 examples/s]20178 examples [00:20, 971.46 examples/s]20276 examples [00:20, 969.96 examples/s]20376 examples [00:20, 976.37 examples/s]20478 examples [00:20, 986.36 examples/s]20577 examples [00:21, 981.49 examples/s]20682 examples [00:21, 1000.97 examples/s]20785 examples [00:21, 1008.32 examples/s]20893 examples [00:21, 1027.66 examples/s]21004 examples [00:21, 1049.98 examples/s]21113 examples [00:21, 1059.10 examples/s]21220 examples [00:21, 1050.03 examples/s]21326 examples [00:21, 1028.57 examples/s]21430 examples [00:21, 1029.53 examples/s]21534 examples [00:21, 1020.30 examples/s]21637 examples [00:22, 1002.52 examples/s]21742 examples [00:22, 1014.98 examples/s]21844 examples [00:22, 1015.16 examples/s]21949 examples [00:22, 1024.34 examples/s]22052 examples [00:22, 1009.80 examples/s]22159 examples [00:22, 1025.46 examples/s]22262 examples [00:22, 1019.43 examples/s]22366 examples [00:22, 1024.40 examples/s]22475 examples [00:22, 1042.70 examples/s]22582 examples [00:22, 1049.98 examples/s]22690 examples [00:23, 1057.82 examples/s]22796 examples [00:23, 1033.81 examples/s]22900 examples [00:23, 1016.66 examples/s]23002 examples [00:23, 1006.32 examples/s]23103 examples [00:23, 984.14 examples/s] 23202 examples [00:23, 979.50 examples/s]23301 examples [00:23, 982.19 examples/s]23400 examples [00:23, 976.43 examples/s]23503 examples [00:23, 991.60 examples/s]23603 examples [00:23, 991.65 examples/s]23703 examples [00:24, 986.24 examples/s]23802 examples [00:24, 946.71 examples/s]23900 examples [00:24, 955.89 examples/s]24004 examples [00:24, 977.89 examples/s]24109 examples [00:24, 997.60 examples/s]24210 examples [00:24, 996.35 examples/s]24311 examples [00:24, 999.90 examples/s]24412 examples [00:24, 991.87 examples/s]24512 examples [00:24, 989.29 examples/s]24615 examples [00:25, 993.07 examples/s]24715 examples [00:25, 925.96 examples/s]24809 examples [00:25, 929.62 examples/s]24903 examples [00:25, 928.73 examples/s]24997 examples [00:25, 911.64 examples/s]25096 examples [00:25, 932.77 examples/s]25198 examples [00:25, 957.24 examples/s]25298 examples [00:25, 968.75 examples/s]25396 examples [00:25, 968.30 examples/s]25501 examples [00:25, 990.71 examples/s]25606 examples [00:26, 1005.55 examples/s]25707 examples [00:26, 991.19 examples/s] 25808 examples [00:26, 996.57 examples/s]25908 examples [00:26, 994.80 examples/s]26010 examples [00:26, 1001.52 examples/s]26111 examples [00:26, 998.97 examples/s] 26211 examples [00:26, 995.84 examples/s]26316 examples [00:26, 1009.99 examples/s]26418 examples [00:26, 999.92 examples/s] 26521 examples [00:26, 1006.44 examples/s]26623 examples [00:27, 1008.20 examples/s]26724 examples [00:27, 990.84 examples/s] 26828 examples [00:27, 1005.02 examples/s]26929 examples [00:27, 989.44 examples/s] 27029 examples [00:27, 986.25 examples/s]27128 examples [00:27, 982.03 examples/s]27227 examples [00:27, 961.93 examples/s]27329 examples [00:27, 978.46 examples/s]27431 examples [00:27, 989.09 examples/s]27533 examples [00:27, 997.80 examples/s]27633 examples [00:28, 992.40 examples/s]27733 examples [00:28, 990.63 examples/s]27841 examples [00:28, 1014.84 examples/s]27947 examples [00:28, 1027.90 examples/s]28053 examples [00:28, 1037.31 examples/s]28157 examples [00:28, 1018.11 examples/s]28261 examples [00:28, 1022.40 examples/s]28365 examples [00:28, 1027.07 examples/s]28468 examples [00:28, 1002.42 examples/s]28572 examples [00:29, 1011.05 examples/s]28677 examples [00:29, 1021.90 examples/s]28780 examples [00:29, 1013.71 examples/s]28882 examples [00:29, 987.96 examples/s] 28984 examples [00:29, 996.54 examples/s]29090 examples [00:29, 1013.25 examples/s]29196 examples [00:29, 1025.58 examples/s]29299 examples [00:29, 1024.33 examples/s]29402 examples [00:29, 1023.17 examples/s]29505 examples [00:29, 1020.70 examples/s]29608 examples [00:30, 1009.20 examples/s]29711 examples [00:30, 1012.66 examples/s]29813 examples [00:30, 988.84 examples/s] 29913 examples [00:30, 951.12 examples/s]30009 examples [00:30, 896.01 examples/s]30101 examples [00:30, 901.21 examples/s]30200 examples [00:30, 924.82 examples/s]30304 examples [00:30, 954.27 examples/s]30403 examples [00:30, 963.92 examples/s]30505 examples [00:30, 977.93 examples/s]30610 examples [00:31, 996.83 examples/s]30711 examples [00:31, 996.54 examples/s]30819 examples [00:31, 1018.65 examples/s]30922 examples [00:31, 1016.94 examples/s]31027 examples [00:31, 1026.59 examples/s]31130 examples [00:31, 1026.29 examples/s]31233 examples [00:31, 1027.33 examples/s]31343 examples [00:31, 1045.34 examples/s]31448 examples [00:31, 1010.12 examples/s]31556 examples [00:31, 1029.63 examples/s]31660 examples [00:32, 1013.35 examples/s]31762 examples [00:32, 1004.18 examples/s]31863 examples [00:32, 999.73 examples/s] 31964 examples [00:32, 992.75 examples/s]32064 examples [00:32, 989.80 examples/s]32167 examples [00:32, 1000.71 examples/s]32268 examples [00:32, 1000.23 examples/s]32372 examples [00:32, 1009.37 examples/s]32474 examples [00:32, 1000.67 examples/s]32575 examples [00:33, 998.81 examples/s] 32676 examples [00:33, 1000.49 examples/s]32778 examples [00:33, 1004.55 examples/s]32879 examples [00:33, 1004.74 examples/s]32980 examples [00:33, 990.49 examples/s] 33082 examples [00:33, 996.49 examples/s]33182 examples [00:33, 996.13 examples/s]33286 examples [00:33, 1006.79 examples/s]33388 examples [00:33, 1007.29 examples/s]33489 examples [00:33, 989.10 examples/s] 33589 examples [00:34, 982.95 examples/s]33688 examples [00:34, 977.81 examples/s]33786 examples [00:34, 977.93 examples/s]33886 examples [00:34, 981.65 examples/s]33989 examples [00:34, 994.18 examples/s]34094 examples [00:34, 1008.53 examples/s]34195 examples [00:34, 1004.23 examples/s]34296 examples [00:34, 1004.36 examples/s]34400 examples [00:34, 1014.53 examples/s]34505 examples [00:34, 1023.35 examples/s]34608 examples [00:35, 1022.04 examples/s]34711 examples [00:35, 999.70 examples/s] 34812 examples [00:35, 992.32 examples/s]34912 examples [00:35, 961.69 examples/s]35009 examples [00:35, 932.92 examples/s]35107 examples [00:35, 945.67 examples/s]35205 examples [00:35, 953.81 examples/s]35305 examples [00:35, 965.55 examples/s]35403 examples [00:35, 968.55 examples/s]35505 examples [00:35, 982.50 examples/s]35613 examples [00:36, 1006.58 examples/s]35718 examples [00:36, 1018.38 examples/s]35821 examples [00:36, 1017.57 examples/s]35931 examples [00:36, 1038.27 examples/s]36036 examples [00:36, 1038.36 examples/s]36140 examples [00:36, 1033.01 examples/s]36244 examples [00:36, 1031.04 examples/s]36350 examples [00:36, 1036.46 examples/s]36454 examples [00:36, 1037.28 examples/s]36558 examples [00:37, 1005.69 examples/s]36659 examples [00:37, 999.22 examples/s] 36762 examples [00:37, 1006.45 examples/s]36864 examples [00:37, 1006.97 examples/s]36965 examples [00:37, 987.07 examples/s] 37064 examples [00:37, 963.77 examples/s]37161 examples [00:37, 962.05 examples/s]37258 examples [00:37, 957.97 examples/s]37355 examples [00:37, 960.59 examples/s]37452 examples [00:37, 955.62 examples/s]37549 examples [00:38, 957.44 examples/s]37649 examples [00:38, 968.32 examples/s]37753 examples [00:38, 987.07 examples/s]37859 examples [00:38, 1007.11 examples/s]37964 examples [00:38, 1018.82 examples/s]38067 examples [00:38, 1019.99 examples/s]38171 examples [00:38, 1025.91 examples/s]38274 examples [00:38, 1017.92 examples/s]38377 examples [00:38, 1019.56 examples/s]38484 examples [00:38, 1031.62 examples/s]38588 examples [00:39, 1029.12 examples/s]38691 examples [00:39, 1016.27 examples/s]38793 examples [00:39, 1016.41 examples/s]38895 examples [00:39, 1009.46 examples/s]38999 examples [00:39, 1018.22 examples/s]39105 examples [00:39, 1029.42 examples/s]39209 examples [00:39, 1029.39 examples/s]39312 examples [00:39, 1021.35 examples/s]39415 examples [00:39, 1018.32 examples/s]39521 examples [00:39, 1027.17 examples/s]39624 examples [00:40, 1007.04 examples/s]39725 examples [00:40, 997.84 examples/s] 39825 examples [00:40, 985.05 examples/s]39924 examples [00:40, 971.70 examples/s]40022 examples [00:40, 921.15 examples/s]40120 examples [00:40, 936.78 examples/s]40218 examples [00:40, 947.25 examples/s]40322 examples [00:40, 971.17 examples/s]40424 examples [00:40, 983.75 examples/s]40523 examples [00:40, 980.52 examples/s]40622 examples [00:41, 968.53 examples/s]40725 examples [00:41, 985.82 examples/s]40834 examples [00:41, 1014.67 examples/s]40938 examples [00:41, 1020.81 examples/s]41045 examples [00:41, 1033.08 examples/s]41149 examples [00:41, 1029.21 examples/s]41254 examples [00:41, 1032.72 examples/s]41358 examples [00:41, 1026.55 examples/s]41461 examples [00:41, 1011.22 examples/s]41563 examples [00:42, 1001.79 examples/s]41664 examples [00:42, 1001.58 examples/s]41767 examples [00:42, 1009.93 examples/s]41869 examples [00:42, 1012.87 examples/s]41975 examples [00:42, 1024.10 examples/s]42082 examples [00:42, 1036.19 examples/s]42186 examples [00:42, 1033.71 examples/s]42290 examples [00:42, 1032.70 examples/s]42398 examples [00:42, 1044.50 examples/s]42503 examples [00:42, 1032.24 examples/s]42607 examples [00:43, 1016.00 examples/s]42709 examples [00:43, 1013.75 examples/s]42811 examples [00:43, 1007.45 examples/s]42916 examples [00:43, 1018.13 examples/s]43018 examples [00:43, 1005.94 examples/s]43119 examples [00:43, 986.94 examples/s] 43221 examples [00:43, 996.59 examples/s]43323 examples [00:43, 1002.32 examples/s]43424 examples [00:43, 1002.34 examples/s]43529 examples [00:43, 1015.92 examples/s]43631 examples [00:44, 1004.08 examples/s]43732 examples [00:44, 999.71 examples/s] 43833 examples [00:44, 997.57 examples/s]43933 examples [00:44, 996.60 examples/s]44033 examples [00:44, 996.06 examples/s]44133 examples [00:44, 983.10 examples/s]44232 examples [00:44, 977.20 examples/s]44330 examples [00:44, 964.24 examples/s]44427 examples [00:44, 963.06 examples/s]44529 examples [00:44, 977.89 examples/s]44627 examples [00:45, 953.30 examples/s]44728 examples [00:45, 969.12 examples/s]44826 examples [00:45, 971.30 examples/s]44929 examples [00:45, 986.82 examples/s]45029 examples [00:45, 989.04 examples/s]45131 examples [00:45, 995.62 examples/s]45233 examples [00:45, 1000.80 examples/s]45334 examples [00:45, 973.68 examples/s] 45439 examples [00:45, 994.04 examples/s]45550 examples [00:45, 1024.50 examples/s]45653 examples [00:46, 1013.18 examples/s]45760 examples [00:46, 1028.96 examples/s]45869 examples [00:46, 1044.78 examples/s]45974 examples [00:46, 1034.07 examples/s]46078 examples [00:46, 988.97 examples/s] 46182 examples [00:46, 1001.88 examples/s]46283 examples [00:46, 998.15 examples/s] 46386 examples [00:46, 1007.44 examples/s]46487 examples [00:46, 1006.44 examples/s]46590 examples [00:47, 1012.45 examples/s]46692 examples [00:47, 992.11 examples/s] 46792 examples [00:47, 990.50 examples/s]46893 examples [00:47, 994.94 examples/s]46995 examples [00:47, 1000.13 examples/s]47096 examples [00:47, 974.27 examples/s] 47194 examples [00:47, 967.39 examples/s]47292 examples [00:47, 970.90 examples/s]47396 examples [00:47, 990.02 examples/s]47505 examples [00:47, 1017.38 examples/s]47611 examples [00:48, 1028.56 examples/s]47715 examples [00:48, 1028.07 examples/s]47818 examples [00:48, 1010.66 examples/s]47920 examples [00:48, 964.66 examples/s] 48021 examples [00:48, 976.50 examples/s]48126 examples [00:48, 995.52 examples/s]48231 examples [00:48, 1009.47 examples/s]48339 examples [00:48, 1028.36 examples/s]48443 examples [00:48, 991.51 examples/s] 48543 examples [00:48, 992.59 examples/s]48643 examples [00:49, 992.49 examples/s]48743 examples [00:49, 986.06 examples/s]48845 examples [00:49, 993.94 examples/s]48949 examples [00:49, 1004.92 examples/s]49054 examples [00:49, 1015.70 examples/s]49156 examples [00:49, 995.49 examples/s] 49256 examples [00:49, 984.74 examples/s]49356 examples [00:49, 988.48 examples/s]49455 examples [00:49, 980.13 examples/s]49554 examples [00:50, 962.21 examples/s]49652 examples [00:50, 965.05 examples/s]49751 examples [00:50, 970.12 examples/s]49850 examples [00:50, 974.53 examples/s]49953 examples [00:50, 989.89 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7049/50000 [00:00<00:00, 70485.87 examples/s] 39%|███▉      | 19446/50000 [00:00<00:00, 80964.57 examples/s] 64%|██████▎   | 31751/50000 [00:00<00:00, 90220.68 examples/s] 87%|████████▋ | 43264/50000 [00:00<00:00, 96482.89 examples/s]                                                               0 examples [00:00, ? examples/s]75 examples [00:00, 749.69 examples/s]175 examples [00:00, 809.75 examples/s]276 examples [00:00, 858.89 examples/s]366 examples [00:00, 868.97 examples/s]465 examples [00:00, 901.68 examples/s]568 examples [00:00, 934.83 examples/s]669 examples [00:00, 954.87 examples/s]774 examples [00:00, 980.81 examples/s]874 examples [00:00, 984.51 examples/s]976 examples [00:01, 994.14 examples/s]1081 examples [00:01, 1009.38 examples/s]1184 examples [00:01, 1012.55 examples/s]1286 examples [00:01, 1013.62 examples/s]1387 examples [00:01, 986.33 examples/s] 1493 examples [00:01, 1006.57 examples/s]1598 examples [00:01, 1016.04 examples/s]1700 examples [00:01, 997.00 examples/s] 1800 examples [00:01, 994.73 examples/s]1900 examples [00:01, 979.46 examples/s]1999 examples [00:02, 973.34 examples/s]2101 examples [00:02, 984.64 examples/s]2202 examples [00:02, 989.84 examples/s]2305 examples [00:02, 1000.98 examples/s]2406 examples [00:02, 993.07 examples/s] 2506 examples [00:02, 995.11 examples/s]2606 examples [00:02, 989.19 examples/s]2705 examples [00:02, 981.97 examples/s]2804 examples [00:02, 983.82 examples/s]2907 examples [00:02, 994.88 examples/s]3010 examples [00:03, 1005.03 examples/s]3111 examples [00:03, 988.38 examples/s] 3213 examples [00:03, 997.41 examples/s]3315 examples [00:03, 1002.52 examples/s]3416 examples [00:03, 998.53 examples/s] 3519 examples [00:03, 1006.56 examples/s]3622 examples [00:03, 1009.96 examples/s]3724 examples [00:03, 1006.52 examples/s]3827 examples [00:03, 1011.26 examples/s]3929 examples [00:03, 1005.46 examples/s]4030 examples [00:04, 1004.14 examples/s]4137 examples [00:04, 1021.14 examples/s]4240 examples [00:04, 1017.95 examples/s]4342 examples [00:04, 1014.42 examples/s]4444 examples [00:04, 1013.70 examples/s]4553 examples [00:04, 1033.69 examples/s]4657 examples [00:04, 1030.43 examples/s]4761 examples [00:04, 845.09 examples/s] 4855 examples [00:04, 869.56 examples/s]4959 examples [00:05, 912.56 examples/s]5057 examples [00:05, 929.34 examples/s]5153 examples [00:05, 916.96 examples/s]5252 examples [00:05, 936.61 examples/s]5347 examples [00:05, 937.03 examples/s]5449 examples [00:05, 957.95 examples/s]5552 examples [00:05, 976.92 examples/s]5654 examples [00:05, 988.52 examples/s]5763 examples [00:05, 1014.28 examples/s]5868 examples [00:05, 1023.88 examples/s]5973 examples [00:06, 1031.00 examples/s]6077 examples [00:06, 1028.34 examples/s]6181 examples [00:06, 1021.85 examples/s]6285 examples [00:06, 1026.91 examples/s]6388 examples [00:06, 986.94 examples/s] 6499 examples [00:06, 1020.85 examples/s]6604 examples [00:06, 1028.43 examples/s]6708 examples [00:06, 1030.52 examples/s]6812 examples [00:06, 1029.00 examples/s]6916 examples [00:06, 1020.35 examples/s]7019 examples [00:07, 987.44 examples/s] 7119 examples [00:07, 986.60 examples/s]7218 examples [00:07, 985.60 examples/s]7319 examples [00:07, 991.14 examples/s]7425 examples [00:07, 1008.98 examples/s]7527 examples [00:07, 1008.02 examples/s]7628 examples [00:07, 1004.74 examples/s]7730 examples [00:07, 1008.13 examples/s]7831 examples [00:07, 981.98 examples/s] 7930 examples [00:08, 979.21 examples/s]8029 examples [00:08, 976.35 examples/s]8132 examples [00:08, 990.09 examples/s]8232 examples [00:08, 957.38 examples/s]8330 examples [00:08, 962.92 examples/s]8428 examples [00:08, 965.64 examples/s]8526 examples [00:08, 967.92 examples/s]8626 examples [00:08, 974.70 examples/s]8724 examples [00:08, 969.27 examples/s]8825 examples [00:08, 979.55 examples/s]8924 examples [00:09, 970.62 examples/s]9022 examples [00:09, 945.22 examples/s]9119 examples [00:09, 950.43 examples/s]9219 examples [00:09, 962.61 examples/s]9316 examples [00:09, 957.50 examples/s]9412 examples [00:09, 952.28 examples/s]9508 examples [00:09, 954.41 examples/s]9614 examples [00:09, 983.18 examples/s]9714 examples [00:09, 987.41 examples/s]9815 examples [00:09, 992.22 examples/s]9916 examples [00:10, 995.47 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteAXD8IP/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteAXD8IP/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f775d993400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f775d993400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f775d993400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f775b9f71d0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f76e89d10f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f775d993400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f775d993400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f775d993400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f76e89d12e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f775bb53198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.93 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.93 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.93 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.82 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.82 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.25 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.25 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.26 file/s]2020-08-30 12:08:17.690528: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-30 12:08:17.694427: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-30 12:08:17.694573: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56222c3f9470 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-30 12:08:17.694589: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 307911.47it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 396233.72it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 547505.12it/s] 31%|███       | 3039232/9912422 [00:00<00:08, 771138.67it/s] 59%|█████▊    | 5799936/9912422 [00:00<00:03, 1084654.29it/s] 89%|████████▉ | 8847360/9912422 [00:01<00:00, 1518879.41it/s]9920512it [00:01, 9729922.34it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 140204.34it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 296105.51it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 382534.93it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 529141.47it/s]1654784it [00:00, 2657758.47it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 52178.40it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
