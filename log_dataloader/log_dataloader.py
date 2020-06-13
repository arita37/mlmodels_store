
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/19be216fef0a1e9baf27c7f6397b10603f2487c0', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '19be216fef0a1e9baf27c7f6397b10603f2487c0', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/19be216fef0a1e9baf27c7f6397b10603f2487c0

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/19be216fef0a1e9baf27c7f6397b10603f2487c0

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/19be216fef0a1e9baf27c7f6397b10603f2487c0

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f319ca7f620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f319ca7f620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f32080460d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f32080460d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3221d97ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3221d97ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f31b58c1950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f31b58c1950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f31b58c1950> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 134843.49it/s] 70%|██████▉   | 6922240/9912422 [00:00<00:15, 192469.98it/s]9920512it [00:00, 40176817.33it/s]                           
0it [00:00, ?it/s]32768it [00:00, 573006.11it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160294.74it/s]1654784it [00:00, 11192548.40it/s]                         
0it [00:00, ?it/s]8192it [00:00, 193681.83it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f31b2ce1d30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f31b2cd89b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f31b58c1598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f31b58c1598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f31b58c1598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:20,  1.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:20,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:20,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:19,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:19,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:18,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:18,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:17,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:54,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:53,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:52,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:51,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:51,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:36,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:35,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:34,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:15,  7.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:15,  7.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11, 10.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 19.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 19.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 24.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 24.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 31.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 31.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 38.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 38.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 38.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 38.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 38.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 38.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 45.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 45.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 45.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 52.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 52.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 59.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 59.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 59.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 59.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 59.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 59.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 59.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 59.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.49s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.49s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.49s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.49s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.93 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ url]
0 examples [00:00, ? examples/s]2020-06-13 14:21:11.016551: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-13 14:21:11.021945: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-13 14:21:11.022088: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557405557bf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 14:21:11.022101: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
69 examples [00:00, 689.75 examples/s]186 examples [00:00, 786.20 examples/s]302 examples [00:00, 869.41 examples/s]404 examples [00:00, 907.65 examples/s]522 examples [00:00, 973.87 examples/s]639 examples [00:00, 1024.60 examples/s]761 examples [00:00, 1075.51 examples/s]867 examples [00:00, 1066.73 examples/s]981 examples [00:00, 1086.15 examples/s]1089 examples [00:01, 1037.46 examples/s]1207 examples [00:01, 1074.86 examples/s]1324 examples [00:01, 1099.00 examples/s]1441 examples [00:01, 1117.24 examples/s]1557 examples [00:01, 1127.14 examples/s]1670 examples [00:01, 1119.35 examples/s]1783 examples [00:01, 1079.19 examples/s]1895 examples [00:01, 1090.85 examples/s]2012 examples [00:01, 1111.89 examples/s]2133 examples [00:01, 1136.84 examples/s]2248 examples [00:02, 1107.71 examples/s]2365 examples [00:02, 1123.95 examples/s]2478 examples [00:02, 1108.23 examples/s]2590 examples [00:02, 1110.24 examples/s]2712 examples [00:02, 1138.82 examples/s]2827 examples [00:02, 1140.56 examples/s]2942 examples [00:02, 1137.33 examples/s]3056 examples [00:02, 1077.76 examples/s]3166 examples [00:02, 1082.42 examples/s]3282 examples [00:02, 1102.60 examples/s]3393 examples [00:03, 1096.60 examples/s]3503 examples [00:03, 1088.95 examples/s]3621 examples [00:03, 1113.05 examples/s]3736 examples [00:03, 1122.14 examples/s]3852 examples [00:03, 1131.39 examples/s]3966 examples [00:03, 1126.32 examples/s]4079 examples [00:03, 1121.96 examples/s]4203 examples [00:03, 1153.80 examples/s]4319 examples [00:03, 1141.78 examples/s]4436 examples [00:03, 1148.35 examples/s]4552 examples [00:04, 1132.35 examples/s]4670 examples [00:04, 1144.34 examples/s]4789 examples [00:04, 1154.81 examples/s]4905 examples [00:04, 1140.50 examples/s]5025 examples [00:04, 1157.41 examples/s]5141 examples [00:04, 1154.19 examples/s]5258 examples [00:04, 1140.97 examples/s]5373 examples [00:04, 1107.64 examples/s]5485 examples [00:04, 1100.07 examples/s]5596 examples [00:05, 1102.27 examples/s]5716 examples [00:05, 1127.97 examples/s]5830 examples [00:05, 1127.30 examples/s]5943 examples [00:05, 1112.91 examples/s]6055 examples [00:05, 1109.13 examples/s]6173 examples [00:05, 1127.77 examples/s]6295 examples [00:05, 1153.43 examples/s]6414 examples [00:05, 1163.77 examples/s]6535 examples [00:05, 1176.57 examples/s]6653 examples [00:05, 1130.58 examples/s]6768 examples [00:06, 1136.16 examples/s]6903 examples [00:06, 1190.24 examples/s]7026 examples [00:06, 1200.56 examples/s]7147 examples [00:06, 1203.17 examples/s]7268 examples [00:06, 1193.88 examples/s]7388 examples [00:06, 1182.81 examples/s]7507 examples [00:06, 1183.65 examples/s]7627 examples [00:06, 1188.33 examples/s]7746 examples [00:06, 1183.31 examples/s]7871 examples [00:06, 1200.48 examples/s]8000 examples [00:07, 1221.14 examples/s]8123 examples [00:07, 1214.04 examples/s]8245 examples [00:07, 1193.20 examples/s]8365 examples [00:07, 1154.86 examples/s]8490 examples [00:07, 1179.93 examples/s]8609 examples [00:07, 1176.55 examples/s]8727 examples [00:07, 1169.87 examples/s]8845 examples [00:07, 1115.58 examples/s]8958 examples [00:07, 1113.95 examples/s]9070 examples [00:08, 1115.14 examples/s]9185 examples [00:08, 1125.09 examples/s]9299 examples [00:08, 1127.47 examples/s]9412 examples [00:08, 1119.83 examples/s]9527 examples [00:08, 1127.54 examples/s]9642 examples [00:08, 1133.76 examples/s]9756 examples [00:08, 1129.05 examples/s]9872 examples [00:08, 1135.79 examples/s]9986 examples [00:08, 1128.78 examples/s]10099 examples [00:08, 1056.65 examples/s]10206 examples [00:09, 1052.21 examples/s]10320 examples [00:09, 1076.77 examples/s]10437 examples [00:09, 1101.73 examples/s]10549 examples [00:09, 1105.26 examples/s]10661 examples [00:09, 1109.37 examples/s]10779 examples [00:09, 1128.14 examples/s]10897 examples [00:09, 1140.98 examples/s]11014 examples [00:09, 1147.79 examples/s]11130 examples [00:09, 1149.78 examples/s]11246 examples [00:09, 1139.48 examples/s]11361 examples [00:10, 1140.84 examples/s]11476 examples [00:10, 1139.63 examples/s]11591 examples [00:10, 1139.98 examples/s]11706 examples [00:10, 1139.28 examples/s]11820 examples [00:10, 1129.62 examples/s]11933 examples [00:10, 1125.97 examples/s]12046 examples [00:10, 1126.28 examples/s]12159 examples [00:10, 1117.39 examples/s]12271 examples [00:10, 1077.64 examples/s]12380 examples [00:10, 1076.16 examples/s]12491 examples [00:11, 1083.71 examples/s]12605 examples [00:11, 1097.81 examples/s]12722 examples [00:11, 1116.07 examples/s]12840 examples [00:11, 1132.86 examples/s]12958 examples [00:11, 1145.33 examples/s]13073 examples [00:11, 1141.00 examples/s]13192 examples [00:11, 1152.89 examples/s]13308 examples [00:11, 1147.83 examples/s]13423 examples [00:11, 1136.00 examples/s]13537 examples [00:11, 1123.87 examples/s]13650 examples [00:12, 1106.20 examples/s]13766 examples [00:12, 1119.63 examples/s]13880 examples [00:12, 1125.44 examples/s]13993 examples [00:12, 1121.29 examples/s]14106 examples [00:12, 1108.16 examples/s]14219 examples [00:12, 1113.69 examples/s]14331 examples [00:12, 1106.24 examples/s]14448 examples [00:12, 1124.55 examples/s]14575 examples [00:12, 1162.01 examples/s]14695 examples [00:13, 1171.94 examples/s]14813 examples [00:13, 1157.51 examples/s]14939 examples [00:13, 1184.50 examples/s]15058 examples [00:13, 1167.40 examples/s]15176 examples [00:13, 1160.59 examples/s]15293 examples [00:13, 1150.61 examples/s]15409 examples [00:13, 1141.47 examples/s]15526 examples [00:13, 1148.44 examples/s]15641 examples [00:13, 1140.17 examples/s]15756 examples [00:13, 1116.18 examples/s]15870 examples [00:14, 1121.71 examples/s]15988 examples [00:14, 1135.27 examples/s]16104 examples [00:14, 1141.96 examples/s]16221 examples [00:14, 1148.34 examples/s]16336 examples [00:14, 1143.05 examples/s]16456 examples [00:14, 1158.09 examples/s]16577 examples [00:14, 1170.89 examples/s]16708 examples [00:14, 1207.78 examples/s]16830 examples [00:14, 1189.56 examples/s]16950 examples [00:14, 1158.26 examples/s]17067 examples [00:15, 1158.97 examples/s]17184 examples [00:15, 1136.47 examples/s]17305 examples [00:15, 1156.07 examples/s]17425 examples [00:15, 1166.67 examples/s]17549 examples [00:15, 1184.89 examples/s]17668 examples [00:15, 1183.17 examples/s]17793 examples [00:15, 1200.86 examples/s]17914 examples [00:15, 1175.85 examples/s]18032 examples [00:15, 1151.80 examples/s]18148 examples [00:15, 1129.82 examples/s]18262 examples [00:16, 1104.73 examples/s]18373 examples [00:16, 1105.48 examples/s]18491 examples [00:16, 1124.14 examples/s]18613 examples [00:16, 1150.35 examples/s]18729 examples [00:16, 1145.29 examples/s]18846 examples [00:16, 1151.70 examples/s]18962 examples [00:16, 1150.62 examples/s]19079 examples [00:16, 1155.61 examples/s]19195 examples [00:16, 1127.09 examples/s]19308 examples [00:17, 1108.79 examples/s]19420 examples [00:17, 1086.12 examples/s]19538 examples [00:17, 1110.90 examples/s]19656 examples [00:17, 1129.92 examples/s]19770 examples [00:17, 1126.19 examples/s]19883 examples [00:17, 1119.82 examples/s]20001 examples [00:17, 1084.90 examples/s]20117 examples [00:17, 1103.66 examples/s]20238 examples [00:17, 1132.89 examples/s]20358 examples [00:17, 1151.55 examples/s]20477 examples [00:18, 1160.21 examples/s]20594 examples [00:18, 1129.09 examples/s]20713 examples [00:18, 1146.33 examples/s]20828 examples [00:18, 1092.96 examples/s]20939 examples [00:18, 1088.91 examples/s]21053 examples [00:18, 1102.99 examples/s]21164 examples [00:18, 1096.10 examples/s]21284 examples [00:18, 1124.40 examples/s]21397 examples [00:18, 1118.85 examples/s]21510 examples [00:18, 1112.30 examples/s]21622 examples [00:19, 1052.95 examples/s]21729 examples [00:19, 1037.43 examples/s]21846 examples [00:19, 1072.47 examples/s]21966 examples [00:19, 1105.50 examples/s]22083 examples [00:19, 1123.09 examples/s]22196 examples [00:19, 1124.54 examples/s]22309 examples [00:19, 1109.28 examples/s]22421 examples [00:19, 1107.19 examples/s]22539 examples [00:19, 1127.10 examples/s]22652 examples [00:20, 1122.34 examples/s]22765 examples [00:20, 1121.33 examples/s]22878 examples [00:20, 1092.60 examples/s]22988 examples [00:20, 1077.08 examples/s]23101 examples [00:20, 1091.12 examples/s]23213 examples [00:20, 1097.67 examples/s]23340 examples [00:20, 1144.14 examples/s]23456 examples [00:20, 1125.96 examples/s]23570 examples [00:20, 1123.14 examples/s]23688 examples [00:20, 1138.56 examples/s]23804 examples [00:21, 1142.20 examples/s]23919 examples [00:21, 1100.86 examples/s]24031 examples [00:21, 1105.03 examples/s]24145 examples [00:21, 1114.89 examples/s]24264 examples [00:21, 1135.08 examples/s]24378 examples [00:21, 1133.69 examples/s]24495 examples [00:21, 1143.17 examples/s]24612 examples [00:21, 1149.82 examples/s]24728 examples [00:21, 1145.57 examples/s]24848 examples [00:21, 1159.16 examples/s]24965 examples [00:22, 1137.55 examples/s]25085 examples [00:22, 1153.55 examples/s]25201 examples [00:22, 1154.05 examples/s]25326 examples [00:22, 1179.41 examples/s]25447 examples [00:22, 1186.86 examples/s]25574 examples [00:22, 1209.83 examples/s]25696 examples [00:22, 1188.73 examples/s]25816 examples [00:22, 1172.24 examples/s]25934 examples [00:22, 1165.23 examples/s]26059 examples [00:23, 1187.70 examples/s]26178 examples [00:23, 1181.60 examples/s]26297 examples [00:23, 1110.75 examples/s]26426 examples [00:23, 1158.74 examples/s]26549 examples [00:23, 1177.61 examples/s]26677 examples [00:23, 1205.33 examples/s]26799 examples [00:23, 1195.78 examples/s]26923 examples [00:23, 1206.39 examples/s]27049 examples [00:23, 1220.78 examples/s]27172 examples [00:23, 1210.47 examples/s]27294 examples [00:24, 1178.74 examples/s]27413 examples [00:24, 1157.33 examples/s]27542 examples [00:24, 1191.72 examples/s]27662 examples [00:24, 1188.09 examples/s]27782 examples [00:24, 1185.48 examples/s]27902 examples [00:24, 1189.24 examples/s]28022 examples [00:24, 1175.00 examples/s]28144 examples [00:24, 1186.19 examples/s]28270 examples [00:24, 1206.13 examples/s]28393 examples [00:24, 1210.62 examples/s]28518 examples [00:25, 1221.85 examples/s]28650 examples [00:25, 1247.55 examples/s]28786 examples [00:25, 1278.49 examples/s]28915 examples [00:25, 1250.07 examples/s]29041 examples [00:25, 1251.71 examples/s]29169 examples [00:25, 1259.50 examples/s]29298 examples [00:25, 1267.67 examples/s]29435 examples [00:25, 1295.39 examples/s]29569 examples [00:25, 1306.50 examples/s]29706 examples [00:25, 1324.61 examples/s]29839 examples [00:26, 1303.55 examples/s]29970 examples [00:26, 1262.26 examples/s]30097 examples [00:26, 1191.90 examples/s]30227 examples [00:26, 1220.55 examples/s]30351 examples [00:26, 1225.42 examples/s]30475 examples [00:26, 1205.52 examples/s]30597 examples [00:26, 1201.40 examples/s]30729 examples [00:26, 1232.76 examples/s]30853 examples [00:26, 1212.34 examples/s]30985 examples [00:27, 1240.60 examples/s]31115 examples [00:27, 1256.30 examples/s]31241 examples [00:27, 1254.76 examples/s]31377 examples [00:27, 1283.00 examples/s]31509 examples [00:27, 1293.25 examples/s]31649 examples [00:27, 1320.23 examples/s]31782 examples [00:27, 1277.53 examples/s]31911 examples [00:27, 1257.10 examples/s]32039 examples [00:27, 1260.69 examples/s]32166 examples [00:27, 1230.13 examples/s]32299 examples [00:28, 1256.05 examples/s]32428 examples [00:28, 1263.07 examples/s]32561 examples [00:28, 1281.57 examples/s]32696 examples [00:28, 1299.28 examples/s]32830 examples [00:28, 1309.94 examples/s]32969 examples [00:28, 1330.41 examples/s]33103 examples [00:28, 1262.90 examples/s]33234 examples [00:28, 1274.65 examples/s]33369 examples [00:28, 1293.72 examples/s]33499 examples [00:28, 1247.20 examples/s]33625 examples [00:29, 1184.81 examples/s]33747 examples [00:29, 1193.42 examples/s]33869 examples [00:29, 1199.90 examples/s]33992 examples [00:29, 1208.31 examples/s]34116 examples [00:29, 1215.27 examples/s]34248 examples [00:29, 1243.25 examples/s]34373 examples [00:29, 1230.57 examples/s]34502 examples [00:29, 1247.40 examples/s]34627 examples [00:29, 1240.25 examples/s]34752 examples [00:30, 1226.99 examples/s]34879 examples [00:30, 1238.76 examples/s]35009 examples [00:30, 1256.42 examples/s]35147 examples [00:30, 1290.78 examples/s]35277 examples [00:30, 1270.18 examples/s]35405 examples [00:30, 1268.71 examples/s]35537 examples [00:30, 1281.99 examples/s]35666 examples [00:30, 1260.87 examples/s]35793 examples [00:30, 1250.56 examples/s]35925 examples [00:30, 1270.55 examples/s]36053 examples [00:31, 1220.13 examples/s]36180 examples [00:31, 1232.73 examples/s]36304 examples [00:31, 1183.84 examples/s]36424 examples [00:31, 1169.45 examples/s]36542 examples [00:31, 1158.97 examples/s]36659 examples [00:31, 1153.99 examples/s]36775 examples [00:31, 1145.19 examples/s]36890 examples [00:31, 1137.23 examples/s]37010 examples [00:31, 1153.08 examples/s]37131 examples [00:31, 1167.57 examples/s]37248 examples [00:32, 1162.76 examples/s]37365 examples [00:32, 1114.89 examples/s]37484 examples [00:32, 1134.64 examples/s]37598 examples [00:32, 1126.30 examples/s]37711 examples [00:32, 1092.96 examples/s]37821 examples [00:32, 1091.05 examples/s]37931 examples [00:32, 1081.59 examples/s]38043 examples [00:32, 1092.38 examples/s]38153 examples [00:32, 1093.33 examples/s]38263 examples [00:33, 1074.18 examples/s]38371 examples [00:33, 1030.99 examples/s]38477 examples [00:33, 1037.95 examples/s]38588 examples [00:33, 1056.25 examples/s]38696 examples [00:33, 1062.94 examples/s]38805 examples [00:33, 1069.70 examples/s]38916 examples [00:33, 1080.96 examples/s]39025 examples [00:33, 1075.92 examples/s]39133 examples [00:33, 1054.66 examples/s]39244 examples [00:33, 1070.35 examples/s]39356 examples [00:34, 1083.53 examples/s]39469 examples [00:34, 1095.22 examples/s]39580 examples [00:34, 1097.69 examples/s]39693 examples [00:34, 1104.21 examples/s]39805 examples [00:34, 1107.13 examples/s]39916 examples [00:34, 1100.04 examples/s]40027 examples [00:34, 1022.69 examples/s]40135 examples [00:34, 1037.78 examples/s]40248 examples [00:34, 1063.47 examples/s]40361 examples [00:34, 1080.45 examples/s]40470 examples [00:35, 1065.17 examples/s]40583 examples [00:35, 1081.33 examples/s]40692 examples [00:35, 1083.13 examples/s]40804 examples [00:35, 1091.53 examples/s]40918 examples [00:35, 1104.67 examples/s]41033 examples [00:35, 1115.29 examples/s]41148 examples [00:35, 1123.89 examples/s]41261 examples [00:35, 1119.02 examples/s]41374 examples [00:35, 1121.30 examples/s]41487 examples [00:36, 1117.05 examples/s]41599 examples [00:36, 1088.87 examples/s]41711 examples [00:36, 1096.23 examples/s]41822 examples [00:36, 1099.40 examples/s]41937 examples [00:36, 1111.88 examples/s]42049 examples [00:36, 1112.24 examples/s]42161 examples [00:36, 1085.75 examples/s]42270 examples [00:36, 1073.04 examples/s]42381 examples [00:36, 1081.77 examples/s]42493 examples [00:36, 1091.62 examples/s]42604 examples [00:37, 1096.63 examples/s]42715 examples [00:37, 1098.73 examples/s]42825 examples [00:37, 1097.02 examples/s]42935 examples [00:37, 1092.59 examples/s]43047 examples [00:37, 1098.07 examples/s]43161 examples [00:37, 1109.53 examples/s]43276 examples [00:37, 1118.65 examples/s]43388 examples [00:37, 1113.90 examples/s]43500 examples [00:37, 1109.31 examples/s]43614 examples [00:37, 1117.14 examples/s]43726 examples [00:38, 1109.04 examples/s]43840 examples [00:38, 1115.52 examples/s]43952 examples [00:38, 1108.52 examples/s]44063 examples [00:38, 1103.04 examples/s]44177 examples [00:38, 1113.49 examples/s]44293 examples [00:38, 1126.05 examples/s]44406 examples [00:38, 1105.80 examples/s]44521 examples [00:38, 1116.75 examples/s]44633 examples [00:38, 1101.43 examples/s]44746 examples [00:38, 1108.97 examples/s]44858 examples [00:39, 1110.15 examples/s]44970 examples [00:39, 1073.09 examples/s]45084 examples [00:39, 1089.74 examples/s]45197 examples [00:39, 1100.59 examples/s]45308 examples [00:39, 1082.73 examples/s]45422 examples [00:39, 1097.60 examples/s]45535 examples [00:39, 1104.43 examples/s]45646 examples [00:39, 1101.58 examples/s]45757 examples [00:39, 1071.42 examples/s]45867 examples [00:39, 1078.73 examples/s]45978 examples [00:40, 1085.21 examples/s]46088 examples [00:40, 1087.29 examples/s]46198 examples [00:40, 1090.33 examples/s]46309 examples [00:40, 1095.25 examples/s]46421 examples [00:40, 1100.06 examples/s]46534 examples [00:40, 1107.61 examples/s]46647 examples [00:40, 1112.87 examples/s]46759 examples [00:40, 1111.89 examples/s]46871 examples [00:40, 1105.98 examples/s]46985 examples [00:41, 1114.76 examples/s]47098 examples [00:41, 1117.20 examples/s]47211 examples [00:41, 1120.85 examples/s]47343 examples [00:41, 1172.54 examples/s]47476 examples [00:41, 1214.22 examples/s]47599 examples [00:41, 1205.30 examples/s]47721 examples [00:41, 1204.04 examples/s]47842 examples [00:41, 1200.29 examples/s]47970 examples [00:41, 1222.84 examples/s]48101 examples [00:41, 1247.20 examples/s]48238 examples [00:42, 1281.45 examples/s]48367 examples [00:42, 1282.74 examples/s]48502 examples [00:42, 1300.28 examples/s]48633 examples [00:42, 1231.77 examples/s]48758 examples [00:42, 1201.68 examples/s]48884 examples [00:42, 1217.38 examples/s]49017 examples [00:42, 1248.22 examples/s]49154 examples [00:42, 1282.30 examples/s]49283 examples [00:42, 1277.12 examples/s]49416 examples [00:42, 1290.48 examples/s]49546 examples [00:43, 1285.06 examples/s]49676 examples [00:43, 1288.66 examples/s]49812 examples [00:43, 1306.18 examples/s]49943 examples [00:43, 1269.35 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 19%|█▉        | 9472/50000 [00:00<00:00, 94719.19 examples/s] 48%|████▊     | 24166/50000 [00:00<00:00, 106022.09 examples/s] 79%|███████▉  | 39536/50000 [00:00<00:00, 116899.81 examples/s]                                                                0 examples [00:00, ? examples/s]110 examples [00:00, 1097.96 examples/s]227 examples [00:00, 1117.75 examples/s]346 examples [00:00, 1138.01 examples/s]470 examples [00:00, 1164.33 examples/s]601 examples [00:00, 1204.05 examples/s]732 examples [00:00, 1232.72 examples/s]862 examples [00:00, 1249.77 examples/s]992 examples [00:00, 1262.06 examples/s]1120 examples [00:00, 1265.02 examples/s]1252 examples [00:01, 1280.93 examples/s]1378 examples [00:01, 1020.54 examples/s]1511 examples [00:01, 1095.53 examples/s]1635 examples [00:01, 1133.57 examples/s]1764 examples [00:01, 1175.50 examples/s]1886 examples [00:01, 1139.02 examples/s]2006 examples [00:01, 1156.49 examples/s]2124 examples [00:01, 1162.29 examples/s]2257 examples [00:01, 1205.08 examples/s]2383 examples [00:02, 1220.50 examples/s]2507 examples [00:02, 1209.82 examples/s]2630 examples [00:02, 1215.45 examples/s]2761 examples [00:02, 1242.11 examples/s]2886 examples [00:02, 1243.05 examples/s]3011 examples [00:02, 1226.92 examples/s]3134 examples [00:02, 1175.10 examples/s]3253 examples [00:02, 1157.12 examples/s]3376 examples [00:02, 1175.85 examples/s]3504 examples [00:02, 1203.81 examples/s]3633 examples [00:03, 1227.53 examples/s]3768 examples [00:03, 1260.87 examples/s]3895 examples [00:03, 1247.02 examples/s]4022 examples [00:03, 1252.76 examples/s]4159 examples [00:03, 1285.74 examples/s]4290 examples [00:03, 1290.83 examples/s]4420 examples [00:03, 1289.88 examples/s]4551 examples [00:03, 1293.52 examples/s]4681 examples [00:03, 1279.75 examples/s]4810 examples [00:03, 1273.58 examples/s]4938 examples [00:04, 1243.13 examples/s]5066 examples [00:04, 1251.69 examples/s]5192 examples [00:04, 1236.45 examples/s]5316 examples [00:04, 1233.10 examples/s]5446 examples [00:04, 1250.08 examples/s]5572 examples [00:04, 1228.65 examples/s]5696 examples [00:04, 1213.65 examples/s]5827 examples [00:04, 1240.17 examples/s]5952 examples [00:04, 1215.06 examples/s]6078 examples [00:04, 1227.40 examples/s]6209 examples [00:05, 1250.07 examples/s]6346 examples [00:05, 1282.62 examples/s]6475 examples [00:05, 1243.12 examples/s]6600 examples [00:05, 1177.32 examples/s]6736 examples [00:05, 1226.02 examples/s]6870 examples [00:05, 1254.92 examples/s]7005 examples [00:05, 1281.04 examples/s]7134 examples [00:05, 1281.38 examples/s]7270 examples [00:05, 1301.18 examples/s]7401 examples [00:06, 1277.31 examples/s]7531 examples [00:06, 1281.49 examples/s]7660 examples [00:06, 1274.18 examples/s]7788 examples [00:06, 1269.28 examples/s]7924 examples [00:06, 1293.81 examples/s]8054 examples [00:06, 1274.21 examples/s]8182 examples [00:06, 1268.30 examples/s]8309 examples [00:06, 1245.12 examples/s]8434 examples [00:06, 1218.64 examples/s]8570 examples [00:06, 1257.37 examples/s]8703 examples [00:07, 1277.65 examples/s]8834 examples [00:07, 1285.87 examples/s]8963 examples [00:07, 1277.00 examples/s]9099 examples [00:07, 1300.38 examples/s]9236 examples [00:07, 1318.86 examples/s]9369 examples [00:07, 1291.55 examples/s]9499 examples [00:07, 1254.99 examples/s]9631 examples [00:07, 1272.93 examples/s]9759 examples [00:07, 1270.54 examples/s]9895 examples [00:07, 1295.75 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUYHCZT/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUYHCZT/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f31b58c1950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f31b58c1950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f31b58c1950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f32012ae400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f315061cdd8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f31b58c1950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f31b58c1950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f31b58c1950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f31b2ee5b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f31b2ee5908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f312d395268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f312d395268> 

  function with postional parmater data_info <function split_train_valid at 0x7f312d395268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f312d395378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f312d395378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f312d395378> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<50:45:12, 4.72kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<35:45:37, 6.70kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<25:04:58, 9.55kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<17:33:30, 13.6kB/s].vector_cache/glove.6B.zip:   0%|          | 3.39M/862M [00:02<12:15:33, 19.5kB/s].vector_cache/glove.6B.zip:   1%|          | 8.59M/862M [00:02<8:31:51, 27.8kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:02<5:56:46, 39.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.5M/862M [00:02<4:08:21, 56.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:02<2:53:10, 80.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:02<2:00:32, 116kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.5M/862M [00:02<1:24:06, 165kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:03<58:33, 235kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 39.3M/862M [00:03<40:57, 335kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.3M/862M [00:03<28:35, 477kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.1M/862M [00:03<20:01, 677kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:03<14:32, 928kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<12:02, 1.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:06<12:05, 1.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:06<09:13, 1.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:06<06:40, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<08:33, 1.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:08<07:32, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:08<05:36, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<06:43, 1.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:10<06:09, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:10<04:40, 2.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:11<06:11, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<05:40, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:12<04:17, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<06:07, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<06:58, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:14<05:32, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<05:59, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:15<05:33, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:16<04:12, 3.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:17<06:00, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:17<05:32, 2.35MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:17<04:11, 3.09MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<05:59, 2.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<05:30, 2.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<04:07, 3.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<05:57, 2.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<05:17, 2.44MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:21<04:01, 3.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:23<05:50, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:23<05:24, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<04:06, 3.11MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:25<05:53, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:25<06:43, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:25<05:20, 2.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:46, 2.19MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:21, 2.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:04, 3.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:46, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<06:36, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<05:10, 2.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<03:46, 3.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<04:41, 2.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<9:18:04, 22.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<6:30:57, 32.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<4:32:45, 45.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<3:24:57, 60.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<2:25:59, 85.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<1:42:37, 121kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<1:11:47, 173kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<56:37, 219kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:35<40:54, 303kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<28:51, 429kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<23:02, 535kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<17:22, 710kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<12:26, 988kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<11:35, 1.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<09:22, 1.31MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<06:48, 1.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<07:39, 1.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<06:36, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<04:55, 2.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<06:18, 1.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<06:54, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<05:24, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<03:54, 3.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<13:58, 863kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<11:00, 1.10MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<07:56, 1.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<08:22, 1.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<08:12, 1.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<06:15, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<04:31, 2.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<09:00, 1.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<07:32, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<05:34, 2.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<06:38, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<05:53, 2.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<04:24, 2.68MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:50, 2.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<05:17, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<03:57, 2.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:31, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<05:03, 2.32MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<03:50, 3.05MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<05:24, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<04:58, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<03:46, 3.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:20, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:05, 1.90MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<04:46, 2.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<03:27, 3.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<13:02, 882kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<10:19, 1.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:30, 1.53MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:55, 1.44MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:52, 1.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:03<06:05, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:22, 2.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<1:25:03, 134kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<1:00:30, 188kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<42:29, 267kB/s]  .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<29:49, 379kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<50:35, 223kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<37:42, 300kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<26:56, 419kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<20:35, 546kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<15:34, 721kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<11:07, 1.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<10:21, 1.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<09:31, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:10, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<05:07, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<34:38, 320kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<25:23, 437kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<17:58, 616kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<15:18, 720kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<11:14, 979kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<07:58, 1.38MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<13:00, 843kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<10:13, 1.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:25, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:45, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<8:26:58, 21.5kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<5:55:04, 30.6kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<4:07:33, 43.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<3:07:10, 57.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<2:13:11, 81.3kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<1:33:39, 115kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<1:07:00, 161kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<48:01, 224kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<33:46, 318kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<26:02, 411kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<19:18, 553kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<13:42, 777kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<12:06, 878kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<10:37, 999kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<07:58, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<07:16, 1.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<06:09, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:34, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<05:38, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<06:10, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:46, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<03:28, 3.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:32<07:17, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<06:11, 1.68MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:35, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<05:37, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<06:02, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<04:40, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:23, 3.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:36<08:46, 1.17MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<07:11, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:17, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<06:05, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<05:18, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<03:56, 2.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:09, 1.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<05:40, 1.79MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:23, 2.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<03:13, 3.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:30, 1.55MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<05:36, 1.80MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:08, 2.42MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:14, 1.91MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<04:41, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<03:29, 2.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:48, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<05:22, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<04:15, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<04:34, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:13, 2.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<03:11, 3.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<04:32, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:10, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<04:05, 2.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<02:58, 3.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<9:23:38, 17.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<6:35:16, 24.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:52<4:36:12, 35.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<3:14:53, 49.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<2:18:20, 69.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<1:37:08, 99.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<1:07:53, 142kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<51:54, 185kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<37:17, 257kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<26:14, 364kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<20:32, 464kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<16:18, 583kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:50, 803kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<09:47, 966kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<07:49, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:42, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<06:11, 1.52MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<06:20, 1.48MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:50, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<03:32, 2.64MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<05:43, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<04:58, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<03:42, 2.50MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:05, 3.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<7:08:22, 21.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<4:59:57, 30.8kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<3:28:58, 44.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<2:38:02, 58.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<1:52:27, 81.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<1:19:04, 116kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<55:08, 165kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<9:25:38, 16.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<6:36:37, 23.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<4:37:04, 32.8kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<3:15:18, 46.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<2:17:32, 65.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<1:36:13, 93.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<1:09:13, 130kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<50:16, 179kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<35:31, 252kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<24:51, 359kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<23:30, 379kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<17:22, 513kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<12:18, 721kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<10:39, 829kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<08:21, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:03, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<06:18, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<05:18, 1.65MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:55, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<04:48, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<04:14, 2.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:08, 2.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<04:15, 2.03MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<04:44, 1.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:45, 2.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:00, 2.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<02:45, 3.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:55, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<04:28, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<03:30, 2.41MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<02:32, 3.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<17:16, 488kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<12:56, 651kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<09:13, 911kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<08:23, 996kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<06:43, 1.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:54, 1.70MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:23, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:36, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<03:25, 2.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:21, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:39, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<03:40, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<02:39, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<7:51:14, 17.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<5:30:27, 24.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<3:50:46, 35.2kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<2:42:43, 49.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<1:54:37, 70.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<1:20:11, 100kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<57:46, 139kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<42:03, 191kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<29:44, 269kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<20:49, 382kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<19:02, 418kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<14:07, 562kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<10:03, 787kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<08:51, 890kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:48, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:47, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:10, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<05:41, 1.37MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:47, 1.63MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:32, 2.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:17, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:35, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<03:36, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:44, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:25, 2.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:35, 2.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<02:13, 3.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<5:48:28, 21.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<4:03:55, 31.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<2:49:43, 44.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<2:08:32, 58.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<1:31:28, 82.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<1:04:17, 117kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<45:55, 163kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<33:44, 221kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<23:58, 311kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<16:46, 442kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:58<33:56, 218kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<23:51, 309kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<18:11, 403kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<14:09, 518kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<10:11, 718kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<07:14, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<07:39, 949kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<06:07, 1.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:26, 1.63MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<04:42, 1.53MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<04:46, 1.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<03:39, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:38, 2.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:06<05:32, 1.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<04:28, 1.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:18, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:58, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:29, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<02:37, 2.68MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<03:29, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<03:09, 2.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:23, 2.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<03:17, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<03:01, 2.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<02:16, 3.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<03:12, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<03:38, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<02:53, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:06, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<02:52, 2.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<02:09, 3.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:04, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<02:49, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<02:08, 3.12MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:03, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<02:49, 2.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<02:08, 3.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:02, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:47, 2.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<02:05, 3.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:00, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:40, 2.43MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<02:01, 3.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:56, 2.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:43, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:03, 3.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:56, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:42, 2.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<02:02, 3.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<02:54, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:19, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:36, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<01:53, 3.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<05:11, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<04:16, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:08, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<03:37, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<03:10, 1.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:21, 2.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<03:05, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:47, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:05, 2.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:53, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:14, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:34, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:39<02:45, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:33, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<01:56, 3.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<01:39, 3.56MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<4:33:29, 21.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<3:11:18, 30.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<2:13:02, 43.9kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<1:36:23, 60.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:43<1:08:38, 84.9kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<48:13, 120kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<34:23, 167kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<24:38, 233kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<17:19, 330kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<13:22, 426kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<10:30, 541kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<07:35, 748kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<05:21, 1.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<06:02, 931kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<04:47, 1.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:27, 1.61MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<03:43, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<03:43, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<02:51, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:02, 2.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<11:32, 475kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<08:38, 634kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<06:09, 886kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:55<05:33, 975kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<04:26, 1.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:13, 1.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:57<03:30, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:59, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:13, 2.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:59<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:59<07:11, 735kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<05:09, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:01<04:32, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<03:46, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:45, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<03:04, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<03:14, 1.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:32, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<01:49, 2.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:05<16:29, 308kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:05<12:03, 420kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<08:31, 591kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<07:06, 705kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<05:28, 913kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:54, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:54, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:09<03:41, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<02:47, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<01:59, 2.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<04:42, 1.03MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<03:47, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:46, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<03:02, 1.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:13<03:09, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:25, 1.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:43, 2.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<09:10, 515kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<06:54, 684kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<04:55, 953kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<04:30, 1.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<04:06, 1.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:17<03:06, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:11, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<4:27:34, 17.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<3:07:26, 24.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<2:10:34, 34.9kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<1:31:45, 49.3kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<1:05:06, 69.4kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<45:38, 98.8kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<31:40, 141kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<27:51, 160kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<19:56, 223kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<13:58, 317kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<10:43, 409kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<08:23, 522kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<06:05, 718kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:15, 1.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<4:11:36, 17.2kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<2:56:18, 24.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<2:02:46, 34.9kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<1:26:13, 49.3kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<1:01:10, 69.4kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<42:54, 98.7kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<30:20, 138kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<21:38, 193kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<15:09, 274kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<11:28, 358kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<08:52, 463kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<06:22, 643kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<04:27, 909kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<05:34, 724kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<04:19, 934kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:06, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<03:05, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:58, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:16, 1.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<02:12, 1.77MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<01:56, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:26, 2.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<01:54, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<01:42, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:17, 2.96MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<01:47, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<02:00, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:35, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:08, 3.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<3:34:24, 17.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<2:30:12, 24.6kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<1:44:29, 35.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<1:13:17, 49.6kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<51:59, 69.8kB/s]  .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<36:27, 99.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<25:12, 141kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<43:05, 82.7kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<30:28, 117kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<21:16, 166kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<15:33, 225kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<11:13, 311kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<07:53, 439kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<06:16, 547kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<04:43, 723kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<03:22, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<03:07, 1.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<02:31, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:49, 1.82MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<02:02, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<02:06, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:37, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<01:38, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<01:29, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:06, 2.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:30, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<01:22, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:01, 3.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:26, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<01:38, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:16, 2.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<00:55, 3.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:49, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:35, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:09, 2.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:30, 1.96MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:38, 1.79MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<01:16, 2.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<00:54, 3.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<05:22, 536kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<03:52, 740kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:41, 1.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<09:22, 300kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<07:07, 394kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<05:05, 548kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:32, 776kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<2:40:40, 17.1kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<1:52:29, 24.3kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<1:18:04, 34.7kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<54:32, 49.0kB/s]  .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<38:42, 69.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<27:05, 98.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<18:38, 140kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<29:44, 87.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<21:01, 124kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<14:38, 176kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<10:42, 237kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<07:59, 317kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<05:42, 442kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:56, 626kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<20:22, 121kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<14:28, 170kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<10:05, 241kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<07:31, 319kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<05:29, 436kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<03:51, 612kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<03:12, 726kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<02:43, 855kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:59, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:24, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<02:16, 994kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:49, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:18, 1.70MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:25, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:26, 1.52MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:06, 1.97MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:06, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:59, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:44, 2.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:59, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:06, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:51, 2.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:37, 3.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:32, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:16, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:56, 2.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:05, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<00:57, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:42, 2.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:55, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:49, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:37, 2.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:50, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:57, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:44, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:31, 3.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<01:56, 881kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<01:31, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:05, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:08, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:07, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:51, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:36, 2.61MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:01, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:52, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:38, 2.40MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:47, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:41, 2.19MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:30, 2.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<00:41, 2.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<00:46, 1.85MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:36, 2.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:25, 3.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<10:04, 136kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<07:09, 191kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<04:56, 271kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<03:40, 354kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<02:41, 481kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:52, 676kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:34, 786kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<01:20, 914kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:59, 1.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:40, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:23, 838kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:03, 1.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:45, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:31, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<04:47, 229kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<03:26, 317kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<02:22, 447kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:51, 555kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:30, 682kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<01:04, 935kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:44, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:01, 937kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:48, 1.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:34, 1.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:35, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:30, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:21, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:26, 1.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:23, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:16, 2.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:22, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:19, 2.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:14, 2.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:19, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:21, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:17, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:16, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:15, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:11, 3.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:15, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:17, 1.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:13, 2.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:09, 3.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:17, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:15, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:10, 2.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:12, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:11, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:07, 2.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:09, 2.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:08, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:06, 3.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:07, 2.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:06, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:04, 3.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:06, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:04, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:03, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:03, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:01, 3.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:01, 2.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:02, 1.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.41MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 3.32MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<61:14:32,  1.81it/s]  0%|          | 972/400000 [00:00<42:46:08,  2.59it/s]  0%|          | 1866/400000 [00:00<29:52:29,  3.70it/s]  1%|          | 2773/400000 [00:00<20:52:06,  5.29it/s]  1%|          | 3734/400000 [00:00<14:34:33,  7.55it/s]  1%|          | 4771/400000 [00:01<10:10:46, 10.78it/s]  1%|▏         | 5770/400000 [00:01<7:06:39, 15.40it/s]   2%|▏         | 6708/400000 [00:01<4:58:09, 21.98it/s]  2%|▏         | 7595/400000 [00:01<3:28:28, 31.37it/s]  2%|▏         | 8510/400000 [00:01<2:25:48, 44.75it/s]  2%|▏         | 9485/400000 [00:01<1:42:00, 63.80it/s]  3%|▎         | 10545/400000 [00:01<1:11:23, 90.91it/s]  3%|▎         | 11625/400000 [00:01<50:01, 129.41it/s]   3%|▎         | 12620/400000 [00:01<35:08, 183.77it/s]  3%|▎         | 13593/400000 [00:01<24:44, 260.36it/s]  4%|▎         | 14559/400000 [00:02<17:28, 367.62it/s]  4%|▍         | 15654/400000 [00:02<12:22, 517.71it/s]  4%|▍         | 16710/400000 [00:02<08:49, 724.36it/s]  4%|▍         | 17728/400000 [00:02<06:21, 1002.04it/s]  5%|▍         | 18725/400000 [00:02<04:38, 1370.88it/s]  5%|▍         | 19715/400000 [00:02<03:26, 1845.63it/s]  5%|▌         | 20696/400000 [00:02<02:36, 2430.47it/s]  5%|▌         | 21663/400000 [00:02<02:01, 3113.14it/s]  6%|▌         | 22611/400000 [00:02<01:37, 3869.94it/s]  6%|▌         | 23543/400000 [00:03<01:20, 4676.73it/s]  6%|▌         | 24469/400000 [00:03<01:08, 5459.39it/s]  6%|▋         | 25423/400000 [00:03<00:59, 6262.84it/s]  7%|▋         | 26434/400000 [00:03<00:52, 7069.16it/s]  7%|▋         | 27387/400000 [00:03<00:48, 7639.18it/s]  7%|▋         | 28337/400000 [00:03<00:46, 8045.89it/s]  7%|▋         | 29285/400000 [00:03<00:43, 8428.00it/s]  8%|▊         | 30350/400000 [00:03<00:41, 8989.41it/s]  8%|▊         | 31329/400000 [00:03<00:40, 9203.22it/s]  8%|▊         | 32307/400000 [00:03<00:40, 8992.17it/s]  8%|▊         | 33274/400000 [00:04<00:39, 9184.98it/s]  9%|▊         | 34259/400000 [00:04<00:39, 9374.14it/s]  9%|▉         | 35242/400000 [00:04<00:38, 9504.80it/s]  9%|▉         | 36241/400000 [00:04<00:37, 9644.93it/s]  9%|▉         | 37217/400000 [00:04<00:38, 9440.61it/s] 10%|▉         | 38171/400000 [00:04<00:39, 9151.00it/s] 10%|▉         | 39167/400000 [00:04<00:38, 9378.93it/s] 10%|█         | 40171/400000 [00:04<00:37, 9567.01it/s] 10%|█         | 41281/400000 [00:04<00:35, 9980.51it/s] 11%|█         | 42288/400000 [00:04<00:37, 9551.30it/s] 11%|█         | 43282/400000 [00:05<00:36, 9664.61it/s] 11%|█         | 44316/400000 [00:05<00:36, 9857.30it/s] 11%|█▏        | 45359/400000 [00:05<00:35, 10020.97it/s] 12%|█▏        | 46367/400000 [00:05<00:35, 9874.82it/s]  12%|█▏        | 47359/400000 [00:05<00:36, 9585.17it/s] 12%|█▏        | 48329/400000 [00:05<00:36, 9618.63it/s] 12%|█▏        | 49364/400000 [00:05<00:35, 9825.28it/s] 13%|█▎        | 50415/400000 [00:05<00:34, 10019.84it/s] 13%|█▎        | 51421/400000 [00:05<00:34, 9981.18it/s]  13%|█▎        | 52422/400000 [00:05<00:35, 9903.88it/s] 13%|█▎        | 53415/400000 [00:06<00:36, 9611.98it/s] 14%|█▎        | 54411/400000 [00:06<00:35, 9711.25it/s] 14%|█▍        | 55420/400000 [00:06<00:35, 9820.72it/s] 14%|█▍        | 56405/400000 [00:06<00:35, 9738.03it/s] 14%|█▍        | 57381/400000 [00:06<00:35, 9603.76it/s] 15%|█▍        | 58391/400000 [00:06<00:35, 9745.66it/s] 15%|█▍        | 59428/400000 [00:06<00:34, 9924.20it/s] 15%|█▌        | 60423/400000 [00:06<00:34, 9713.96it/s] 15%|█▌        | 61397/400000 [00:06<00:36, 9394.56it/s] 16%|█▌        | 62341/400000 [00:07<00:36, 9224.97it/s] 16%|█▌        | 63267/400000 [00:07<00:36, 9209.86it/s] 16%|█▌        | 64208/400000 [00:07<00:36, 9267.55it/s] 16%|█▋        | 65147/400000 [00:07<00:36, 9301.38it/s] 17%|█▋        | 66101/400000 [00:07<00:35, 9370.68it/s] 17%|█▋        | 67040/400000 [00:07<00:35, 9375.68it/s] 17%|█▋        | 67979/400000 [00:07<00:35, 9310.90it/s] 17%|█▋        | 68911/400000 [00:07<00:36, 9078.96it/s] 17%|█▋        | 69821/400000 [00:07<00:36, 8967.33it/s] 18%|█▊        | 70741/400000 [00:07<00:36, 9034.44it/s] 18%|█▊        | 71835/400000 [00:08<00:34, 9531.09it/s] 18%|█▊        | 72797/400000 [00:08<00:34, 9455.42it/s] 18%|█▊        | 73895/400000 [00:08<00:33, 9865.45it/s] 19%|█▊        | 74891/400000 [00:08<00:33, 9635.07it/s] 19%|█▉        | 75944/400000 [00:08<00:32, 9884.52it/s] 19%|█▉        | 76940/400000 [00:08<00:33, 9775.83it/s] 19%|█▉        | 77923/400000 [00:08<00:33, 9706.29it/s] 20%|█▉        | 78899/400000 [00:08<00:33, 9719.54it/s] 20%|█▉        | 79879/400000 [00:08<00:32, 9740.80it/s] 20%|██        | 80924/400000 [00:08<00:32, 9942.33it/s] 20%|██        | 81988/400000 [00:09<00:31, 10141.24it/s] 21%|██        | 83005/400000 [00:09<00:31, 10104.21it/s] 21%|██        | 84018/400000 [00:09<00:31, 10016.89it/s] 21%|██▏       | 85022/400000 [00:09<00:31, 9844.35it/s]  22%|██▏       | 86009/400000 [00:09<00:32, 9635.42it/s] 22%|██▏       | 86975/400000 [00:09<00:32, 9553.65it/s] 22%|██▏       | 87951/400000 [00:09<00:32, 9612.81it/s] 22%|██▏       | 88936/400000 [00:09<00:32, 9681.55it/s] 22%|██▏       | 89919/400000 [00:09<00:31, 9723.25it/s] 23%|██▎       | 90905/400000 [00:09<00:31, 9762.33it/s] 23%|██▎       | 91882/400000 [00:10<00:31, 9721.68it/s] 23%|██▎       | 92855/400000 [00:10<00:32, 9587.26it/s] 23%|██▎       | 93847/400000 [00:10<00:31, 9683.26it/s] 24%|██▎       | 94817/400000 [00:10<00:32, 9499.42it/s] 24%|██▍       | 95796/400000 [00:10<00:31, 9583.43it/s] 24%|██▍       | 96796/400000 [00:10<00:31, 9702.35it/s] 24%|██▍       | 97771/400000 [00:10<00:31, 9713.68it/s] 25%|██▍       | 98744/400000 [00:10<00:31, 9715.93it/s] 25%|██▍       | 99717/400000 [00:10<00:31, 9644.63it/s] 25%|██▌       | 100711/400000 [00:11<00:30, 9730.36it/s] 25%|██▌       | 101685/400000 [00:11<00:31, 9576.03it/s] 26%|██▌       | 102644/400000 [00:11<00:31, 9434.08it/s] 26%|██▌       | 103593/400000 [00:11<00:31, 9449.56it/s] 26%|██▌       | 104555/400000 [00:11<00:31, 9497.08it/s] 26%|██▋       | 105506/400000 [00:11<00:31, 9265.32it/s] 27%|██▋       | 106454/400000 [00:11<00:31, 9327.26it/s] 27%|██▋       | 107388/400000 [00:11<00:31, 9313.39it/s] 27%|██▋       | 108379/400000 [00:11<00:30, 9484.14it/s] 27%|██▋       | 109329/400000 [00:11<00:31, 9143.22it/s] 28%|██▊       | 110289/400000 [00:12<00:31, 9275.50it/s] 28%|██▊       | 111293/400000 [00:12<00:30, 9491.05it/s] 28%|██▊       | 112246/400000 [00:12<00:30, 9485.42it/s] 28%|██▊       | 113262/400000 [00:12<00:29, 9677.99it/s] 29%|██▊       | 114233/400000 [00:12<00:30, 9507.76it/s] 29%|██▉       | 115227/400000 [00:12<00:29, 9631.43it/s] 29%|██▉       | 116194/400000 [00:12<00:29, 9640.80it/s] 29%|██▉       | 117160/400000 [00:12<00:29, 9482.99it/s] 30%|██▉       | 118114/400000 [00:12<00:29, 9498.04it/s] 30%|██▉       | 119065/400000 [00:12<00:30, 9116.59it/s] 30%|███       | 120013/400000 [00:13<00:30, 9220.07it/s] 30%|███       | 120973/400000 [00:13<00:29, 9330.63it/s] 30%|███       | 121928/400000 [00:13<00:29, 9393.48it/s] 31%|███       | 122888/400000 [00:13<00:29, 9453.35it/s] 31%|███       | 123837/400000 [00:13<00:29, 9462.07it/s] 31%|███       | 124837/400000 [00:13<00:28, 9614.50it/s] 31%|███▏      | 125800/400000 [00:13<00:28, 9508.09it/s] 32%|███▏      | 126752/400000 [00:13<00:29, 9402.03it/s] 32%|███▏      | 127716/400000 [00:13<00:28, 9472.16it/s] 32%|███▏      | 128665/400000 [00:13<00:29, 9277.88it/s] 32%|███▏      | 129595/400000 [00:14<00:29, 9139.63it/s] 33%|███▎      | 130511/400000 [00:14<00:29, 9036.95it/s] 33%|███▎      | 131417/400000 [00:14<00:30, 8925.68it/s] 33%|███▎      | 132311/400000 [00:14<00:30, 8903.68it/s] 33%|███▎      | 133203/400000 [00:14<00:30, 8745.59it/s] 34%|███▎      | 134107/400000 [00:14<00:30, 8831.83it/s] 34%|███▍      | 135050/400000 [00:14<00:29, 9002.45it/s] 34%|███▍      | 135959/400000 [00:14<00:29, 9026.33it/s] 34%|███▍      | 136869/400000 [00:14<00:29, 9046.42it/s] 34%|███▍      | 137775/400000 [00:14<00:29, 9035.88it/s] 35%|███▍      | 138764/400000 [00:15<00:28, 9274.81it/s] 35%|███▍      | 139694/400000 [00:15<00:28, 9191.15it/s] 35%|███▌      | 140615/400000 [00:15<00:28, 9123.46it/s] 35%|███▌      | 141529/400000 [00:15<00:28, 8974.07it/s] 36%|███▌      | 142428/400000 [00:15<00:29, 8853.91it/s] 36%|███▌      | 143315/400000 [00:15<00:29, 8792.89it/s] 36%|███▌      | 144196/400000 [00:15<00:29, 8775.95it/s] 36%|███▋      | 145075/400000 [00:15<00:29, 8734.05it/s] 36%|███▋      | 145949/400000 [00:15<00:29, 8690.52it/s] 37%|███▋      | 146823/400000 [00:16<00:29, 8704.68it/s] 37%|███▋      | 147698/400000 [00:16<00:28, 8717.77it/s] 37%|███▋      | 148571/400000 [00:16<00:29, 8645.03it/s] 37%|███▋      | 149436/400000 [00:16<00:29, 8638.95it/s] 38%|███▊      | 150301/400000 [00:16<00:29, 8531.29it/s] 38%|███▊      | 151199/400000 [00:16<00:28, 8658.94it/s] 38%|███▊      | 152146/400000 [00:16<00:27, 8886.14it/s] 38%|███▊      | 153037/400000 [00:16<00:27, 8870.82it/s] 38%|███▊      | 153926/400000 [00:16<00:27, 8827.53it/s] 39%|███▊      | 154810/400000 [00:16<00:28, 8749.81it/s] 39%|███▉      | 155686/400000 [00:17<00:28, 8704.70it/s] 39%|███▉      | 156558/400000 [00:17<00:28, 8652.37it/s] 39%|███▉      | 157424/400000 [00:17<00:28, 8639.76it/s] 40%|███▉      | 158322/400000 [00:17<00:27, 8737.72it/s] 40%|███▉      | 159197/400000 [00:17<00:27, 8714.72it/s] 40%|████      | 160078/400000 [00:17<00:27, 8740.41it/s] 40%|████      | 160953/400000 [00:17<00:27, 8732.01it/s] 40%|████      | 161835/400000 [00:17<00:27, 8755.99it/s] 41%|████      | 162768/400000 [00:17<00:26, 8920.34it/s] 41%|████      | 163661/400000 [00:17<00:26, 8795.41it/s] 41%|████      | 164542/400000 [00:18<00:26, 8739.25it/s] 41%|████▏     | 165422/400000 [00:18<00:26, 8756.04it/s] 42%|████▏     | 166299/400000 [00:18<00:26, 8747.65it/s] 42%|████▏     | 167175/400000 [00:18<00:26, 8723.23it/s] 42%|████▏     | 168048/400000 [00:18<00:26, 8714.39it/s] 42%|████▏     | 168920/400000 [00:18<00:26, 8695.76it/s] 42%|████▏     | 169790/400000 [00:18<00:26, 8681.42it/s] 43%|████▎     | 170668/400000 [00:18<00:26, 8708.30it/s] 43%|████▎     | 171539/400000 [00:18<00:26, 8650.79it/s] 43%|████▎     | 172425/400000 [00:18<00:26, 8710.80it/s] 43%|████▎     | 173306/400000 [00:19<00:25, 8739.08it/s] 44%|████▎     | 174181/400000 [00:19<00:25, 8728.78it/s] 44%|████▍     | 175066/400000 [00:19<00:25, 8763.71it/s] 44%|████▍     | 175943/400000 [00:19<00:25, 8711.86it/s] 44%|████▍     | 176819/400000 [00:19<00:25, 8725.92it/s] 44%|████▍     | 177692/400000 [00:19<00:25, 8704.92it/s] 45%|████▍     | 178563/400000 [00:19<00:25, 8668.26it/s] 45%|████▍     | 179439/400000 [00:19<00:25, 8694.57it/s] 45%|████▌     | 180309/400000 [00:19<00:25, 8683.25it/s] 45%|████▌     | 181204/400000 [00:19<00:24, 8761.45it/s] 46%|████▌     | 182090/400000 [00:20<00:24, 8788.99it/s] 46%|████▌     | 182970/400000 [00:20<00:24, 8752.95it/s] 46%|████▌     | 183856/400000 [00:20<00:24, 8782.35it/s] 46%|████▌     | 184735/400000 [00:20<00:25, 8598.97it/s] 46%|████▋     | 185596/400000 [00:20<00:25, 8513.41it/s] 47%|████▋     | 186478/400000 [00:20<00:24, 8602.31it/s] 47%|████▋     | 187347/400000 [00:20<00:24, 8626.73it/s] 47%|████▋     | 188236/400000 [00:20<00:24, 8702.06it/s] 47%|████▋     | 189107/400000 [00:20<00:24, 8667.97it/s] 47%|████▋     | 189981/400000 [00:20<00:24, 8689.07it/s] 48%|████▊     | 190852/400000 [00:21<00:24, 8693.65it/s] 48%|████▊     | 191722/400000 [00:21<00:23, 8691.86it/s] 48%|████▊     | 192648/400000 [00:21<00:23, 8852.25it/s] 48%|████▊     | 193535/400000 [00:21<00:23, 8634.39it/s] 49%|████▊     | 194401/400000 [00:21<00:23, 8641.87it/s] 49%|████▉     | 195267/400000 [00:21<00:23, 8610.29it/s] 49%|████▉     | 196181/400000 [00:21<00:23, 8761.26it/s] 49%|████▉     | 197059/400000 [00:21<00:23, 8764.83it/s] 49%|████▉     | 197937/400000 [00:21<00:23, 8732.64it/s] 50%|████▉     | 198829/400000 [00:21<00:22, 8786.46it/s] 50%|████▉     | 199709/400000 [00:22<00:22, 8766.14it/s] 50%|█████     | 200586/400000 [00:22<00:22, 8760.70it/s] 50%|█████     | 201497/400000 [00:22<00:22, 8860.99it/s] 51%|█████     | 202384/400000 [00:22<00:22, 8839.84it/s] 51%|█████     | 203269/400000 [00:22<00:22, 8820.13it/s] 51%|█████     | 204152/400000 [00:22<00:22, 8811.86it/s] 51%|█████▏    | 205034/400000 [00:22<00:22, 8789.24it/s] 51%|█████▏    | 205914/400000 [00:22<00:22, 8727.12it/s] 52%|█████▏    | 206815/400000 [00:22<00:21, 8808.01it/s] 52%|█████▏    | 207743/400000 [00:22<00:21, 8943.49it/s] 52%|█████▏    | 208639/400000 [00:23<00:21, 8878.58it/s] 52%|█████▏    | 209528/400000 [00:23<00:21, 8710.73it/s] 53%|█████▎    | 210401/400000 [00:23<00:21, 8634.50it/s] 53%|█████▎    | 211274/400000 [00:23<00:21, 8662.67it/s] 53%|█████▎    | 212154/400000 [00:23<00:21, 8703.00it/s] 53%|█████▎    | 213032/400000 [00:23<00:21, 8724.65it/s] 53%|█████▎    | 213910/400000 [00:23<00:21, 8735.12it/s] 54%|█████▎    | 214791/400000 [00:23<00:21, 8754.64it/s] 54%|█████▍    | 215667/400000 [00:23<00:21, 8737.58it/s] 54%|█████▍    | 216543/400000 [00:24<00:20, 8741.79it/s] 54%|█████▍    | 217422/400000 [00:24<00:20, 8755.47it/s] 55%|█████▍    | 218326/400000 [00:24<00:20, 8837.12it/s] 55%|█████▍    | 219210/400000 [00:24<00:20, 8765.92it/s] 55%|█████▌    | 220090/400000 [00:24<00:20, 8773.44it/s] 55%|█████▌    | 220974/400000 [00:24<00:20, 8792.52it/s] 55%|█████▌    | 221854/400000 [00:24<00:20, 8744.95it/s] 56%|█████▌    | 222729/400000 [00:24<00:20, 8732.49it/s] 56%|█████▌    | 223603/400000 [00:24<00:20, 8673.65it/s] 56%|█████▌    | 224482/400000 [00:24<00:20, 8708.06it/s] 56%|█████▋    | 225416/400000 [00:25<00:19, 8886.25it/s] 57%|█████▋    | 226341/400000 [00:25<00:19, 8990.29it/s] 57%|█████▋    | 227244/400000 [00:25<00:19, 8999.35it/s] 57%|█████▋    | 228159/400000 [00:25<00:19, 9041.85it/s] 57%|█████▋    | 229064/400000 [00:25<00:18, 9014.80it/s] 57%|█████▋    | 229966/400000 [00:25<00:18, 8986.00it/s] 58%|█████▊    | 230865/400000 [00:25<00:18, 8911.93it/s] 58%|█████▊    | 231769/400000 [00:25<00:18, 8948.18it/s] 58%|█████▊    | 232707/400000 [00:25<00:18, 9072.25it/s] 58%|█████▊    | 233625/400000 [00:25<00:18, 9103.45it/s] 59%|█████▊    | 234536/400000 [00:26<00:18, 9037.80it/s] 59%|█████▉    | 235482/400000 [00:26<00:17, 9158.10it/s] 59%|█████▉    | 236399/400000 [00:26<00:17, 9144.70it/s] 59%|█████▉    | 237429/400000 [00:26<00:17, 9461.71it/s] 60%|█████▉    | 238379/400000 [00:26<00:17, 9240.01it/s] 60%|█████▉    | 239307/400000 [00:26<00:17, 9115.93it/s] 60%|██████    | 240222/400000 [00:26<00:18, 8871.70it/s] 60%|██████    | 241127/400000 [00:26<00:17, 8923.25it/s] 61%|██████    | 242022/400000 [00:26<00:17, 8888.29it/s] 61%|██████    | 242937/400000 [00:26<00:17, 8964.47it/s] 61%|██████    | 243835/400000 [00:27<00:17, 8941.80it/s] 61%|██████    | 244731/400000 [00:27<00:17, 8829.05it/s] 61%|██████▏   | 245615/400000 [00:27<00:17, 8816.22it/s] 62%|██████▏   | 246500/400000 [00:27<00:17, 8826.05it/s] 62%|██████▏   | 247384/400000 [00:27<00:17, 8812.86it/s] 62%|██████▏   | 248266/400000 [00:27<00:17, 8786.80it/s] 62%|██████▏   | 249145/400000 [00:27<00:17, 8752.37it/s] 63%|██████▎   | 250037/400000 [00:27<00:17, 8800.84it/s] 63%|██████▎   | 250918/400000 [00:27<00:16, 8802.77it/s] 63%|██████▎   | 251799/400000 [00:27<00:17, 8677.84it/s] 63%|██████▎   | 252694/400000 [00:28<00:16, 8756.79it/s] 63%|██████▎   | 253571/400000 [00:28<00:16, 8622.47it/s] 64%|██████▎   | 254435/400000 [00:28<00:16, 8590.06it/s] 64%|██████▍   | 255341/400000 [00:28<00:16, 8724.51it/s] 64%|██████▍   | 256241/400000 [00:28<00:16, 8805.14it/s] 64%|██████▍   | 257123/400000 [00:28<00:16, 8775.36it/s] 65%|██████▍   | 258004/400000 [00:28<00:16, 8783.82it/s] 65%|██████▍   | 258943/400000 [00:28<00:15, 8955.18it/s] 65%|██████▍   | 259840/400000 [00:28<00:15, 8914.79it/s] 65%|██████▌   | 260733/400000 [00:28<00:15, 8850.49it/s] 65%|██████▌   | 261640/400000 [00:29<00:15, 8913.26it/s] 66%|██████▌   | 262536/400000 [00:29<00:15, 8924.80it/s] 66%|██████▌   | 263431/400000 [00:29<00:15, 8932.29it/s] 66%|██████▌   | 264344/400000 [00:29<00:15, 8990.50it/s] 66%|██████▋   | 265244/400000 [00:29<00:15, 8916.35it/s] 67%|██████▋   | 266136/400000 [00:29<00:15, 8852.82it/s] 67%|██████▋   | 267050/400000 [00:29<00:14, 8934.29it/s] 67%|██████▋   | 267958/400000 [00:29<00:14, 8976.26it/s] 67%|██████▋   | 268862/400000 [00:29<00:14, 8995.07it/s] 67%|██████▋   | 269764/400000 [00:29<00:14, 9002.02it/s] 68%|██████▊   | 270678/400000 [00:30<00:14, 9042.62it/s] 68%|██████▊   | 271583/400000 [00:30<00:14, 8931.25it/s] 68%|██████▊   | 272497/400000 [00:30<00:14, 8990.23it/s] 68%|██████▊   | 273487/400000 [00:30<00:13, 9243.56it/s] 69%|██████▊   | 274414/400000 [00:30<00:13, 9145.92it/s] 69%|██████▉   | 275331/400000 [00:30<00:13, 9011.94it/s] 69%|██████▉   | 276243/400000 [00:30<00:13, 9042.13it/s] 69%|██████▉   | 277161/400000 [00:30<00:13, 9081.60it/s] 70%|██████▉   | 278071/400000 [00:30<00:13, 9015.48it/s] 70%|██████▉   | 278974/400000 [00:30<00:13, 8906.13it/s] 70%|██████▉   | 279914/400000 [00:31<00:13, 9047.26it/s] 70%|███████   | 280820/400000 [00:31<00:13, 9010.68it/s] 70%|███████   | 281736/400000 [00:31<00:13, 9053.15it/s] 71%|███████   | 282642/400000 [00:31<00:13, 9025.12it/s] 71%|███████   | 283545/400000 [00:31<00:12, 8994.30it/s] 71%|███████   | 284445/400000 [00:31<00:12, 8927.14it/s] 71%|███████▏  | 285385/400000 [00:31<00:12, 9061.55it/s] 72%|███████▏  | 286323/400000 [00:31<00:12, 9153.57it/s] 72%|███████▏  | 287297/400000 [00:31<00:12, 9321.84it/s] 72%|███████▏  | 288231/400000 [00:32<00:12, 9150.58it/s] 72%|███████▏  | 289148/400000 [00:32<00:12, 9148.23it/s] 73%|███████▎  | 290176/400000 [00:32<00:11, 9460.50it/s] 73%|███████▎  | 291126/400000 [00:32<00:11, 9376.07it/s] 73%|███████▎  | 292067/400000 [00:32<00:11, 9053.48it/s] 73%|███████▎  | 292997/400000 [00:32<00:11, 9124.25it/s] 73%|███████▎  | 293913/400000 [00:32<00:11, 9066.76it/s] 74%|███████▎  | 294836/400000 [00:32<00:11, 9113.78it/s] 74%|███████▍  | 295830/400000 [00:32<00:11, 9346.27it/s] 74%|███████▍  | 296815/400000 [00:32<00:10, 9490.67it/s] 74%|███████▍  | 297865/400000 [00:33<00:10, 9770.83it/s] 75%|███████▍  | 298846/400000 [00:33<00:10, 9595.77it/s] 75%|███████▍  | 299809/400000 [00:33<00:10, 9374.86it/s] 75%|███████▌  | 300756/400000 [00:33<00:10, 9401.92it/s] 75%|███████▌  | 301699/400000 [00:33<00:10, 9225.91it/s] 76%|███████▌  | 302625/400000 [00:33<00:10, 9170.31it/s] 76%|███████▌  | 303552/400000 [00:33<00:10, 9198.74it/s] 76%|███████▌  | 304474/400000 [00:33<00:10, 9101.81it/s] 76%|███████▋  | 305396/400000 [00:33<00:10, 9134.86it/s] 77%|███████▋  | 306335/400000 [00:33<00:10, 9208.03it/s] 77%|███████▋  | 307310/400000 [00:34<00:09, 9362.16it/s] 77%|███████▋  | 308248/400000 [00:34<00:10, 9084.02it/s] 77%|███████▋  | 309159/400000 [00:34<00:10, 9023.65it/s] 78%|███████▊  | 310072/400000 [00:34<00:09, 9055.11it/s] 78%|███████▊  | 310979/400000 [00:34<00:09, 8945.19it/s] 78%|███████▊  | 311875/400000 [00:34<00:09, 8864.42it/s] 78%|███████▊  | 312763/400000 [00:34<00:09, 8754.83it/s] 78%|███████▊  | 313646/400000 [00:34<00:09, 8776.15it/s] 79%|███████▊  | 314525/400000 [00:34<00:09, 8767.33it/s] 79%|███████▉  | 315405/400000 [00:34<00:09, 8775.44it/s] 79%|███████▉  | 316283/400000 [00:35<00:09, 8772.08it/s] 79%|███████▉  | 317161/400000 [00:35<00:09, 8729.18it/s] 80%|███████▉  | 318040/400000 [00:35<00:09, 8747.22it/s] 80%|███████▉  | 318917/400000 [00:35<00:09, 8753.04it/s] 80%|███████▉  | 319793/400000 [00:35<00:09, 8709.56it/s] 80%|████████  | 320671/400000 [00:35<00:09, 8729.03it/s] 80%|████████  | 321545/400000 [00:35<00:08, 8717.38it/s] 81%|████████  | 322423/400000 [00:35<00:08, 8733.42it/s] 81%|████████  | 323315/400000 [00:35<00:08, 8787.47it/s] 81%|████████  | 324219/400000 [00:35<00:08, 8860.04it/s] 81%|████████▏ | 325110/400000 [00:36<00:08, 8872.23it/s] 82%|████████▏ | 326023/400000 [00:36<00:08, 8945.53it/s] 82%|████████▏ | 326918/400000 [00:36<00:08, 8734.80it/s] 82%|████████▏ | 327793/400000 [00:36<00:08, 8720.06it/s] 82%|████████▏ | 328666/400000 [00:36<00:08, 8703.46it/s] 82%|████████▏ | 329537/400000 [00:36<00:08, 8700.08it/s] 83%|████████▎ | 330418/400000 [00:36<00:07, 8729.97it/s] 83%|████████▎ | 331301/400000 [00:36<00:07, 8759.13it/s] 83%|████████▎ | 332178/400000 [00:36<00:07, 8704.86it/s] 83%|████████▎ | 333049/400000 [00:37<00:07, 8705.80it/s] 83%|████████▎ | 333926/400000 [00:37<00:07, 8724.21it/s] 84%|████████▎ | 334799/400000 [00:37<00:07, 8694.19it/s] 84%|████████▍ | 335669/400000 [00:37<00:07, 8396.55it/s] 84%|████████▍ | 336539/400000 [00:37<00:07, 8484.16it/s] 84%|████████▍ | 337418/400000 [00:37<00:07, 8573.16it/s] 85%|████████▍ | 338277/400000 [00:37<00:07, 8578.01it/s] 85%|████████▍ | 339189/400000 [00:37<00:06, 8733.49it/s] 85%|████████▌ | 340098/400000 [00:37<00:06, 8836.21it/s] 85%|████████▌ | 341025/400000 [00:37<00:06, 8959.91it/s] 86%|████████▌ | 342084/400000 [00:38<00:06, 9392.29it/s] 86%|████████▌ | 343030/400000 [00:38<00:06, 9234.56it/s] 86%|████████▌ | 343959/400000 [00:38<00:06, 9214.12it/s] 86%|████████▌ | 344988/400000 [00:38<00:05, 9511.38it/s] 86%|████████▋ | 345945/400000 [00:38<00:05, 9429.27it/s] 87%|████████▋ | 346905/400000 [00:38<00:05, 9478.13it/s] 87%|████████▋ | 347856/400000 [00:38<00:05, 9473.75it/s] 87%|████████▋ | 348857/400000 [00:38<00:05, 9628.05it/s] 87%|████████▋ | 349825/400000 [00:38<00:05, 9642.23it/s] 88%|████████▊ | 350792/400000 [00:38<00:05, 9647.99it/s] 88%|████████▊ | 351758/400000 [00:39<00:05, 9478.07it/s] 88%|████████▊ | 352785/400000 [00:39<00:04, 9701.98it/s] 88%|████████▊ | 353758/400000 [00:39<00:04, 9613.57it/s] 89%|████████▊ | 354722/400000 [00:39<00:04, 9451.36it/s] 89%|████████▉ | 355669/400000 [00:39<00:04, 9421.72it/s] 89%|████████▉ | 356613/400000 [00:39<00:04, 9159.35it/s] 89%|████████▉ | 357532/400000 [00:39<00:04, 8837.01it/s] 90%|████████▉ | 358444/400000 [00:39<00:04, 8918.45it/s] 90%|████████▉ | 359379/400000 [00:39<00:04, 9042.23it/s] 90%|█████████ | 360315/400000 [00:39<00:04, 9134.49it/s] 90%|█████████ | 361241/400000 [00:40<00:04, 9168.96it/s] 91%|█████████ | 362160/400000 [00:40<00:04, 9041.38it/s] 91%|█████████ | 363066/400000 [00:40<00:04, 8876.49it/s] 91%|█████████ | 363969/400000 [00:40<00:04, 8920.43it/s] 91%|█████████ | 364863/400000 [00:40<00:03, 8883.13it/s] 91%|█████████▏| 365786/400000 [00:40<00:03, 8983.05it/s] 92%|█████████▏| 366692/400000 [00:40<00:03, 9004.89it/s] 92%|█████████▏| 367652/400000 [00:40<00:03, 9174.04it/s] 92%|█████████▏| 368571/400000 [00:40<00:03, 9160.55it/s] 92%|█████████▏| 369492/400000 [00:40<00:03, 9173.08it/s] 93%|█████████▎| 370502/400000 [00:41<00:03, 9430.21it/s] 93%|█████████▎| 371479/400000 [00:41<00:02, 9527.42it/s] 93%|█████████▎| 372484/400000 [00:41<00:02, 9677.70it/s] 93%|█████████▎| 373474/400000 [00:41<00:02, 9743.31it/s] 94%|█████████▎| 374450/400000 [00:41<00:02, 9603.27it/s] 94%|█████████▍| 375426/400000 [00:41<00:02, 9649.72it/s] 94%|█████████▍| 376393/400000 [00:41<00:02, 9646.58it/s] 94%|█████████▍| 377388/400000 [00:41<00:02, 9733.27it/s] 95%|█████████▍| 378363/400000 [00:41<00:02, 9441.28it/s] 95%|█████████▍| 379310/400000 [00:41<00:02, 9407.05it/s] 95%|█████████▌| 380253/400000 [00:42<00:02, 9401.62it/s] 95%|█████████▌| 381195/400000 [00:42<00:02, 9229.29it/s] 96%|█████████▌| 382148/400000 [00:42<00:01, 9315.43it/s] 96%|█████████▌| 383133/400000 [00:42<00:01, 9467.62it/s] 96%|█████████▌| 384082/400000 [00:42<00:01, 9227.32it/s] 96%|█████████▋| 385008/400000 [00:42<00:01, 9216.85it/s] 96%|█████████▋| 385971/400000 [00:42<00:01, 9334.48it/s] 97%|█████████▋| 386906/400000 [00:42<00:01, 9276.43it/s] 97%|█████████▋| 387907/400000 [00:42<00:01, 9484.45it/s] 97%|█████████▋| 388915/400000 [00:43<00:01, 9655.17it/s] 97%|█████████▋| 389883/400000 [00:43<00:01, 9479.82it/s] 98%|█████████▊| 390834/400000 [00:43<00:00, 9290.05it/s] 98%|█████████▊| 391766/400000 [00:43<00:00, 9298.87it/s] 98%|█████████▊| 392727/400000 [00:43<00:00, 9385.41it/s] 98%|█████████▊| 393727/400000 [00:43<00:00, 9560.43it/s] 99%|█████████▊| 394716/400000 [00:43<00:00, 9655.36it/s] 99%|█████████▉| 395683/400000 [00:43<00:00, 9277.67it/s] 99%|█████████▉| 396677/400000 [00:43<00:00, 9465.55it/s] 99%|█████████▉| 397653/400000 [00:43<00:00, 9551.62it/s]100%|█████████▉| 398674/400000 [00:44<00:00, 9738.32it/s]100%|█████████▉| 399651/400000 [00:44<00:00, 9670.17it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 9052.36it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f312d097898>, <torchtext.data.dataset.TabularDataset object at 0x7f312d0979e8>, <torchtext.vocab.Vocab object at 0x7f312d097908>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.66 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.66 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.66 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.30 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.46 file/s]2020-06-13 14:30:02.136764: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-13 14:30:02.141284: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-13 14:30:02.141501: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a0644c9b60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 14:30:02.141516: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3203072/9912422 [00:00<00:00, 31982787.42it/s]9920512it [00:00, 34001056.70it/s]                             
0it [00:00, ?it/s]32768it [00:00, 632170.64it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 458533.35it/s]1654784it [00:00, 11248800.11it/s]                         
0it [00:00, ?it/s]8192it [00:00, 166895.31it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
