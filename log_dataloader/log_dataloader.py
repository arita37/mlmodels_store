
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb60ad559d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb60ad559d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb6759c1510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb6759c1510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb694befea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb694befea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb622cf5950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb622cf5950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb622cf5950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3891200/9912422 [00:00<00:00, 38837867.09it/s]9920512it [00:00, 33613876.70it/s]                             
0it [00:00, ?it/s]32768it [00:00, 512973.56it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 452796.90it/s]1654784it [00:00, 11192801.09it/s]                         
0it [00:00, ?it/s]8192it [00:00, 200419.61it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb60ac0e630>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb60ac23898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb622cf5598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb622cf5598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb622cf5598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:57,  2.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:57,  2.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:57,  2.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:57,  2.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:56,  2.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:56,  2.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:55,  2.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:38,  3.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:38,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:38,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:38,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:38,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:37,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:37,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:37,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:36,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:25,  5.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:25,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 10.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 10.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 10.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:11, 10.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:11, 10.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:11, 10.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:11, 10.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 14.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 19.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 32.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 40.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 72.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 78.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 78.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 83.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 83.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 87.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 87.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 87.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 87.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 87.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 87.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 87.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.08s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.08s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 87.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.08s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 87.33 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.13s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.08s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 87.33 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.13s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.13s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.20 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.13s/ url]
0 examples [00:00, ? examples/s]2020-07-24 00:08:46.956740: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-24 00:08:46.969776: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-24 00:08:46.970916: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a623c69630 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 00:08:46.970935: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.31 examples/s]96 examples [00:00,  4.72 examples/s]192 examples [00:00,  6.73 examples/s]294 examples [00:00,  9.59 examples/s]389 examples [00:00, 13.64 examples/s]477 examples [00:00, 19.36 examples/s]575 examples [00:00, 27.42 examples/s]673 examples [00:01, 38.70 examples/s]765 examples [00:01, 54.31 examples/s]862 examples [00:01, 75.73 examples/s]956 examples [00:01, 104.56 examples/s]1054 examples [00:01, 142.79 examples/s]1151 examples [00:01, 191.79 examples/s]1246 examples [00:01, 251.78 examples/s]1340 examples [00:01, 321.96 examples/s]1437 examples [00:01, 402.21 examples/s]1536 examples [00:01, 488.97 examples/s]1635 examples [00:02, 576.15 examples/s]1732 examples [00:02, 655.98 examples/s]1829 examples [00:02, 724.03 examples/s]1929 examples [00:02, 788.92 examples/s]2029 examples [00:02, 842.10 examples/s]2128 examples [00:02, 869.44 examples/s]2229 examples [00:02, 905.55 examples/s]2334 examples [00:02, 944.17 examples/s]2434 examples [00:02, 951.95 examples/s]2536 examples [00:02, 970.00 examples/s]2637 examples [00:03, 980.71 examples/s]2738 examples [00:03, 942.19 examples/s]2839 examples [00:03, 960.66 examples/s]2937 examples [00:03, 938.82 examples/s]3041 examples [00:03, 965.08 examples/s]3141 examples [00:03, 974.61 examples/s]3244 examples [00:03, 989.13 examples/s]3344 examples [00:03, 980.87 examples/s]3444 examples [00:03, 985.21 examples/s]3543 examples [00:03, 981.80 examples/s]3645 examples [00:04, 991.14 examples/s]3745 examples [00:04, 992.83 examples/s]3845 examples [00:04, 990.39 examples/s]3945 examples [00:04, 980.40 examples/s]4047 examples [00:04, 990.51 examples/s]4147 examples [00:04, 992.79 examples/s]4247 examples [00:04, 981.02 examples/s]4346 examples [00:04, 970.16 examples/s]4448 examples [00:04, 982.89 examples/s]4547 examples [00:04, 977.16 examples/s]4645 examples [00:05, 963.78 examples/s]4745 examples [00:05, 972.12 examples/s]4843 examples [00:05, 969.03 examples/s]4942 examples [00:05, 972.34 examples/s]5040 examples [00:05, 947.82 examples/s]5135 examples [00:05, 945.39 examples/s]5230 examples [00:05, 942.62 examples/s]5326 examples [00:05, 947.22 examples/s]5425 examples [00:05, 959.48 examples/s]5524 examples [00:06, 967.87 examples/s]5621 examples [00:06, 967.11 examples/s]5718 examples [00:06, 964.08 examples/s]5815 examples [00:06, 948.73 examples/s]5913 examples [00:06, 957.17 examples/s]6009 examples [00:06, 937.99 examples/s]6104 examples [00:06, 940.19 examples/s]6212 examples [00:06, 975.77 examples/s]6311 examples [00:06, 971.07 examples/s]6409 examples [00:06, 969.74 examples/s]6511 examples [00:07, 982.57 examples/s]6610 examples [00:07, 980.71 examples/s]6709 examples [00:07, 976.04 examples/s]6808 examples [00:07, 977.46 examples/s]6909 examples [00:07, 986.06 examples/s]7008 examples [00:07, 972.96 examples/s]7106 examples [00:07, 965.85 examples/s]7207 examples [00:07, 978.35 examples/s]7305 examples [00:07, 961.20 examples/s]7402 examples [00:07, 938.75 examples/s]7501 examples [00:08, 951.65 examples/s]7597 examples [00:08, 929.11 examples/s]7691 examples [00:08, 930.95 examples/s]7785 examples [00:08, 926.66 examples/s]7886 examples [00:08, 949.12 examples/s]7982 examples [00:08, 949.86 examples/s]8082 examples [00:08, 962.13 examples/s]8182 examples [00:08, 971.67 examples/s]8280 examples [00:08, 966.38 examples/s]8378 examples [00:08, 970.39 examples/s]8476 examples [00:09, 962.49 examples/s]8573 examples [00:09, 950.14 examples/s]8669 examples [00:09, 942.54 examples/s]8764 examples [00:09, 934.15 examples/s]8862 examples [00:09, 943.73 examples/s]8960 examples [00:09, 952.52 examples/s]9056 examples [00:09, 953.92 examples/s]9159 examples [00:09, 973.42 examples/s]9257 examples [00:09, 966.99 examples/s]9354 examples [00:09, 965.04 examples/s]9453 examples [00:10, 972.16 examples/s]9551 examples [00:10, 968.27 examples/s]9648 examples [00:10, 921.48 examples/s]9743 examples [00:10, 928.64 examples/s]9845 examples [00:10, 951.39 examples/s]9944 examples [00:10, 961.99 examples/s]10041 examples [00:10, 896.53 examples/s]10132 examples [00:10, 897.99 examples/s]10228 examples [00:10, 913.38 examples/s]10320 examples [00:11, 912.50 examples/s]10413 examples [00:11, 916.06 examples/s]10510 examples [00:11, 930.25 examples/s]10607 examples [00:11, 941.79 examples/s]10702 examples [00:11, 933.18 examples/s]10796 examples [00:11, 929.19 examples/s]10892 examples [00:11, 936.42 examples/s]10990 examples [00:11, 948.47 examples/s]11087 examples [00:11, 954.32 examples/s]11183 examples [00:11, 938.60 examples/s]11280 examples [00:12, 947.42 examples/s]11378 examples [00:12, 954.31 examples/s]11477 examples [00:12, 962.54 examples/s]11578 examples [00:12, 975.59 examples/s]11676 examples [00:12, 962.24 examples/s]11777 examples [00:12, 973.25 examples/s]11877 examples [00:12, 979.25 examples/s]11976 examples [00:12, 967.49 examples/s]12073 examples [00:12, 960.46 examples/s]12170 examples [00:12, 935.75 examples/s]12267 examples [00:13, 944.00 examples/s]12365 examples [00:13, 952.12 examples/s]12465 examples [00:13, 963.85 examples/s]12564 examples [00:13, 969.94 examples/s]12664 examples [00:13, 976.43 examples/s]12764 examples [00:13, 983.35 examples/s]12867 examples [00:13, 995.58 examples/s]12967 examples [00:13, 993.45 examples/s]13067 examples [00:13, 915.47 examples/s]13160 examples [00:14, 899.63 examples/s]13260 examples [00:14, 924.83 examples/s]13354 examples [00:14, 884.19 examples/s]13444 examples [00:14, 845.81 examples/s]13530 examples [00:14, 835.55 examples/s]13619 examples [00:14, 848.75 examples/s]13707 examples [00:14, 852.08 examples/s]13796 examples [00:14, 862.30 examples/s]13883 examples [00:14, 858.87 examples/s]13981 examples [00:14, 891.53 examples/s]14080 examples [00:15, 918.94 examples/s]14173 examples [00:15, 919.03 examples/s]14266 examples [00:15, 898.00 examples/s]14366 examples [00:15, 924.46 examples/s]14460 examples [00:15, 927.79 examples/s]14554 examples [00:15, 929.47 examples/s]14648 examples [00:15, 918.78 examples/s]14747 examples [00:15, 936.81 examples/s]14842 examples [00:15, 939.84 examples/s]14937 examples [00:15, 941.00 examples/s]15035 examples [00:16, 950.88 examples/s]15131 examples [00:16, 947.98 examples/s]15226 examples [00:16, 934.95 examples/s]15324 examples [00:16, 947.91 examples/s]15419 examples [00:16, 935.13 examples/s]15516 examples [00:16, 944.37 examples/s]15611 examples [00:16, 936.41 examples/s]15714 examples [00:16, 961.38 examples/s]15813 examples [00:16, 967.81 examples/s]15910 examples [00:17, 958.52 examples/s]16006 examples [00:17, 955.16 examples/s]16107 examples [00:17, 969.78 examples/s]16206 examples [00:17, 973.08 examples/s]16304 examples [00:17, 969.90 examples/s]16402 examples [00:17, 939.45 examples/s]16498 examples [00:17, 944.00 examples/s]16597 examples [00:17, 956.32 examples/s]16697 examples [00:17, 966.34 examples/s]16795 examples [00:17, 970.31 examples/s]16893 examples [00:18, 963.12 examples/s]16994 examples [00:18, 974.56 examples/s]17095 examples [00:18, 982.48 examples/s]17194 examples [00:18, 943.17 examples/s]17293 examples [00:18, 955.83 examples/s]17389 examples [00:18, 955.70 examples/s]17490 examples [00:18, 968.82 examples/s]17592 examples [00:18, 981.95 examples/s]17691 examples [00:18, 982.68 examples/s]17791 examples [00:18, 986.86 examples/s]17890 examples [00:19, 979.69 examples/s]17991 examples [00:19, 986.63 examples/s]18090 examples [00:19, 957.82 examples/s]18190 examples [00:19, 969.86 examples/s]18288 examples [00:19, 966.00 examples/s]18385 examples [00:19, 964.68 examples/s]18487 examples [00:19, 979.18 examples/s]18590 examples [00:19, 992.68 examples/s]18691 examples [00:19, 997.29 examples/s]18791 examples [00:19, 997.94 examples/s]18891 examples [00:20, 939.24 examples/s]18995 examples [00:20, 967.07 examples/s]19096 examples [00:20, 979.10 examples/s]19198 examples [00:20, 988.31 examples/s]19301 examples [00:20, 998.00 examples/s]19402 examples [00:20, 980.11 examples/s]19504 examples [00:20, 989.09 examples/s]19604 examples [00:20, 983.18 examples/s]19703 examples [00:20, 976.73 examples/s]19802 examples [00:21, 978.75 examples/s]19900 examples [00:21, 931.54 examples/s]19997 examples [00:21, 941.09 examples/s]20092 examples [00:21, 893.21 examples/s]20191 examples [00:21, 918.91 examples/s]20285 examples [00:21, 924.73 examples/s]20381 examples [00:21, 934.21 examples/s]20477 examples [00:21, 941.58 examples/s]20577 examples [00:21, 957.94 examples/s]20676 examples [00:21, 967.14 examples/s]20774 examples [00:22, 970.78 examples/s]20872 examples [00:22, 971.48 examples/s]20974 examples [00:22, 984.51 examples/s]21075 examples [00:22, 991.30 examples/s]21175 examples [00:22, 966.79 examples/s]21272 examples [00:22, 960.66 examples/s]21370 examples [00:22, 963.75 examples/s]21468 examples [00:22, 966.22 examples/s]21568 examples [00:22, 974.78 examples/s]21666 examples [00:22, 974.25 examples/s]21764 examples [00:23, 968.13 examples/s]21866 examples [00:23, 981.13 examples/s]21965 examples [00:23, 982.52 examples/s]22064 examples [00:23, 984.31 examples/s]22163 examples [00:23, 970.52 examples/s]22261 examples [00:23, 942.99 examples/s]22359 examples [00:23, 951.55 examples/s]22459 examples [00:23, 963.82 examples/s]22561 examples [00:23, 979.58 examples/s]22660 examples [00:23, 972.73 examples/s]22758 examples [00:24, 959.05 examples/s]22855 examples [00:24, 953.54 examples/s]22954 examples [00:24, 963.27 examples/s]23053 examples [00:24, 968.90 examples/s]23156 examples [00:24, 986.29 examples/s]23255 examples [00:24, 965.88 examples/s]23352 examples [00:24, 964.37 examples/s]23449 examples [00:24, 947.25 examples/s]23544 examples [00:24, 941.83 examples/s]23642 examples [00:25, 952.12 examples/s]23739 examples [00:25, 955.40 examples/s]23836 examples [00:25, 958.69 examples/s]23934 examples [00:25, 964.87 examples/s]24031 examples [00:25, 956.75 examples/s]24127 examples [00:25, 937.64 examples/s]24221 examples [00:25, 926.47 examples/s]24314 examples [00:25, 919.59 examples/s]24409 examples [00:25, 927.23 examples/s]24502 examples [00:25, 925.23 examples/s]24598 examples [00:26, 933.57 examples/s]24695 examples [00:26, 943.98 examples/s]24790 examples [00:26, 925.99 examples/s]24884 examples [00:26, 928.19 examples/s]24983 examples [00:26, 943.42 examples/s]25078 examples [00:26, 945.31 examples/s]25173 examples [00:26, 924.57 examples/s]25267 examples [00:26, 927.44 examples/s]25364 examples [00:26, 938.00 examples/s]25458 examples [00:26, 929.01 examples/s]25556 examples [00:27, 941.57 examples/s]25651 examples [00:27, 932.56 examples/s]25745 examples [00:27, 928.52 examples/s]25838 examples [00:27, 917.20 examples/s]25937 examples [00:27, 935.39 examples/s]26035 examples [00:27, 948.32 examples/s]26133 examples [00:27, 955.76 examples/s]26230 examples [00:27, 958.03 examples/s]26331 examples [00:27, 970.25 examples/s]26432 examples [00:27, 980.44 examples/s]26532 examples [00:28, 984.83 examples/s]26631 examples [00:28, 974.42 examples/s]26730 examples [00:28, 978.87 examples/s]26828 examples [00:28, 975.12 examples/s]26926 examples [00:28, 972.91 examples/s]27026 examples [00:28, 980.83 examples/s]27125 examples [00:28, 972.40 examples/s]27223 examples [00:28, 947.98 examples/s]27323 examples [00:28, 960.34 examples/s]27420 examples [00:29, 958.37 examples/s]27516 examples [00:29, 953.70 examples/s]27612 examples [00:29, 931.12 examples/s]27706 examples [00:29, 915.85 examples/s]27801 examples [00:29, 925.80 examples/s]27896 examples [00:29, 932.63 examples/s]27992 examples [00:29, 939.25 examples/s]28087 examples [00:29, 925.54 examples/s]28188 examples [00:29, 947.47 examples/s]28286 examples [00:29, 954.60 examples/s]28382 examples [00:30, 934.25 examples/s]28476 examples [00:30, 895.05 examples/s]28570 examples [00:30, 907.09 examples/s]28665 examples [00:30, 918.43 examples/s]28758 examples [00:30, 907.03 examples/s]28855 examples [00:30, 924.07 examples/s]28948 examples [00:30, 922.06 examples/s]29041 examples [00:30, 906.10 examples/s]29138 examples [00:30, 923.29 examples/s]29239 examples [00:30, 945.54 examples/s]29334 examples [00:31, 936.50 examples/s]29432 examples [00:31, 947.23 examples/s]29529 examples [00:31, 953.18 examples/s]29625 examples [00:31, 953.79 examples/s]29727 examples [00:31, 970.86 examples/s]29827 examples [00:31, 976.38 examples/s]29927 examples [00:31, 980.21 examples/s]30026 examples [00:31, 913.43 examples/s]30122 examples [00:31, 925.40 examples/s]30219 examples [00:32, 936.50 examples/s]30319 examples [00:32, 952.57 examples/s]30415 examples [00:32, 913.88 examples/s]30510 examples [00:32, 923.36 examples/s]30611 examples [00:32, 945.33 examples/s]30709 examples [00:32, 954.87 examples/s]30805 examples [00:32, 921.62 examples/s]30907 examples [00:32, 947.75 examples/s]31003 examples [00:32, 946.63 examples/s]31102 examples [00:32, 958.30 examples/s]31199 examples [00:33, 957.02 examples/s]31295 examples [00:33, 953.03 examples/s]31393 examples [00:33, 958.63 examples/s]31489 examples [00:33, 958.70 examples/s]31590 examples [00:33, 971.23 examples/s]31688 examples [00:33, 946.61 examples/s]31783 examples [00:33, 947.53 examples/s]31887 examples [00:33, 972.32 examples/s]31985 examples [00:33, 973.10 examples/s]32083 examples [00:33, 969.00 examples/s]32182 examples [00:34, 972.64 examples/s]32280 examples [00:34, 968.94 examples/s]32378 examples [00:34, 971.92 examples/s]32476 examples [00:34, 964.96 examples/s]32574 examples [00:34, 967.17 examples/s]32673 examples [00:34, 971.75 examples/s]32773 examples [00:34, 979.94 examples/s]32872 examples [00:34, 975.56 examples/s]32970 examples [00:34, 975.82 examples/s]33071 examples [00:34, 985.63 examples/s]33173 examples [00:35, 995.19 examples/s]33273 examples [00:35, 978.71 examples/s]33371 examples [00:35, 971.83 examples/s]33469 examples [00:35, 965.41 examples/s]33568 examples [00:35, 970.40 examples/s]33666 examples [00:35, 959.96 examples/s]33765 examples [00:35, 966.52 examples/s]33864 examples [00:35, 971.25 examples/s]33962 examples [00:35, 967.99 examples/s]34062 examples [00:35, 976.04 examples/s]34165 examples [00:36, 990.31 examples/s]34265 examples [00:36, 989.77 examples/s]34367 examples [00:36, 996.42 examples/s]34467 examples [00:36, 990.71 examples/s]34567 examples [00:36, 992.84 examples/s]34667 examples [00:36, 983.97 examples/s]34766 examples [00:36, 971.46 examples/s]34864 examples [00:36, 969.58 examples/s]34962 examples [00:36, 935.03 examples/s]35056 examples [00:37, 930.56 examples/s]35156 examples [00:37, 948.04 examples/s]35255 examples [00:37, 958.57 examples/s]35358 examples [00:37, 976.20 examples/s]35456 examples [00:37, 971.32 examples/s]35556 examples [00:37, 979.39 examples/s]35657 examples [00:37, 987.93 examples/s]35760 examples [00:37, 998.73 examples/s]35862 examples [00:37, 1003.54 examples/s]35963 examples [00:37, 985.14 examples/s] 36066 examples [00:38, 996.10 examples/s]36166 examples [00:38, 994.24 examples/s]36268 examples [00:38, 1000.22 examples/s]36369 examples [00:38, 978.96 examples/s] 36468 examples [00:38, 961.21 examples/s]36565 examples [00:38, 961.59 examples/s]36666 examples [00:38, 974.89 examples/s]36766 examples [00:38, 980.89 examples/s]36865 examples [00:38, 981.54 examples/s]36964 examples [00:38, 972.65 examples/s]37064 examples [00:39, 979.15 examples/s]37162 examples [00:39, 975.98 examples/s]37260 examples [00:39, 968.82 examples/s]37357 examples [00:39, 958.02 examples/s]37454 examples [00:39, 961.30 examples/s]37555 examples [00:39, 973.59 examples/s]37655 examples [00:39, 980.23 examples/s]37754 examples [00:39, 982.61 examples/s]37853 examples [00:39, 976.76 examples/s]37951 examples [00:39, 955.31 examples/s]38049 examples [00:40, 960.98 examples/s]38146 examples [00:40, 942.94 examples/s]38245 examples [00:40, 954.72 examples/s]38345 examples [00:40, 966.57 examples/s]38442 examples [00:40, 961.64 examples/s]38545 examples [00:40, 978.50 examples/s]38646 examples [00:40, 987.16 examples/s]38750 examples [00:40, 1000.65 examples/s]38853 examples [00:40, 1008.31 examples/s]38954 examples [00:40, 989.14 examples/s] 39054 examples [00:41, 989.83 examples/s]39154 examples [00:41, 969.38 examples/s]39254 examples [00:41, 975.48 examples/s]39357 examples [00:41, 988.69 examples/s]39457 examples [00:41, 968.72 examples/s]39555 examples [00:41, 960.19 examples/s]39652 examples [00:41, 943.74 examples/s]39750 examples [00:41, 953.39 examples/s]39849 examples [00:41, 962.44 examples/s]39947 examples [00:42, 965.53 examples/s]40044 examples [00:42, 895.35 examples/s]40145 examples [00:42, 926.15 examples/s]40251 examples [00:42, 961.89 examples/s]40349 examples [00:42, 961.78 examples/s]40450 examples [00:42, 973.75 examples/s]40552 examples [00:42, 985.66 examples/s]40651 examples [00:42, 978.78 examples/s]40752 examples [00:42, 985.51 examples/s]40852 examples [00:42, 988.21 examples/s]40951 examples [00:43, 987.44 examples/s]41050 examples [00:43, 965.56 examples/s]41150 examples [00:43, 974.72 examples/s]41253 examples [00:43, 990.55 examples/s]41353 examples [00:43, 992.36 examples/s]41453 examples [00:43, 991.99 examples/s]41555 examples [00:43, 1000.15 examples/s]41656 examples [00:43, 1002.40 examples/s]41758 examples [00:43, 1007.23 examples/s]41859 examples [00:43, 995.18 examples/s] 41959 examples [00:44, 957.68 examples/s]42056 examples [00:44, 946.38 examples/s]42158 examples [00:44, 964.75 examples/s]42260 examples [00:44, 978.74 examples/s]42359 examples [00:44, 977.35 examples/s]42459 examples [00:44, 983.58 examples/s]42558 examples [00:44, 960.99 examples/s]42657 examples [00:44, 966.99 examples/s]42754 examples [00:44, 966.80 examples/s]42851 examples [00:45, 957.90 examples/s]42948 examples [00:45, 960.99 examples/s]43047 examples [00:45, 967.47 examples/s]43144 examples [00:45, 964.15 examples/s]43245 examples [00:45, 975.30 examples/s]43344 examples [00:45, 977.52 examples/s]43448 examples [00:45, 994.54 examples/s]43549 examples [00:45, 998.78 examples/s]43649 examples [00:45, 999.02 examples/s]43754 examples [00:45, 1012.33 examples/s]43856 examples [00:46, 989.84 examples/s] 43956 examples [00:46, 980.95 examples/s]44055 examples [00:46, 974.63 examples/s]44153 examples [00:46, 964.00 examples/s]44250 examples [00:46, 940.52 examples/s]44345 examples [00:46, 934.64 examples/s]44444 examples [00:46, 949.30 examples/s]44542 examples [00:46, 956.07 examples/s]44638 examples [00:46, 954.49 examples/s]44734 examples [00:46, 952.06 examples/s]44830 examples [00:47, 944.81 examples/s]44925 examples [00:47, 910.05 examples/s]45029 examples [00:47, 943.85 examples/s]45131 examples [00:47, 964.32 examples/s]45234 examples [00:47, 980.98 examples/s]45333 examples [00:47, 979.29 examples/s]45432 examples [00:47, 964.08 examples/s]45529 examples [00:47, 962.95 examples/s]45626 examples [00:47, 959.47 examples/s]45723 examples [00:47, 961.62 examples/s]45820 examples [00:48, 959.46 examples/s]45918 examples [00:48, 964.28 examples/s]46019 examples [00:48, 977.11 examples/s]46118 examples [00:48, 980.48 examples/s]46219 examples [00:48, 986.97 examples/s]46318 examples [00:48, 978.92 examples/s]46420 examples [00:48, 990.24 examples/s]46520 examples [00:48, 941.20 examples/s]46618 examples [00:48, 950.74 examples/s]46718 examples [00:49, 964.65 examples/s]46815 examples [00:49, 962.02 examples/s]46913 examples [00:49, 966.30 examples/s]47010 examples [00:49, 956.22 examples/s]47108 examples [00:49, 962.36 examples/s]47207 examples [00:49, 969.80 examples/s]47305 examples [00:49, 962.78 examples/s]47403 examples [00:49, 966.24 examples/s]47500 examples [00:49, 959.67 examples/s]47600 examples [00:49, 970.27 examples/s]47699 examples [00:50, 975.87 examples/s]47797 examples [00:50, 966.26 examples/s]47902 examples [00:50, 986.67 examples/s]48003 examples [00:50, 990.79 examples/s]48103 examples [00:50, 972.65 examples/s]48203 examples [00:50, 978.09 examples/s]48301 examples [00:50, 940.45 examples/s]48402 examples [00:50, 958.09 examples/s]48504 examples [00:50, 973.18 examples/s]48606 examples [00:50, 986.13 examples/s]48707 examples [00:51, 990.89 examples/s]48807 examples [00:51, 972.64 examples/s]48910 examples [00:51, 988.16 examples/s]49013 examples [00:51, 997.60 examples/s]49114 examples [00:51, 998.67 examples/s]49214 examples [00:51, 996.25 examples/s]49314 examples [00:51, 985.40 examples/s]49414 examples [00:51, 988.58 examples/s]49515 examples [00:51, 993.44 examples/s]49615 examples [00:51, 979.14 examples/s]49713 examples [00:52, 971.09 examples/s]49814 examples [00:52, 981.03 examples/s]49913 examples [00:52, 955.94 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6251/50000 [00:00<00:00, 62382.15 examples/s] 34%|███▎      | 16809/50000 [00:00<00:00, 71107.85 examples/s] 57%|█████▋    | 28327/50000 [00:00<00:00, 80328.51 examples/s] 81%|████████  | 40605/50000 [00:00<00:00, 89623.63 examples/s]                                                               0 examples [00:00, ? examples/s]80 examples [00:00, 795.67 examples/s]181 examples [00:00, 849.46 examples/s]277 examples [00:00, 878.34 examples/s]370 examples [00:00, 893.00 examples/s]469 examples [00:00, 918.65 examples/s]558 examples [00:00, 907.62 examples/s]654 examples [00:00, 920.16 examples/s]745 examples [00:00, 915.21 examples/s]838 examples [00:00, 913.18 examples/s]930 examples [00:01, 913.75 examples/s]1030 examples [00:01, 936.97 examples/s]1130 examples [00:01, 954.06 examples/s]1227 examples [00:01, 956.13 examples/s]1325 examples [00:01, 961.79 examples/s]1422 examples [00:01, 962.37 examples/s]1521 examples [00:01, 969.28 examples/s]1618 examples [00:01, 963.89 examples/s]1715 examples [00:01, 959.53 examples/s]1811 examples [00:01, 950.88 examples/s]1914 examples [00:02, 973.08 examples/s]2014 examples [00:02, 978.93 examples/s]2112 examples [00:02, 977.83 examples/s]2212 examples [00:02, 983.17 examples/s]2314 examples [00:02, 992.33 examples/s]2414 examples [00:02, 993.15 examples/s]2518 examples [00:02, 1004.98 examples/s]2619 examples [00:02, 968.81 examples/s] 2717 examples [00:02, 962.10 examples/s]2819 examples [00:02, 978.48 examples/s]2918 examples [00:03, 979.56 examples/s]3020 examples [00:03, 991.11 examples/s]3120 examples [00:03, 986.34 examples/s]3219 examples [00:03, 973.06 examples/s]3318 examples [00:03, 975.23 examples/s]3418 examples [00:03, 981.08 examples/s]3517 examples [00:03, 973.77 examples/s]3615 examples [00:03, 966.39 examples/s]3712 examples [00:03, 965.69 examples/s]3813 examples [00:03, 978.25 examples/s]3912 examples [00:04, 980.94 examples/s]4011 examples [00:04, 968.56 examples/s]4108 examples [00:04, 962.32 examples/s]4207 examples [00:04, 968.50 examples/s]4307 examples [00:04, 976.26 examples/s]4406 examples [00:04, 979.96 examples/s]4505 examples [00:04, 975.10 examples/s]4608 examples [00:04, 990.40 examples/s]4708 examples [00:04, 984.49 examples/s]4809 examples [00:04, 988.80 examples/s]4908 examples [00:05, 980.11 examples/s]5009 examples [00:05, 987.58 examples/s]5108 examples [00:05, 987.85 examples/s]5207 examples [00:05, 974.87 examples/s]5308 examples [00:05, 983.40 examples/s]5411 examples [00:05, 995.73 examples/s]5512 examples [00:05, 999.35 examples/s]5615 examples [00:05, 1008.21 examples/s]5716 examples [00:05, 1000.28 examples/s]5820 examples [00:05, 1009.93 examples/s]5922 examples [00:06, 1002.51 examples/s]6023 examples [00:06, 987.79 examples/s] 6122 examples [00:06, 982.80 examples/s]6221 examples [00:06, 969.84 examples/s]6324 examples [00:06, 987.11 examples/s]6423 examples [00:06, 973.26 examples/s]6526 examples [00:06, 989.18 examples/s]6631 examples [00:06, 1004.15 examples/s]6732 examples [00:06, 1002.63 examples/s]6833 examples [00:07, 1002.19 examples/s]6936 examples [00:07, 1008.31 examples/s]7037 examples [00:07, 1008.65 examples/s]7138 examples [00:07, 996.71 examples/s] 7238 examples [00:07, 889.30 examples/s]7336 examples [00:07, 912.20 examples/s]7436 examples [00:07, 936.88 examples/s]7536 examples [00:07, 954.69 examples/s]7633 examples [00:07, 956.49 examples/s]7730 examples [00:07, 942.50 examples/s]7837 examples [00:08, 976.63 examples/s]7942 examples [00:08, 994.85 examples/s]8046 examples [00:08, 1005.32 examples/s]8149 examples [00:08, 1011.90 examples/s]8251 examples [00:08, 1000.65 examples/s]8353 examples [00:08, 1003.83 examples/s]8456 examples [00:08, 1009.11 examples/s]8558 examples [00:08, 1003.53 examples/s]8661 examples [00:08, 1009.76 examples/s]8764 examples [00:08, 1013.17 examples/s]8867 examples [00:09, 1017.00 examples/s]8969 examples [00:09, 1016.95 examples/s]9071 examples [00:09, 1007.41 examples/s]9172 examples [00:09, 1005.83 examples/s]9273 examples [00:09, 976.64 examples/s] 9376 examples [00:09, 990.22 examples/s]9480 examples [00:09, 1003.94 examples/s]9582 examples [00:09, 1008.66 examples/s]9683 examples [00:09, 1009.00 examples/s]9784 examples [00:10, 985.30 examples/s] 9884 examples [00:10, 988.92 examples/s]9988 examples [00:10, 1003.07 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteGV7BQK/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteGV7BQK/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb622cf5950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb622cf5950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb622cf5950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb68e1f9f28>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb5e106c400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb622cf5950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb622cf5950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb622cf5950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb5e106cef0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb60ac234a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb59b6c81e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb59b6c81e0> 

  function with postional parmater data_info <function split_train_valid at 0x7fb59b6c81e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb59b6c82f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb59b6c82f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb59b6c82f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=73861be50b27bdae6421749de01ac1f118ee048f45a11f317cc5fb225d6487f9
  Stored in directory: /tmp/pip-ephem-wheel-cache-aqb6kaqv/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:08:36, 13.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:55:59, 18.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:06:26, 26.3kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:23:02, 37.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:27:29, 53.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.89M/862M [00:01<3:05:58, 76.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:01<2:09:26, 109kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.1M/862M [00:01<1:30:18, 156kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.4M/862M [00:01<1:02:59, 222kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.8M/862M [00:01<43:58, 316kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.0M/862M [00:01<30:40, 451kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.4M/862M [00:01<21:30, 640kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.2M/862M [00:02<15:03, 909kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.8M/862M [00:02<10:36, 1.28MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:02<07:27, 1.82MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<05:57, 2.26MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<06:04, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<05:48, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:04<04:32, 2.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:04<03:19, 4.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<49:15, 271kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<35:57, 372kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:06<25:26, 524kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<20:39, 644kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<17:21, 766kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:08<12:50, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:08<09:07, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<46:02, 287kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<33:35, 394kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:10<23:48, 554kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<19:40, 669kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<14:53, 883kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:12<10:39, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:12<07:38, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<54:49, 239kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<39:42, 330kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:14<28:01, 466kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<22:37, 576kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<18:31, 703kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<13:37, 955kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<11:35, 1.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<09:28, 1.37MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:18<06:57, 1.86MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<07:50, 1.64MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<08:07, 1.59MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<06:14, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.3M/862M [00:20<04:31, 2.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<10:27, 1.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<08:37, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:22<06:21, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<07:25, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<07:47, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<06:06, 2.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:25, 2.86MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<1:33:40, 135kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:06:50, 190kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<47:00, 269kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<35:46, 352kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<26:19, 479kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<18:40, 674kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<15:58, 785kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<12:14, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:53, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<06:21, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<51:36, 242kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<37:10, 335kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<26:15, 474kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<18:29, 671kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<1:02:53, 197kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<46:35, 266kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<33:05, 374kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<23:20, 530kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<20:20, 607kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<15:30, 795kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<11:09, 1.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<10:39, 1.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:42, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:21, 1.93MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<07:20, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:22, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:45, 2.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:11, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:49, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:20, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:41, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<05:14, 2.30MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<03:57, 3.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:35, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:08, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:53, 3.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:32, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:23, 1.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<05:01, 2.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<05:25, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:02, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<03:49, 3.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:25, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:00, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<03:47, 3.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:25, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:16, 1.87MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:02, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:53<03:38, 3.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<17:10, 678kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<13:17, 876kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<09:33, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:14, 1.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:02, 1.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<06:57, 1.66MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<04:58, 2.31MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<17:55, 642kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<13:47, 834kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<09:57, 1.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:28, 1.21MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:10, 1.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<07:01, 1.63MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:01<05:03, 2.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<21:27, 530kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<16:15, 699kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<11:39, 973kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<10:38, 1.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<09:50, 1.15MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<07:24, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<05:19, 2.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:31, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<07:11, 1.56MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:17, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:09, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:48, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:21, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<03:53, 2.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<20:09, 550kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<15:19, 724kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<11:00, 1.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<10:06, 1.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<08:17, 1.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<06:02, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:13<04:21, 2.51MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<1:08:57, 159kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<50:42, 216kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<35:59, 304kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<25:18, 431kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<21:06, 516kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<15:57, 682kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<11:23, 953kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<10:19, 1.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<09:37, 1.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<07:19, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<05:15, 2.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<20:33, 523kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<15:32, 691kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<11:09, 961kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<10:08, 1.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:16, 1.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<06:01, 1.77MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:33, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:53, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:23, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:25<03:53, 2.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<18:30, 569kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<14:05, 747kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<10:05, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<09:22, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:47, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:37, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<04:46, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:29, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:22, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<04:41, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:33, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:11, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:49, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<03:28, 2.96MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<15:15, 673kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<11:47, 870kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<08:30, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<08:11, 1.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:59, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:08, 1.66MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<04:25, 2.29MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<18:59, 533kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<14:13, 711kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<10:12, 988kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<09:20, 1.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<08:40, 1.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:36, 1.52MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<04:43, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<15:46, 633kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<12:07, 823kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<08:44, 1.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<08:17, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:00, 1.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:08, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<04:23, 2.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<15:25, 638kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<11:50, 831kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<08:29, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<08:07, 1.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<07:51, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:56, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<04:17, 2.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<06:33, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<05:38, 1.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<04:12, 2.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:03, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:41, 1.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:24, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<03:14, 2.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:35, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:57, 1.93MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:41, 2.59MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:40, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:17, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:06, 2.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:02, 3.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:06, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:36, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:28, 2.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:29, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:08, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<04:02, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<02:55, 3.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:22, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:10, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:33, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:13, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:39, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:28, 2.64MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:27, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:03, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:57, 2.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<02:53, 3.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:40, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:57, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:42, 2.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:35, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:02, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:59, 2.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<02:53, 3.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<48:55, 183kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<35:11, 254kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<24:46, 360kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<19:14, 461kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<14:25, 615kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<10:18, 858kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<09:07, 964kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<08:21, 1.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:14, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<04:28, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:09, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:57, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:22, 1.99MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:57, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:18, 1.63MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:06, 2.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<02:59, 2.89MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:00, 1.43MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:08, 1.67MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:49, 2.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:33, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:01, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:56, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<02:51, 2.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<1:02:18, 136kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<44:30, 190kB/s]  .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<31:17, 269kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<23:39, 354kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<17:16, 485kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<12:13, 683kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<08:38, 962kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<27:54, 298kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<21:17, 390kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<15:15, 544kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<10:47, 767kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<10:12, 808kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<08:02, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:48, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:52, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:57, 1.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:36, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<03:18, 2.45MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<15:04, 538kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<11:23, 712kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<08:09, 990kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<07:32, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<06:06, 1.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:28, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:57, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:09, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:01, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<02:54, 2.72MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<25:48, 306kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<18:54, 418kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<13:22, 589kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<11:05, 707kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<08:37, 908kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<06:13, 1.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<06:02, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<05:05, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<03:44, 2.07MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:18, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<04:39, 1.65MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:35, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:41, 2.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:45, 2.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:27, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:35, 2.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:29, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:07, 1.83MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:13, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:23, 3.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:56, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:32, 2.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:40, 2.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:34, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:15, 2.27MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:28, 2.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:24, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:57, 1.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:10, 2.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:58<02:18, 3.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<12:45, 571kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<09:41, 751kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<06:55, 1.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<06:28, 1.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:04, 1.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:34, 1.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<03:18, 2.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:58, 1.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:16, 1.67MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<03:10, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:45, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:08, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:12, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<02:20, 3.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:20, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:48, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:50, 2.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:30, 1.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:00, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:10, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<02:18, 2.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<12:22, 555kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<09:24, 729kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<06:44, 1.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:11, 1.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:04, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:43, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:05, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:35, 1.87MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:41, 2.49MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:20, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:03, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:17, 2.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:02, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:32, 1.86MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:47, 2.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<02:01, 3.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:30, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:51, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:50, 2.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:27, 1.86MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:51, 1.67MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:00, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:10, 2.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:41, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:58, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:56, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:27, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:04, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:18, 2.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:03, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:49, 2.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:06, 2.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:49, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:21, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:38, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<01:54, 3.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<05:09, 1.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:14, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:06, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:33, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:49, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:00, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<02:10, 2.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<10:54, 547kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<08:16, 720kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<05:54, 1.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:24, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:05, 1.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:53, 1.51MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<02:46, 2.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<11:05, 525kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<08:23, 693kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<05:59, 968kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:26, 1.06MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:02, 1.14MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:47, 1.51MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<02:42, 2.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:43, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:49, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:49, 2.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:12, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:29, 1.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:44, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<01:58, 2.80MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<09:52, 562kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<07:29, 739kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<05:22, 1.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:59, 1.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:41, 1.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:34, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<02:32, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<08:32, 634kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<06:33, 823kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<04:41, 1.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:26, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:14, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:12, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<02:18, 2.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:03, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:25, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:30, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:53, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:10, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:28, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<01:47, 2.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<07:24, 693kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<05:44, 893kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<04:08, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:59, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:21, 1.50MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:27, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:49, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:29, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<01:50, 2.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:26, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:09, 2.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:38, 2.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:12, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:37, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:03, 2.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:30, 3.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:41, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:23, 2.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:47, 2.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:19, 2.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:37, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:03, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:29, 3.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:54, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:32, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:53, 2.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:21, 1.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:39, 1.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:03, 2.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:30, 3.00MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:33, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:17, 1.97MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:41, 2.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:09, 2.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:27, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:55, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:23, 3.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<17:45, 246kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<12:53, 339kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<09:04, 479kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<07:15, 593kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<05:27, 788kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:53, 1.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<02:45, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<13:03, 324kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<09:34, 442kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<06:46, 621kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<05:39, 736kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<04:50, 859kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<03:34, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:32, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:26, 1.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:51, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<02:05, 1.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:20, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:03, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:32, 2.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:59, 1.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:13, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:45, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:16, 3.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<12:35, 309kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<09:13, 421kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<06:30, 593kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<05:21, 714kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<04:36, 830kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<03:23, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<02:24, 1.57MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:13, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:40, 1.40MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:57, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:10, 1.70MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:17, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:46, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:43<01:16, 2.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<11:50, 305kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<08:39, 416kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<06:06, 587kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<05:02, 704kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<04:14, 834kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<03:07, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<02:12, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<03:39, 951kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:55, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<02:06, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:13, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:19, 1.46MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:48, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:51<01:17, 2.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<05:54, 564kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<04:29, 741kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<03:12, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:56, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:45, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:04, 1.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:29, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:59, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:43, 1.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:16, 2.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:36, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:47, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:23, 2.24MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<01:00, 3.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:15, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:54, 1.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:23, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:38, 1.83MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:47, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:23, 2.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<00:59, 2.95MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:42, 1.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:09, 1.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:34, 1.83MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:43, 1.65MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:50, 1.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:25, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<01:00, 2.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<03:40, 756kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:52, 966kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<02:03, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:01, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:41, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:13, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:28, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:36, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:16, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<00:54, 2.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<04:40, 550kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:33, 723kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<02:31, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:17, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:08, 1.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:37, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<01:08, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<04:37, 526kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<03:29, 694kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:29, 966kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:14, 1.06MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:04, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:34, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<01:06, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<04:08, 554kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<03:08, 728kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<02:13, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:01, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:54, 1.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:25, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<01:01, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:33, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:18, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:57, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:08, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:14, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:57, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:41, 2.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:17, 1.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:06, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:49, 2.41MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:59, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:08, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:52, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:37, 3.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:12, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:02, 1.79MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:45, 2.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:55, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:02, 1.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:48, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<00:34, 3.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:07, 1.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:58, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:43, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:51, 1.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:46, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:34, 2.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:45, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:51, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:40, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:29, 3.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:55, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:48, 1.89MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:35, 2.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:43, 2.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:39, 2.19MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:29, 2.91MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:38, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:44, 1.86MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:35, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:24, 3.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:54, 694kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:28, 894kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<01:02, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:59, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:56, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:42, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:29, 2.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:07, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:54, 1.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:39, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:41, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:42, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:32, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:22, 2.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:25, 735kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:06, 945kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:46, 1.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:44, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:44, 1.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:33, 1.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:23, 2.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<01:41, 536kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:16, 709kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:53, 986kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:47, 1.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:44, 1.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:32, 1.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:22, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:38, 1.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:32, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:22, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:24, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:25, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:13, 2.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<05:15, 120kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<03:30, 171kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<02:25, 232kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<01:44, 319kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<01:10, 451kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:52, 564kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:42, 688kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:30, 940kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:19, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:27, 915kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:21, 1.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:14, 1.57MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:14, 1.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:14, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 1.92MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:11, 1.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:09, 1.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:06, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:07, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:03, 3.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:04, 2.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.82MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.33MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 3.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.61MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<36:59:44,  3.00it/s]  0%|          | 831/400000 [00:00<25:50:50,  4.29it/s]  0%|          | 1693/400000 [00:00<18:03:28,  6.13it/s]  1%|          | 2513/400000 [00:00<12:37:06,  8.75it/s]  1%|          | 3358/400000 [00:00<8:49:05, 12.49it/s]   1%|          | 4109/400000 [00:00<6:09:55, 17.84it/s]  1%|          | 4941/400000 [00:00<4:18:38, 25.46it/s]  1%|▏         | 5790/400000 [00:01<3:00:53, 36.32it/s]  2%|▏         | 6621/400000 [00:01<2:06:35, 51.79it/s]  2%|▏         | 7479/400000 [00:01<1:28:39, 73.79it/s]  2%|▏         | 8288/400000 [00:01<1:02:10, 105.01it/s]  2%|▏         | 9119/400000 [00:01<43:39, 149.21it/s]    2%|▏         | 9932/400000 [00:01<30:44, 211.49it/s]  3%|▎         | 10766/400000 [00:01<21:42, 298.87it/s]  3%|▎         | 11620/400000 [00:01<15:23, 420.65it/s]  3%|▎         | 12450/400000 [00:01<10:59, 587.67it/s]  3%|▎         | 13310/400000 [00:01<07:54, 815.62it/s]  4%|▎         | 14162/400000 [00:02<05:44, 1119.22it/s]  4%|▍         | 15018/400000 [00:02<04:14, 1514.04it/s]  4%|▍         | 15862/400000 [00:02<03:12, 1997.87it/s]  4%|▍         | 16688/400000 [00:02<02:28, 2580.95it/s]  4%|▍         | 17541/400000 [00:02<01:57, 3263.80it/s]  5%|▍         | 18385/400000 [00:02<01:35, 3999.15it/s]  5%|▍         | 19220/400000 [00:02<01:20, 4714.83it/s]  5%|▌         | 20072/400000 [00:02<01:09, 5443.15it/s]  5%|▌         | 20907/400000 [00:02<01:02, 6039.18it/s]  5%|▌         | 21757/400000 [00:02<00:57, 6611.67it/s]  6%|▌         | 22595/400000 [00:03<00:53, 7057.58it/s]  6%|▌         | 23430/400000 [00:03<00:52, 7236.80it/s]  6%|▌         | 24246/400000 [00:03<00:50, 7423.26it/s]  6%|▋         | 25061/400000 [00:03<00:49, 7625.26it/s]  6%|▋         | 25927/400000 [00:03<00:47, 7907.51it/s]  7%|▋         | 26791/400000 [00:03<00:46, 8113.18it/s]  7%|▋         | 27629/400000 [00:03<00:45, 8173.06it/s]  7%|▋         | 28498/400000 [00:03<00:44, 8320.01it/s]  7%|▋         | 29344/400000 [00:03<00:45, 8175.33it/s]  8%|▊         | 30197/400000 [00:03<00:44, 8277.90it/s]  8%|▊         | 31057/400000 [00:04<00:44, 8369.86it/s]  8%|▊         | 31917/400000 [00:04<00:43, 8437.49it/s]  8%|▊         | 32798/400000 [00:04<00:42, 8545.55it/s]  8%|▊         | 33656/400000 [00:04<00:43, 8467.01it/s]  9%|▊         | 34521/400000 [00:04<00:42, 8518.65it/s]  9%|▉         | 35403/400000 [00:04<00:42, 8606.57it/s]  9%|▉         | 36267/400000 [00:04<00:42, 8612.34it/s]  9%|▉         | 37130/400000 [00:04<00:43, 8328.82it/s]  9%|▉         | 37966/400000 [00:04<00:43, 8331.12it/s] 10%|▉         | 38830/400000 [00:04<00:42, 8420.34it/s] 10%|▉         | 39674/400000 [00:05<00:42, 8396.81it/s] 10%|█         | 40515/400000 [00:05<00:43, 8234.24it/s] 10%|█         | 41351/400000 [00:05<00:43, 8268.54it/s] 11%|█         | 42179/400000 [00:05<00:43, 8221.25it/s] 11%|█         | 43002/400000 [00:05<00:43, 8143.22it/s] 11%|█         | 43818/400000 [00:05<00:44, 7957.05it/s] 11%|█         | 44643/400000 [00:05<00:44, 8041.46it/s] 11%|█▏        | 45492/400000 [00:05<00:43, 8170.28it/s] 12%|█▏        | 46311/400000 [00:05<00:43, 8081.48it/s] 12%|█▏        | 47181/400000 [00:06<00:42, 8255.86it/s] 12%|█▏        | 48009/400000 [00:06<00:42, 8251.26it/s] 12%|█▏        | 48836/400000 [00:06<00:42, 8223.74it/s] 12%|█▏        | 49660/400000 [00:06<00:42, 8191.45it/s] 13%|█▎        | 50480/400000 [00:06<00:43, 8084.99it/s] 13%|█▎        | 51336/400000 [00:06<00:42, 8220.40it/s] 13%|█▎        | 52180/400000 [00:06<00:41, 8281.73it/s] 13%|█▎        | 53035/400000 [00:06<00:41, 8357.66it/s] 13%|█▎        | 53893/400000 [00:06<00:41, 8421.83it/s] 14%|█▎        | 54736/400000 [00:06<00:41, 8319.36it/s] 14%|█▍        | 55579/400000 [00:07<00:41, 8349.55it/s] 14%|█▍        | 56442/400000 [00:07<00:40, 8431.22it/s] 14%|█▍        | 57286/400000 [00:07<00:40, 8424.16it/s] 15%|█▍        | 58148/400000 [00:07<00:40, 8480.27it/s] 15%|█▍        | 58997/400000 [00:07<00:42, 8102.34it/s] 15%|█▍        | 59857/400000 [00:07<00:41, 8244.01it/s] 15%|█▌        | 60724/400000 [00:07<00:40, 8364.25it/s] 15%|█▌        | 61589/400000 [00:07<00:40, 8445.90it/s] 16%|█▌        | 62436/400000 [00:07<00:39, 8449.98it/s] 16%|█▌        | 63283/400000 [00:07<00:40, 8262.29it/s] 16%|█▌        | 64132/400000 [00:08<00:40, 8308.40it/s] 16%|█▌        | 64971/400000 [00:08<00:40, 8331.93it/s] 16%|█▋        | 65807/400000 [00:08<00:40, 8339.24it/s] 17%|█▋        | 66686/400000 [00:08<00:39, 8468.38it/s] 17%|█▋        | 67534/400000 [00:08<00:39, 8412.96it/s] 17%|█▋        | 68401/400000 [00:08<00:39, 8485.85it/s] 17%|█▋        | 69251/400000 [00:08<00:38, 8485.96it/s] 18%|█▊        | 70126/400000 [00:08<00:38, 8561.08it/s] 18%|█▊        | 70983/400000 [00:08<00:38, 8547.32it/s] 18%|█▊        | 71839/400000 [00:08<00:39, 8333.93it/s] 18%|█▊        | 72691/400000 [00:09<00:39, 8387.54it/s] 18%|█▊        | 73563/400000 [00:09<00:38, 8483.48it/s] 19%|█▊        | 74428/400000 [00:09<00:38, 8531.66it/s] 19%|█▉        | 75297/400000 [00:09<00:37, 8578.47it/s] 19%|█▉        | 76156/400000 [00:09<00:38, 8382.56it/s] 19%|█▉        | 76996/400000 [00:09<00:38, 8356.06it/s] 19%|█▉        | 77843/400000 [00:09<00:38, 8388.29it/s] 20%|█▉        | 78700/400000 [00:09<00:38, 8439.20it/s] 20%|█▉        | 79545/400000 [00:09<00:38, 8432.98it/s] 20%|██        | 80389/400000 [00:09<00:38, 8332.00it/s] 20%|██        | 81248/400000 [00:10<00:37, 8405.66it/s] 21%|██        | 82090/400000 [00:10<00:38, 8293.98it/s] 21%|██        | 82921/400000 [00:10<00:39, 8073.40it/s] 21%|██        | 83731/400000 [00:10<00:39, 8077.76it/s] 21%|██        | 84553/400000 [00:10<00:38, 8119.69it/s] 21%|██▏       | 85425/400000 [00:10<00:37, 8289.37it/s] 22%|██▏       | 86271/400000 [00:10<00:37, 8335.75it/s] 22%|██▏       | 87141/400000 [00:10<00:37, 8441.77it/s] 22%|██▏       | 88007/400000 [00:10<00:36, 8505.11it/s] 22%|██▏       | 88859/400000 [00:10<00:36, 8422.46it/s] 22%|██▏       | 89703/400000 [00:11<00:37, 8256.93it/s] 23%|██▎       | 90542/400000 [00:11<00:37, 8294.61it/s] 23%|██▎       | 91413/400000 [00:11<00:36, 8414.98it/s] 23%|██▎       | 92262/400000 [00:11<00:36, 8436.36it/s] 23%|██▎       | 93107/400000 [00:11<00:37, 8285.94it/s] 23%|██▎       | 93961/400000 [00:11<00:36, 8358.51it/s] 24%|██▎       | 94825/400000 [00:11<00:36, 8438.43it/s] 24%|██▍       | 95693/400000 [00:11<00:35, 8508.89it/s] 24%|██▍       | 96545/400000 [00:11<00:36, 8378.27it/s] 24%|██▍       | 97384/400000 [00:12<00:36, 8317.21it/s] 25%|██▍       | 98246/400000 [00:12<00:35, 8403.81it/s] 25%|██▍       | 99120/400000 [00:12<00:35, 8500.25it/s] 25%|██▍       | 99982/400000 [00:12<00:35, 8534.34it/s] 25%|██▌       | 100837/400000 [00:12<00:35, 8489.35it/s] 25%|██▌       | 101687/400000 [00:12<00:35, 8374.45it/s] 26%|██▌       | 102527/400000 [00:12<00:35, 8380.72it/s] 26%|██▌       | 103366/400000 [00:12<00:35, 8326.35it/s] 26%|██▌       | 104237/400000 [00:12<00:35, 8435.22it/s] 26%|██▋       | 105082/400000 [00:12<00:35, 8350.79it/s] 26%|██▋       | 105918/400000 [00:13<00:35, 8305.01it/s] 27%|██▋       | 106781/400000 [00:13<00:34, 8399.64it/s] 27%|██▋       | 107641/400000 [00:13<00:34, 8458.32it/s] 27%|██▋       | 108509/400000 [00:13<00:34, 8521.54it/s] 27%|██▋       | 109374/400000 [00:13<00:33, 8557.92it/s] 28%|██▊       | 110231/400000 [00:13<00:34, 8389.98it/s] 28%|██▊       | 111077/400000 [00:13<00:34, 8409.59it/s] 28%|██▊       | 111939/400000 [00:13<00:34, 8469.80it/s] 28%|██▊       | 112790/400000 [00:13<00:33, 8479.89it/s] 28%|██▊       | 113639/400000 [00:13<00:33, 8422.85it/s] 29%|██▊       | 114482/400000 [00:14<00:34, 8287.75it/s] 29%|██▉       | 115338/400000 [00:14<00:34, 8366.80it/s] 29%|██▉       | 116176/400000 [00:14<00:34, 8221.25it/s] 29%|██▉       | 117029/400000 [00:14<00:34, 8309.83it/s] 29%|██▉       | 117887/400000 [00:14<00:33, 8388.19it/s] 30%|██▉       | 118727/400000 [00:14<00:33, 8281.44it/s] 30%|██▉       | 119570/400000 [00:14<00:33, 8324.20it/s] 30%|███       | 120423/400000 [00:14<00:33, 8384.62it/s] 30%|███       | 121263/400000 [00:14<00:33, 8273.75it/s] 31%|███       | 122093/400000 [00:14<00:33, 8280.04it/s] 31%|███       | 122922/400000 [00:15<00:33, 8237.04it/s] 31%|███       | 123788/400000 [00:15<00:33, 8357.25it/s] 31%|███       | 124651/400000 [00:15<00:32, 8435.47it/s] 31%|███▏      | 125496/400000 [00:15<00:33, 8278.77it/s] 32%|███▏      | 126326/400000 [00:15<00:34, 7932.54it/s] 32%|███▏      | 127143/400000 [00:15<00:34, 8001.82it/s] 32%|███▏      | 127986/400000 [00:15<00:33, 8124.63it/s] 32%|███▏      | 128816/400000 [00:15<00:33, 8174.06it/s] 32%|███▏      | 129636/400000 [00:15<00:33, 8096.68it/s] 33%|███▎      | 130448/400000 [00:15<00:33, 8033.20it/s] 33%|███▎      | 131253/400000 [00:16<00:33, 7995.74it/s] 33%|███▎      | 132081/400000 [00:16<00:33, 8077.37it/s] 33%|███▎      | 132890/400000 [00:16<00:33, 8073.10it/s] 33%|███▎      | 133707/400000 [00:16<00:32, 8100.17it/s] 34%|███▎      | 134552/400000 [00:16<00:32, 8200.72it/s] 34%|███▍      | 135382/400000 [00:16<00:32, 8228.90it/s] 34%|███▍      | 136230/400000 [00:16<00:31, 8300.70it/s] 34%|███▍      | 137061/400000 [00:16<00:31, 8276.50it/s] 34%|███▍      | 137909/400000 [00:16<00:31, 8334.85it/s] 35%|███▍      | 138743/400000 [00:16<00:31, 8291.27it/s] 35%|███▍      | 139573/400000 [00:17<00:31, 8184.04it/s] 35%|███▌      | 140437/400000 [00:17<00:31, 8315.15it/s] 35%|███▌      | 141296/400000 [00:17<00:30, 8395.17it/s] 36%|███▌      | 142137/400000 [00:17<00:30, 8348.59it/s] 36%|███▌      | 142973/400000 [00:17<00:31, 8139.35it/s] 36%|███▌      | 143789/400000 [00:17<00:31, 8020.43it/s] 36%|███▌      | 144593/400000 [00:17<00:31, 8011.16it/s] 36%|███▋      | 145455/400000 [00:17<00:31, 8183.94it/s] 37%|███▋      | 146288/400000 [00:17<00:30, 8225.46it/s] 37%|███▋      | 147112/400000 [00:18<00:30, 8178.37it/s] 37%|███▋      | 147935/400000 [00:18<00:30, 8192.24it/s] 37%|███▋      | 148755/400000 [00:18<00:30, 8126.45it/s] 37%|███▋      | 149601/400000 [00:18<00:30, 8221.87it/s] 38%|███▊      | 150432/400000 [00:18<00:30, 8245.49it/s] 38%|███▊      | 151258/400000 [00:18<00:30, 8199.21it/s] 38%|███▊      | 152079/400000 [00:18<00:30, 8169.51it/s] 38%|███▊      | 152899/400000 [00:18<00:30, 8175.91it/s] 38%|███▊      | 153719/400000 [00:18<00:30, 8182.03it/s] 39%|███▊      | 154538/400000 [00:18<00:30, 8146.64it/s] 39%|███▉      | 155353/400000 [00:19<00:30, 8042.49it/s] 39%|███▉      | 156185/400000 [00:19<00:30, 8121.19it/s] 39%|███▉      | 157036/400000 [00:19<00:29, 8233.65it/s] 39%|███▉      | 157897/400000 [00:19<00:29, 8340.81it/s] 40%|███▉      | 158770/400000 [00:19<00:28, 8453.24it/s] 40%|███▉      | 159617/400000 [00:19<00:28, 8409.35it/s] 40%|████      | 160460/400000 [00:19<00:28, 8415.44it/s] 40%|████      | 161303/400000 [00:19<00:28, 8418.20it/s] 41%|████      | 162173/400000 [00:19<00:27, 8500.55it/s] 41%|████      | 163024/400000 [00:19<00:27, 8489.03it/s] 41%|████      | 163874/400000 [00:20<00:28, 8265.55it/s] 41%|████      | 164703/400000 [00:20<00:28, 8243.12it/s] 41%|████▏     | 165552/400000 [00:20<00:28, 8314.05it/s] 42%|████▏     | 166416/400000 [00:20<00:27, 8407.36it/s] 42%|████▏     | 167267/400000 [00:20<00:27, 8435.38it/s] 42%|████▏     | 168112/400000 [00:20<00:28, 8089.58it/s] 42%|████▏     | 168925/400000 [00:20<00:28, 7983.09it/s] 42%|████▏     | 169775/400000 [00:20<00:28, 8129.22it/s] 43%|████▎     | 170648/400000 [00:20<00:27, 8298.14it/s] 43%|████▎     | 171502/400000 [00:20<00:27, 8369.06it/s] 43%|████▎     | 172341/400000 [00:21<00:27, 8315.49it/s] 43%|████▎     | 173178/400000 [00:21<00:27, 8330.93it/s] 44%|████▎     | 174040/400000 [00:21<00:26, 8412.05it/s] 44%|████▎     | 174883/400000 [00:21<00:26, 8371.16it/s] 44%|████▍     | 175740/400000 [00:21<00:26, 8428.29it/s] 44%|████▍     | 176584/400000 [00:21<00:26, 8407.58it/s] 44%|████▍     | 177444/400000 [00:21<00:26, 8463.57it/s] 45%|████▍     | 178291/400000 [00:21<00:26, 8397.22it/s] 45%|████▍     | 179146/400000 [00:21<00:26, 8440.50it/s] 45%|████▍     | 179991/400000 [00:21<00:26, 8380.89it/s] 45%|████▌     | 180832/400000 [00:22<00:26, 8388.45it/s] 45%|████▌     | 181672/400000 [00:22<00:26, 8215.59it/s] 46%|████▌     | 182525/400000 [00:22<00:26, 8306.67it/s] 46%|████▌     | 183385/400000 [00:22<00:25, 8390.36it/s] 46%|████▌     | 184240/400000 [00:22<00:25, 8435.15it/s] 46%|████▋     | 185094/400000 [00:22<00:25, 8465.63it/s] 46%|████▋     | 185942/400000 [00:22<00:25, 8321.38it/s] 47%|████▋     | 186793/400000 [00:22<00:25, 8375.82it/s] 47%|████▋     | 187663/400000 [00:22<00:25, 8469.07it/s] 47%|████▋     | 188511/400000 [00:22<00:25, 8454.12it/s] 47%|████▋     | 189365/400000 [00:23<00:24, 8477.34it/s] 48%|████▊     | 190214/400000 [00:23<00:24, 8393.59it/s] 48%|████▊     | 191054/400000 [00:23<00:25, 8316.85it/s] 48%|████▊     | 191900/400000 [00:23<00:24, 8356.56it/s] 48%|████▊     | 192737/400000 [00:23<00:25, 8136.78it/s] 48%|████▊     | 193553/400000 [00:23<00:25, 8001.69it/s] 49%|████▊     | 194355/400000 [00:23<00:25, 7998.12it/s] 49%|████▉     | 195156/400000 [00:23<00:26, 7873.61it/s] 49%|████▉     | 196006/400000 [00:23<00:25, 8050.11it/s] 49%|████▉     | 196859/400000 [00:24<00:24, 8185.97it/s] 49%|████▉     | 197680/400000 [00:24<00:24, 8166.41it/s] 50%|████▉     | 198498/400000 [00:24<00:24, 8146.13it/s] 50%|████▉     | 199317/400000 [00:24<00:24, 8158.77it/s] 50%|█████     | 200170/400000 [00:24<00:24, 8265.63it/s] 50%|█████     | 201023/400000 [00:24<00:23, 8341.06it/s] 50%|█████     | 201858/400000 [00:24<00:23, 8285.00it/s] 51%|█████     | 202688/400000 [00:24<00:23, 8229.30it/s] 51%|█████     | 203537/400000 [00:24<00:23, 8305.64it/s] 51%|█████     | 204376/400000 [00:24<00:23, 8329.83it/s] 51%|█████▏    | 205214/400000 [00:25<00:23, 8342.66it/s] 52%|█████▏    | 206058/400000 [00:25<00:23, 8371.35it/s] 52%|█████▏    | 206896/400000 [00:25<00:23, 8210.04it/s] 52%|█████▏    | 207718/400000 [00:25<00:23, 8089.76it/s] 52%|█████▏    | 208534/400000 [00:25<00:23, 8109.38it/s] 52%|█████▏    | 209383/400000 [00:25<00:23, 8217.95it/s] 53%|█████▎    | 210222/400000 [00:25<00:22, 8266.84it/s] 53%|█████▎    | 211050/400000 [00:25<00:23, 8112.61it/s] 53%|█████▎    | 211863/400000 [00:25<00:24, 7641.52it/s] 53%|█████▎    | 212637/400000 [00:25<00:24, 7670.76it/s] 53%|█████▎    | 213488/400000 [00:26<00:23, 7902.05it/s] 54%|█████▎    | 214323/400000 [00:26<00:23, 8029.28it/s] 54%|█████▍    | 215173/400000 [00:26<00:22, 8163.79it/s] 54%|█████▍    | 216016/400000 [00:26<00:22, 8240.73it/s] 54%|█████▍    | 216870/400000 [00:26<00:21, 8325.62it/s] 54%|█████▍    | 217714/400000 [00:26<00:21, 8358.65it/s] 55%|█████▍    | 218552/400000 [00:26<00:21, 8265.58it/s] 55%|█████▍    | 219397/400000 [00:26<00:21, 8319.40it/s] 55%|█████▌    | 220270/400000 [00:26<00:21, 8437.66it/s] 55%|█████▌    | 221115/400000 [00:26<00:21, 8304.05it/s] 55%|█████▌    | 221965/400000 [00:27<00:21, 8361.74it/s] 56%|█████▌    | 222803/400000 [00:27<00:21, 8354.58it/s] 56%|█████▌    | 223640/400000 [00:27<00:21, 8358.63it/s] 56%|█████▌    | 224491/400000 [00:27<00:20, 8402.91it/s] 56%|█████▋    | 225332/400000 [00:27<00:20, 8400.74it/s] 57%|█████▋    | 226173/400000 [00:27<00:20, 8325.35it/s] 57%|█████▋    | 227012/400000 [00:27<00:20, 8344.33it/s] 57%|█████▋    | 227847/400000 [00:27<00:20, 8323.79it/s] 57%|█████▋    | 228694/400000 [00:27<00:20, 8364.73it/s] 57%|█████▋    | 229539/400000 [00:27<00:20, 8387.70it/s] 58%|█████▊    | 230389/400000 [00:28<00:20, 8418.91it/s] 58%|█████▊    | 231232/400000 [00:28<00:20, 8365.29it/s] 58%|█████▊    | 232069/400000 [00:28<00:20, 8316.97it/s] 58%|█████▊    | 232934/400000 [00:28<00:19, 8413.80it/s] 58%|█████▊    | 233776/400000 [00:28<00:19, 8334.96it/s] 59%|█████▊    | 234610/400000 [00:28<00:19, 8303.45it/s] 59%|█████▉    | 235462/400000 [00:28<00:19, 8365.62it/s] 59%|█████▉    | 236299/400000 [00:28<00:19, 8301.93it/s] 59%|█████▉    | 237159/400000 [00:28<00:19, 8388.05it/s] 59%|█████▉    | 237999/400000 [00:29<00:19, 8191.56it/s] 60%|█████▉    | 238853/400000 [00:29<00:19, 8292.44it/s] 60%|█████▉    | 239711/400000 [00:29<00:19, 8375.66it/s] 60%|██████    | 240550/400000 [00:29<00:19, 8334.29it/s] 60%|██████    | 241417/400000 [00:29<00:18, 8431.36it/s] 61%|██████    | 242282/400000 [00:29<00:18, 8494.21it/s] 61%|██████    | 243151/400000 [00:29<00:18, 8550.18it/s] 61%|██████    | 244007/400000 [00:29<00:18, 8549.37it/s] 61%|██████    | 244863/400000 [00:29<00:18, 8401.47it/s] 61%|██████▏   | 245709/400000 [00:29<00:18, 8417.94it/s] 62%|██████▏   | 246570/400000 [00:30<00:18, 8473.27it/s] 62%|██████▏   | 247418/400000 [00:30<00:18, 8433.93it/s] 62%|██████▏   | 248262/400000 [00:30<00:18, 8420.71it/s] 62%|██████▏   | 249105/400000 [00:30<00:18, 8292.54it/s] 62%|██████▏   | 249969/400000 [00:30<00:17, 8391.44it/s] 63%|██████▎   | 250840/400000 [00:30<00:17, 8484.46it/s] 63%|██████▎   | 251690/400000 [00:30<00:17, 8421.93it/s] 63%|██████▎   | 252533/400000 [00:30<00:17, 8363.24it/s] 63%|██████▎   | 253370/400000 [00:30<00:17, 8208.30it/s] 64%|██████▎   | 254192/400000 [00:30<00:18, 7994.76it/s] 64%|██████▍   | 255023/400000 [00:31<00:17, 8086.10it/s] 64%|██████▍   | 255878/400000 [00:31<00:17, 8218.06it/s] 64%|██████▍   | 256740/400000 [00:31<00:17, 8332.80it/s] 64%|██████▍   | 257575/400000 [00:31<00:17, 8303.52it/s] 65%|██████▍   | 258428/400000 [00:31<00:16, 8369.49it/s] 65%|██████▍   | 259286/400000 [00:31<00:16, 8429.56it/s] 65%|██████▌   | 260151/400000 [00:31<00:16, 8492.67it/s] 65%|██████▌   | 261001/400000 [00:31<00:16, 8384.94it/s] 65%|██████▌   | 261841/400000 [00:31<00:16, 8274.48it/s] 66%|██████▌   | 262670/400000 [00:31<00:16, 8173.87it/s] 66%|██████▌   | 263511/400000 [00:32<00:16, 8241.62it/s] 66%|██████▌   | 264351/400000 [00:32<00:16, 8284.69it/s] 66%|██████▋   | 265181/400000 [00:32<00:16, 8239.89it/s] 67%|██████▋   | 266006/400000 [00:32<00:16, 8213.43it/s] 67%|██████▋   | 266851/400000 [00:32<00:16, 8281.78it/s] 67%|██████▋   | 267680/400000 [00:32<00:15, 8283.50it/s] 67%|██████▋   | 268509/400000 [00:32<00:15, 8280.23it/s] 67%|██████▋   | 269372/400000 [00:32<00:15, 8381.19it/s] 68%|██████▊   | 270211/400000 [00:32<00:15, 8302.25it/s] 68%|██████▊   | 271042/400000 [00:32<00:15, 8265.02it/s] 68%|██████▊   | 271874/400000 [00:33<00:15, 8279.94it/s] 68%|██████▊   | 272703/400000 [00:33<00:15, 8257.69it/s] 68%|██████▊   | 273529/400000 [00:33<00:15, 8104.47it/s] 69%|██████▊   | 274341/400000 [00:33<00:15, 7891.66it/s] 69%|██████▉   | 275179/400000 [00:33<00:15, 8028.75it/s] 69%|██████▉   | 276019/400000 [00:33<00:15, 8134.25it/s] 69%|██████▉   | 276840/400000 [00:33<00:15, 8154.48it/s] 69%|██████▉   | 277667/400000 [00:33<00:14, 8187.04it/s] 70%|██████▉   | 278487/400000 [00:33<00:15, 8073.54it/s] 70%|██████▉   | 279296/400000 [00:33<00:15, 8040.17it/s] 70%|███████   | 280158/400000 [00:34<00:14, 8204.76it/s] 70%|███████   | 281019/400000 [00:34<00:14, 8318.93it/s] 70%|███████   | 281866/400000 [00:34<00:14, 8362.61it/s] 71%|███████   | 282704/400000 [00:34<00:14, 8216.49it/s] 71%|███████   | 283542/400000 [00:34<00:14, 8264.52it/s] 71%|███████   | 284409/400000 [00:34<00:13, 8380.35it/s] 71%|███████▏  | 285268/400000 [00:34<00:13, 8441.66it/s] 72%|███████▏  | 286114/400000 [00:34<00:13, 8369.25it/s] 72%|███████▏  | 286952/400000 [00:34<00:14, 8060.31it/s] 72%|███████▏  | 287761/400000 [00:35<00:13, 8024.71it/s] 72%|███████▏  | 288604/400000 [00:35<00:13, 8140.88it/s] 72%|███████▏  | 289452/400000 [00:35<00:13, 8237.41it/s] 73%|███████▎  | 290297/400000 [00:35<00:13, 8297.39it/s] 73%|███████▎  | 291128/400000 [00:35<00:13, 8022.27it/s] 73%|███████▎  | 291936/400000 [00:35<00:13, 8036.57it/s] 73%|███████▎  | 292773/400000 [00:35<00:13, 8133.67it/s] 73%|███████▎  | 293589/400000 [00:35<00:13, 8025.01it/s] 74%|███████▎  | 294427/400000 [00:35<00:12, 8126.63it/s] 74%|███████▍  | 295242/400000 [00:35<00:13, 7966.14it/s] 74%|███████▍  | 296041/400000 [00:36<00:13, 7882.99it/s] 74%|███████▍  | 296831/400000 [00:36<00:13, 7887.43it/s] 74%|███████▍  | 297672/400000 [00:36<00:12, 8036.76it/s] 75%|███████▍  | 298531/400000 [00:36<00:12, 8192.54it/s] 75%|███████▍  | 299352/400000 [00:36<00:12, 8100.08it/s] 75%|███████▌  | 300192/400000 [00:36<00:12, 8186.86it/s] 75%|███████▌  | 301031/400000 [00:36<00:12, 8245.47it/s] 75%|███████▌  | 301857/400000 [00:36<00:11, 8235.28it/s] 76%|███████▌  | 302682/400000 [00:36<00:11, 8229.92it/s] 76%|███████▌  | 303506/400000 [00:36<00:12, 8015.69it/s] 76%|███████▌  | 304345/400000 [00:37<00:11, 8123.29it/s] 76%|███████▋  | 305189/400000 [00:37<00:11, 8215.75it/s] 77%|███████▋  | 306012/400000 [00:37<00:11, 8015.59it/s] 77%|███████▋  | 306816/400000 [00:37<00:11, 7948.92it/s] 77%|███████▋  | 307627/400000 [00:37<00:11, 7995.01it/s] 77%|███████▋  | 308428/400000 [00:37<00:11, 7837.22it/s] 77%|███████▋  | 309268/400000 [00:37<00:11, 7995.60it/s] 78%|███████▊  | 310081/400000 [00:37<00:11, 8034.07it/s] 78%|███████▊  | 310888/400000 [00:37<00:11, 8042.97it/s] 78%|███████▊  | 311694/400000 [00:37<00:11, 7934.85it/s] 78%|███████▊  | 312489/400000 [00:38<00:11, 7901.79it/s] 78%|███████▊  | 313287/400000 [00:38<00:10, 7922.50it/s] 79%|███████▊  | 314102/400000 [00:38<00:10, 7988.88it/s] 79%|███████▊  | 314937/400000 [00:38<00:10, 8092.37it/s] 79%|███████▉  | 315747/400000 [00:38<00:10, 8048.48it/s] 79%|███████▉  | 316606/400000 [00:38<00:10, 8202.54it/s] 79%|███████▉  | 317452/400000 [00:38<00:09, 8277.24it/s] 80%|███████▉  | 318314/400000 [00:38<00:09, 8374.49it/s] 80%|███████▉  | 319153/400000 [00:38<00:09, 8368.28it/s] 80%|███████▉  | 319991/400000 [00:38<00:09, 8254.14it/s] 80%|████████  | 320818/400000 [00:39<00:09, 8113.91it/s] 80%|████████  | 321648/400000 [00:39<00:09, 8168.59it/s] 81%|████████  | 322497/400000 [00:39<00:09, 8261.20it/s] 81%|████████  | 323346/400000 [00:39<00:09, 8327.77it/s] 81%|████████  | 324180/400000 [00:39<00:09, 8266.63it/s] 81%|████████▏ | 325024/400000 [00:39<00:09, 8315.54it/s] 81%|████████▏ | 325859/400000 [00:39<00:08, 8324.91it/s] 82%|████████▏ | 326728/400000 [00:39<00:08, 8430.62it/s] 82%|████████▏ | 327580/400000 [00:39<00:08, 8455.85it/s] 82%|████████▏ | 328426/400000 [00:40<00:08, 8322.55it/s] 82%|████████▏ | 329285/400000 [00:40<00:08, 8399.41it/s] 83%|████████▎ | 330144/400000 [00:40<00:08, 8454.24it/s] 83%|████████▎ | 331017/400000 [00:40<00:08, 8532.96it/s] 83%|████████▎ | 331871/400000 [00:40<00:07, 8531.47it/s] 83%|████████▎ | 332725/400000 [00:40<00:07, 8467.54it/s] 83%|████████▎ | 333573/400000 [00:40<00:07, 8455.91it/s] 84%|████████▎ | 334446/400000 [00:40<00:07, 8536.23it/s] 84%|████████▍ | 335313/400000 [00:40<00:07, 8573.47it/s] 84%|████████▍ | 336171/400000 [00:40<00:07, 8511.67it/s] 84%|████████▍ | 337023/400000 [00:41<00:07, 8424.61it/s] 84%|████████▍ | 337877/400000 [00:41<00:07, 8458.60it/s] 85%|████████▍ | 338724/400000 [00:41<00:07, 8292.75it/s] 85%|████████▍ | 339555/400000 [00:41<00:07, 8173.40it/s] 85%|████████▌ | 340374/400000 [00:41<00:07, 8131.57it/s] 85%|████████▌ | 341188/400000 [00:41<00:07, 8126.87it/s] 86%|████████▌ | 342059/400000 [00:41<00:06, 8291.26it/s] 86%|████████▌ | 342921/400000 [00:41<00:06, 8386.19it/s] 86%|████████▌ | 343781/400000 [00:41<00:06, 8447.25it/s] 86%|████████▌ | 344627/400000 [00:41<00:06, 8429.71it/s] 86%|████████▋ | 345471/400000 [00:42<00:06, 8355.93it/s] 87%|████████▋ | 346324/400000 [00:42<00:06, 8407.34it/s] 87%|████████▋ | 347194/400000 [00:42<00:06, 8491.09it/s] 87%|████████▋ | 348067/400000 [00:42<00:06, 8561.23it/s] 87%|████████▋ | 348924/400000 [00:42<00:06, 8333.06it/s] 87%|████████▋ | 349760/400000 [00:42<00:06, 8162.03it/s] 88%|████████▊ | 350589/400000 [00:42<00:06, 8198.44it/s] 88%|████████▊ | 351411/400000 [00:42<00:05, 8165.41it/s] 88%|████████▊ | 352229/400000 [00:42<00:05, 8079.10it/s] 88%|████████▊ | 353038/400000 [00:42<00:05, 7964.56it/s] 88%|████████▊ | 353836/400000 [00:43<00:06, 7562.72it/s] 89%|████████▊ | 354660/400000 [00:43<00:05, 7752.84it/s] 89%|████████▉ | 355525/400000 [00:43<00:05, 8001.54it/s] 89%|████████▉ | 356392/400000 [00:43<00:05, 8189.80it/s] 89%|████████▉ | 357269/400000 [00:43<00:05, 8354.34it/s] 90%|████████▉ | 358109/400000 [00:43<00:05, 8304.47it/s] 90%|████████▉ | 358988/400000 [00:43<00:04, 8443.00it/s] 90%|████████▉ | 359852/400000 [00:43<00:04, 8500.23it/s] 90%|█████████ | 360711/400000 [00:43<00:04, 8523.06it/s] 90%|█████████ | 361565/400000 [00:43<00:04, 8454.75it/s] 91%|█████████ | 362412/400000 [00:44<00:04, 8347.97it/s] 91%|█████████ | 363248/400000 [00:44<00:04, 8293.08it/s] 91%|█████████ | 364080/400000 [00:44<00:04, 8299.56it/s] 91%|█████████ | 364932/400000 [00:44<00:04, 8361.53it/s] 91%|█████████▏| 365769/400000 [00:44<00:04, 8349.35it/s] 92%|█████████▏| 366605/400000 [00:44<00:04, 8283.15it/s] 92%|█████████▏| 367459/400000 [00:44<00:03, 8356.74it/s] 92%|█████████▏| 368321/400000 [00:44<00:03, 8432.35it/s] 92%|█████████▏| 369184/400000 [00:44<00:03, 8488.46it/s] 93%|█████████▎| 370034/400000 [00:44<00:03, 8455.37it/s] 93%|█████████▎| 370880/400000 [00:45<00:03, 8377.82it/s] 93%|█████████▎| 371735/400000 [00:45<00:03, 8428.46it/s] 93%|█████████▎| 372579/400000 [00:45<00:03, 8378.74it/s] 93%|█████████▎| 373418/400000 [00:45<00:03, 8297.82it/s] 94%|█████████▎| 374253/400000 [00:45<00:03, 8313.36it/s] 94%|█████████▍| 375085/400000 [00:45<00:03, 8223.74it/s] 94%|█████████▍| 375935/400000 [00:45<00:02, 8304.36it/s] 94%|█████████▍| 376793/400000 [00:45<00:02, 8384.47it/s] 94%|█████████▍| 377632/400000 [00:45<00:02, 8296.09it/s] 95%|█████████▍| 378463/400000 [00:46<00:02, 8117.04it/s] 95%|█████████▍| 379276/400000 [00:46<00:02, 8021.20it/s] 95%|█████████▌| 380130/400000 [00:46<00:02, 8167.92it/s] 95%|█████████▌| 380949/400000 [00:46<00:02, 8067.52it/s] 95%|█████████▌| 381758/400000 [00:46<00:02, 7954.02it/s] 96%|█████████▌| 382555/400000 [00:46<00:02, 7947.43it/s] 96%|█████████▌| 383351/400000 [00:46<00:02, 7824.85it/s] 96%|█████████▌| 384136/400000 [00:46<00:02, 7832.33it/s] 96%|█████████▌| 384930/400000 [00:46<00:01, 7863.68it/s] 96%|█████████▋| 385740/400000 [00:46<00:01, 7931.49it/s] 97%|█████████▋| 386608/400000 [00:47<00:01, 8141.01it/s] 97%|█████████▋| 387424/400000 [00:47<00:01, 8083.39it/s] 97%|█████████▋| 388287/400000 [00:47<00:01, 8238.99it/s] 97%|█████████▋| 389153/400000 [00:47<00:01, 8360.80it/s] 98%|█████████▊| 390013/400000 [00:47<00:01, 8429.15it/s] 98%|█████████▊| 390879/400000 [00:47<00:01, 8496.38it/s] 98%|█████████▊| 391730/400000 [00:47<00:00, 8277.35it/s] 98%|█████████▊| 392560/400000 [00:47<00:00, 8237.89it/s] 98%|█████████▊| 393430/400000 [00:47<00:00, 8369.25it/s] 99%|█████████▊| 394269/400000 [00:47<00:00, 8258.28it/s] 99%|█████████▉| 395097/400000 [00:48<00:00, 8202.89it/s] 99%|█████████▉| 395919/400000 [00:48<00:00, 8187.59it/s] 99%|█████████▉| 396784/400000 [00:48<00:00, 8320.52it/s] 99%|█████████▉| 397631/400000 [00:48<00:00, 8363.50it/s]100%|█████████▉| 398507/400000 [00:48<00:00, 8477.34it/s]100%|█████████▉| 399378/400000 [00:48<00:00, 8544.61it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8224.90it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb59b6f1c50>, <torchtext.data.dataset.TabularDataset object at 0x7fb59b6f1da0>, <torchtext.vocab.Vocab object at 0x7fb59b6f1cc0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.40 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 20.15 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.90 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.90 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.37 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.37 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.74 file/s]2020-07-24 00:17:59.792207: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-24 00:17:59.796068: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-24 00:17:59.796229: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5637e564ef70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 00:17:59.796246: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 134036.05it/s] 45%|████▌     | 4464640/9912422 [00:00<00:28, 191232.83it/s]9920512it [00:00, 34561054.37it/s]                           
0it [00:00, ?it/s]32768it [00:00, 718188.18it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 132034.77it/s]1654784it [00:00, 9756514.24it/s]                          
0it [00:00, ?it/s]8192it [00:00, 128467.31it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
