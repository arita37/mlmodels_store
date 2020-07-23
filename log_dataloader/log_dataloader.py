
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fa49c2b0a60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fa49c2b0a60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fa506f1c510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fa506f1c510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fa52614aea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fa52614aea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa4b424f950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa4b424f950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa4b424f950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 27%|██▋       | 2629632/9912422 [00:00<00:00, 26290829.04it/s]9920512it [00:00, 33182883.17it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1490467.11it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 468472.68it/s]1654784it [00:00, 11490181.56it/s]                         
0it [00:00, ?it/s]8192it [00:00, 176158.62it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa492ceb5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa492cff898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fa4b424f598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fa4b424f598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fa4b424f598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:31,  1.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:31,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:31,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:30,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:30,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:29,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:29,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:02,  2.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:02,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:02,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:02,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:01,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:01,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:00,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:00,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:42,  3.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:42,  3.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:42,  3.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:42,  3.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:41,  3.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:41,  3.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:41,  3.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:40,  3.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:40,  3.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:28,  4.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:28,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:28,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:28,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:28,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:27,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:27,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:27,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:19,  6.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:19,  6.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:19,  6.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:19,  6.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:19,  6.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:18,  6.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:18,  6.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:18,  6.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:18,  6.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:13,  9.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:13,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:12,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:12,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:12,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:12,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:12,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:12,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 12.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 17.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 17.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 23.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 30.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 38.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 47.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 65.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 74.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 81.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 81.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 86.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 86.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 86.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 86.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 86.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 86.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.30s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 86.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 86.48 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.37s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 86.48 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.37s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.37s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.09 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.37s/ url]
0 examples [00:00, ? examples/s]2020-07-23 18:10:22.910515: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-23 18:10:22.923348: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-23 18:10:22.923529: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a64d4dffb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-23 18:10:22.923546: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.21 examples/s]112 examples [00:00,  3.16 examples/s]223 examples [00:00,  4.51 examples/s]338 examples [00:00,  6.43 examples/s]434 examples [00:00,  9.15 examples/s]541 examples [00:00, 13.03 examples/s]654 examples [00:01, 18.52 examples/s]767 examples [00:01, 26.27 examples/s]868 examples [00:01, 37.11 examples/s]980 examples [00:01, 52.28 examples/s]1085 examples [00:01, 73.11 examples/s]1192 examples [00:01, 101.45 examples/s]1306 examples [00:01, 139.56 examples/s]1421 examples [00:01, 189.48 examples/s]1534 examples [00:01, 252.50 examples/s]1647 examples [00:01, 329.08 examples/s]1759 examples [00:02, 414.70 examples/s]1871 examples [00:02, 510.97 examples/s]1982 examples [00:02, 605.59 examples/s]2092 examples [00:02, 699.48 examples/s]2204 examples [00:02, 787.23 examples/s]2316 examples [00:02, 864.14 examples/s]2428 examples [00:02, 927.35 examples/s]2541 examples [00:02, 978.42 examples/s]2653 examples [00:02, 1008.77 examples/s]2764 examples [00:02, 1013.40 examples/s]2873 examples [00:03, 1032.19 examples/s]2984 examples [00:03, 1053.49 examples/s]3097 examples [00:03, 1073.76 examples/s]3207 examples [00:03, 1057.06 examples/s]3315 examples [00:03, 1063.20 examples/s]3423 examples [00:03, 1051.91 examples/s]3533 examples [00:03, 1063.46 examples/s]3646 examples [00:03, 1080.35 examples/s]3755 examples [00:03, 1082.15 examples/s]3866 examples [00:04, 1088.75 examples/s]3976 examples [00:04, 1077.71 examples/s]4085 examples [00:04, 1079.17 examples/s]4194 examples [00:04, 1081.03 examples/s]4303 examples [00:04, 1066.91 examples/s]4410 examples [00:04, 1062.31 examples/s]4517 examples [00:04, 1046.24 examples/s]4629 examples [00:04, 1067.30 examples/s]4743 examples [00:04, 1086.29 examples/s]4856 examples [00:04, 1097.60 examples/s]4966 examples [00:05, 1084.70 examples/s]5075 examples [00:05, 1052.66 examples/s]5184 examples [00:05, 1063.20 examples/s]5294 examples [00:05, 1073.73 examples/s]5404 examples [00:05, 1076.15 examples/s]5512 examples [00:05, 1062.49 examples/s]5619 examples [00:05, 1019.71 examples/s]5722 examples [00:05, 999.80 examples/s] 5823 examples [00:05, 994.84 examples/s]5931 examples [00:05, 1018.61 examples/s]6034 examples [00:06, 1011.38 examples/s]6136 examples [00:06, 989.68 examples/s] 6241 examples [00:06, 1005.41 examples/s]6355 examples [00:06, 1040.70 examples/s]6465 examples [00:06, 1056.29 examples/s]6578 examples [00:06, 1075.77 examples/s]6692 examples [00:06, 1092.15 examples/s]6803 examples [00:06, 1095.86 examples/s]6917 examples [00:06, 1105.83 examples/s]7028 examples [00:06, 1105.83 examples/s]7139 examples [00:07, 1102.57 examples/s]7255 examples [00:07, 1116.81 examples/s]7367 examples [00:07, 1102.39 examples/s]7478 examples [00:07, 1104.64 examples/s]7589 examples [00:07, 1081.26 examples/s]7700 examples [00:07, 1087.10 examples/s]7812 examples [00:07, 1096.54 examples/s]7926 examples [00:07, 1106.17 examples/s]8038 examples [00:07, 1109.71 examples/s]8150 examples [00:08, 1108.16 examples/s]8261 examples [00:08, 1087.15 examples/s]8370 examples [00:08, 1059.84 examples/s]8477 examples [00:08, 1056.24 examples/s]8586 examples [00:08, 1064.69 examples/s]8693 examples [00:08, 1055.02 examples/s]8799 examples [00:08, 1032.17 examples/s]8908 examples [00:08, 1046.52 examples/s]9018 examples [00:08, 1059.66 examples/s]9127 examples [00:08, 1068.39 examples/s]9234 examples [00:09, 1067.25 examples/s]9343 examples [00:09, 1071.19 examples/s]9455 examples [00:09, 1084.74 examples/s]9564 examples [00:09, 1082.09 examples/s]9673 examples [00:09, 1052.62 examples/s]9784 examples [00:09, 1067.27 examples/s]9891 examples [00:09, 1048.36 examples/s]10000 examples [00:09, 1057.48 examples/s]10106 examples [00:09, 1009.56 examples/s]10216 examples [00:09, 1034.75 examples/s]10321 examples [00:10, 1039.15 examples/s]10429 examples [00:10, 1049.39 examples/s]10540 examples [00:10, 1065.02 examples/s]10649 examples [00:10, 1071.60 examples/s]10760 examples [00:10, 1080.68 examples/s]10869 examples [00:10, 1060.34 examples/s]10976 examples [00:10, 1024.37 examples/s]11084 examples [00:10, 1039.71 examples/s]11194 examples [00:10, 1056.18 examples/s]11300 examples [00:10, 1044.69 examples/s]11408 examples [00:11, 1053.48 examples/s]11515 examples [00:11, 1057.33 examples/s]11622 examples [00:11, 1060.50 examples/s]11729 examples [00:11, 1062.44 examples/s]11840 examples [00:11, 1072.67 examples/s]11948 examples [00:11, 1051.15 examples/s]12061 examples [00:11, 1070.85 examples/s]12169 examples [00:11, 1072.75 examples/s]12277 examples [00:11, 1067.50 examples/s]12388 examples [00:12, 1079.16 examples/s]12497 examples [00:12, 1073.82 examples/s]12606 examples [00:12, 1077.36 examples/s]12718 examples [00:12, 1089.13 examples/s]12831 examples [00:12, 1099.66 examples/s]12946 examples [00:12, 1113.60 examples/s]13058 examples [00:12, 1103.83 examples/s]13171 examples [00:12, 1110.29 examples/s]13285 examples [00:12, 1117.27 examples/s]13397 examples [00:12, 1117.93 examples/s]13509 examples [00:13, 1117.98 examples/s]13621 examples [00:13, 1094.14 examples/s]13733 examples [00:13, 1098.87 examples/s]13843 examples [00:13, 1091.05 examples/s]13953 examples [00:13, 1093.64 examples/s]14063 examples [00:13, 1092.31 examples/s]14174 examples [00:13, 1096.03 examples/s]14284 examples [00:13, 1091.53 examples/s]14397 examples [00:13, 1100.65 examples/s]14509 examples [00:13, 1104.00 examples/s]14620 examples [00:14, 1102.19 examples/s]14731 examples [00:14, 1096.77 examples/s]14841 examples [00:14, 1086.44 examples/s]14952 examples [00:14, 1091.20 examples/s]15063 examples [00:14, 1096.21 examples/s]15173 examples [00:14, 1055.65 examples/s]15279 examples [00:14, 1052.63 examples/s]15385 examples [00:14, 1046.03 examples/s]15492 examples [00:14, 1052.97 examples/s]15598 examples [00:14, 1049.03 examples/s]15708 examples [00:15, 1062.97 examples/s]15815 examples [00:15, 1061.70 examples/s]15929 examples [00:15, 1082.45 examples/s]16041 examples [00:15, 1092.76 examples/s]16151 examples [00:15, 1093.21 examples/s]16262 examples [00:15, 1097.55 examples/s]16372 examples [00:15, 1086.80 examples/s]16481 examples [00:15, 1087.42 examples/s]16590 examples [00:15, 1024.28 examples/s]16701 examples [00:15, 1048.18 examples/s]16807 examples [00:16, 1051.24 examples/s]16916 examples [00:16, 1060.41 examples/s]17024 examples [00:16, 1065.48 examples/s]17135 examples [00:16, 1076.05 examples/s]17246 examples [00:16, 1084.57 examples/s]17356 examples [00:16, 1088.96 examples/s]17468 examples [00:16, 1097.31 examples/s]17580 examples [00:16, 1101.85 examples/s]17694 examples [00:16, 1111.06 examples/s]17806 examples [00:16, 1107.51 examples/s]17917 examples [00:17, 1106.83 examples/s]18028 examples [00:17, 1099.05 examples/s]18139 examples [00:17, 1101.58 examples/s]18251 examples [00:17, 1105.57 examples/s]18362 examples [00:17, 1103.96 examples/s]18474 examples [00:17, 1107.93 examples/s]18585 examples [00:17, 1095.18 examples/s]18697 examples [00:17, 1100.97 examples/s]18808 examples [00:17, 1102.61 examples/s]18919 examples [00:18, 1103.36 examples/s]19030 examples [00:18, 1088.98 examples/s]19142 examples [00:18, 1095.69 examples/s]19253 examples [00:18, 1097.78 examples/s]19365 examples [00:18, 1103.86 examples/s]19476 examples [00:18, 1100.26 examples/s]19587 examples [00:18, 1098.36 examples/s]19699 examples [00:18, 1101.80 examples/s]19811 examples [00:18, 1106.21 examples/s]19923 examples [00:18, 1110.04 examples/s]20035 examples [00:19, 1053.36 examples/s]20143 examples [00:19, 1059.92 examples/s]20253 examples [00:19, 1069.64 examples/s]20362 examples [00:19, 1073.12 examples/s]20473 examples [00:19, 1082.99 examples/s]20582 examples [00:19, 1080.08 examples/s]20692 examples [00:19, 1084.77 examples/s]20801 examples [00:19, 1077.11 examples/s]20914 examples [00:19, 1092.15 examples/s]21025 examples [00:19, 1095.41 examples/s]21135 examples [00:20, 1065.08 examples/s]21242 examples [00:20, 1059.65 examples/s]21353 examples [00:20, 1072.45 examples/s]21463 examples [00:20, 1080.10 examples/s]21576 examples [00:20, 1092.83 examples/s]21690 examples [00:20, 1105.09 examples/s]21801 examples [00:20, 1099.48 examples/s]21912 examples [00:20, 1102.09 examples/s]22023 examples [00:20, 1092.78 examples/s]22133 examples [00:20, 1067.07 examples/s]22244 examples [00:21, 1079.30 examples/s]22355 examples [00:21, 1086.78 examples/s]22466 examples [00:21, 1093.44 examples/s]22578 examples [00:21, 1100.19 examples/s]22689 examples [00:21, 1099.30 examples/s]22799 examples [00:21, 1096.96 examples/s]22909 examples [00:21, 1068.31 examples/s]23020 examples [00:21, 1078.06 examples/s]23128 examples [00:21, 1064.06 examples/s]23238 examples [00:21, 1074.15 examples/s]23350 examples [00:22, 1087.37 examples/s]23459 examples [00:22, 1063.80 examples/s]23570 examples [00:22, 1075.00 examples/s]23681 examples [00:22, 1083.32 examples/s]23794 examples [00:22, 1094.30 examples/s]23909 examples [00:22, 1107.65 examples/s]24020 examples [00:22, 1107.35 examples/s]24134 examples [00:22, 1114.71 examples/s]24249 examples [00:22, 1122.19 examples/s]24362 examples [00:23, 1109.32 examples/s]24474 examples [00:23, 1069.45 examples/s]24582 examples [00:23, 1070.07 examples/s]24693 examples [00:23, 1080.24 examples/s]24805 examples [00:23, 1091.15 examples/s]24915 examples [00:23, 1093.56 examples/s]25025 examples [00:23, 1094.67 examples/s]25135 examples [00:23, 1081.33 examples/s]25244 examples [00:23, 1081.28 examples/s]25354 examples [00:23, 1086.59 examples/s]25465 examples [00:24, 1092.59 examples/s]25578 examples [00:24, 1103.52 examples/s]25690 examples [00:24, 1105.87 examples/s]25801 examples [00:24, 1082.59 examples/s]25915 examples [00:24, 1098.81 examples/s]26028 examples [00:24, 1105.26 examples/s]26141 examples [00:24, 1112.24 examples/s]26253 examples [00:24, 1107.65 examples/s]26366 examples [00:24, 1112.19 examples/s]26481 examples [00:24, 1121.34 examples/s]26594 examples [00:25, 1118.34 examples/s]26706 examples [00:25, 1114.31 examples/s]26819 examples [00:25, 1118.22 examples/s]26932 examples [00:25, 1120.58 examples/s]27046 examples [00:25, 1123.46 examples/s]27159 examples [00:25, 1124.01 examples/s]27272 examples [00:25, 1121.18 examples/s]27385 examples [00:25, 1121.95 examples/s]27498 examples [00:25, 1111.14 examples/s]27611 examples [00:25, 1116.50 examples/s]27723 examples [00:26, 1111.39 examples/s]27835 examples [00:26, 1107.32 examples/s]27947 examples [00:26, 1109.88 examples/s]28059 examples [00:26, 1074.88 examples/s]28170 examples [00:26, 1084.73 examples/s]28281 examples [00:26, 1091.04 examples/s]28394 examples [00:26, 1102.22 examples/s]28507 examples [00:26, 1109.97 examples/s]28619 examples [00:26, 1104.18 examples/s]28732 examples [00:26, 1108.96 examples/s]28844 examples [00:27, 1110.48 examples/s]28956 examples [00:27, 1106.62 examples/s]29069 examples [00:27, 1112.52 examples/s]29182 examples [00:27, 1115.36 examples/s]29298 examples [00:27, 1125.43 examples/s]29411 examples [00:27, 1125.63 examples/s]29526 examples [00:27, 1130.58 examples/s]29640 examples [00:27, 1130.76 examples/s]29754 examples [00:27, 1090.13 examples/s]29867 examples [00:27, 1098.89 examples/s]29978 examples [00:28, 1084.97 examples/s]30087 examples [00:28, 1038.08 examples/s]30200 examples [00:28, 1061.52 examples/s]30309 examples [00:28, 1068.51 examples/s]30417 examples [00:28, 1068.60 examples/s]30530 examples [00:28, 1084.98 examples/s]30644 examples [00:28, 1099.15 examples/s]30756 examples [00:28, 1103.04 examples/s]30867 examples [00:28, 1086.14 examples/s]30979 examples [00:29, 1095.88 examples/s]31091 examples [00:29, 1101.18 examples/s]31202 examples [00:29, 1097.43 examples/s]31312 examples [00:29, 1094.80 examples/s]31422 examples [00:29, 1089.52 examples/s]31531 examples [00:29, 1078.80 examples/s]31644 examples [00:29, 1091.57 examples/s]31754 examples [00:29, 1088.60 examples/s]31866 examples [00:29, 1095.47 examples/s]31977 examples [00:29, 1098.78 examples/s]32091 examples [00:30, 1107.79 examples/s]32203 examples [00:30, 1109.84 examples/s]32315 examples [00:30, 1091.75 examples/s]32427 examples [00:30, 1099.23 examples/s]32537 examples [00:30, 1098.96 examples/s]32647 examples [00:30, 1077.06 examples/s]32755 examples [00:30, 1067.98 examples/s]32869 examples [00:30, 1086.87 examples/s]32978 examples [00:30, 1087.76 examples/s]33091 examples [00:30, 1098.88 examples/s]33203 examples [00:31, 1104.20 examples/s]33314 examples [00:31, 1091.12 examples/s]33426 examples [00:31, 1097.57 examples/s]33536 examples [00:31, 1097.30 examples/s]33646 examples [00:31, 1073.89 examples/s]33754 examples [00:31, 1061.91 examples/s]33867 examples [00:31, 1080.67 examples/s]33979 examples [00:31, 1091.17 examples/s]34090 examples [00:31, 1095.71 examples/s]34202 examples [00:31, 1100.10 examples/s]34315 examples [00:32, 1108.24 examples/s]34429 examples [00:32, 1116.29 examples/s]34542 examples [00:32, 1119.93 examples/s]34655 examples [00:32, 1098.44 examples/s]34765 examples [00:32, 1071.20 examples/s]34879 examples [00:32, 1088.59 examples/s]34989 examples [00:32, 1076.12 examples/s]35097 examples [00:32, 1072.62 examples/s]35207 examples [00:32, 1080.66 examples/s]35322 examples [00:32, 1098.78 examples/s]35433 examples [00:33, 1075.25 examples/s]35546 examples [00:33, 1089.37 examples/s]35658 examples [00:33, 1097.55 examples/s]35770 examples [00:33, 1103.22 examples/s]35881 examples [00:33, 1100.40 examples/s]35992 examples [00:33, 1101.42 examples/s]36103 examples [00:33, 1102.82 examples/s]36216 examples [00:33, 1109.87 examples/s]36328 examples [00:33, 1100.84 examples/s]36439 examples [00:34, 1090.98 examples/s]36550 examples [00:34, 1093.81 examples/s]36660 examples [00:34, 1092.16 examples/s]36771 examples [00:34, 1096.90 examples/s]36881 examples [00:34, 1093.06 examples/s]36993 examples [00:34, 1097.84 examples/s]37105 examples [00:34, 1103.53 examples/s]37216 examples [00:34, 1103.35 examples/s]37327 examples [00:34, 1061.39 examples/s]37436 examples [00:34, 1069.19 examples/s]37546 examples [00:35, 1076.39 examples/s]37654 examples [00:35, 1075.04 examples/s]37768 examples [00:35, 1093.27 examples/s]37879 examples [00:35, 1097.31 examples/s]37991 examples [00:35, 1103.28 examples/s]38102 examples [00:35, 1070.85 examples/s]38210 examples [00:35, 1053.08 examples/s]38323 examples [00:35, 1073.59 examples/s]38432 examples [00:35, 1077.88 examples/s]38540 examples [00:35, 1071.90 examples/s]38648 examples [00:36, 1058.96 examples/s]38755 examples [00:36, 1061.79 examples/s]38869 examples [00:36, 1082.18 examples/s]38978 examples [00:36, 1084.16 examples/s]39090 examples [00:36, 1092.54 examples/s]39201 examples [00:36, 1096.27 examples/s]39311 examples [00:36, 1094.03 examples/s]39424 examples [00:36, 1102.52 examples/s]39536 examples [00:36, 1106.22 examples/s]39647 examples [00:36, 1097.81 examples/s]39759 examples [00:37, 1102.55 examples/s]39870 examples [00:37, 1095.64 examples/s]39983 examples [00:37, 1105.14 examples/s]40094 examples [00:37, 1038.28 examples/s]40203 examples [00:37, 1052.35 examples/s]40314 examples [00:37, 1066.90 examples/s]40424 examples [00:37, 1076.56 examples/s]40533 examples [00:37, 1064.19 examples/s]40645 examples [00:37, 1079.77 examples/s]40754 examples [00:38, 1080.39 examples/s]40863 examples [00:38, 1083.11 examples/s]40972 examples [00:38, 1077.65 examples/s]41082 examples [00:38, 1083.95 examples/s]41194 examples [00:38, 1092.15 examples/s]41305 examples [00:38, 1096.42 examples/s]41416 examples [00:38, 1099.95 examples/s]41527 examples [00:38, 1087.15 examples/s]41640 examples [00:38, 1098.49 examples/s]41751 examples [00:38, 1099.26 examples/s]41861 examples [00:39, 1075.33 examples/s]41970 examples [00:39, 1078.87 examples/s]42078 examples [00:39, 1071.21 examples/s]42187 examples [00:39, 1075.45 examples/s]42299 examples [00:39, 1088.05 examples/s]42410 examples [00:39, 1093.54 examples/s]42521 examples [00:39, 1097.21 examples/s]42631 examples [00:39, 1086.11 examples/s]42745 examples [00:39, 1100.12 examples/s]42856 examples [00:39, 1102.10 examples/s]42967 examples [00:40, 1076.84 examples/s]43075 examples [00:40, 1077.45 examples/s]43186 examples [00:40, 1084.42 examples/s]43295 examples [00:40, 1083.12 examples/s]43405 examples [00:40, 1085.83 examples/s]43514 examples [00:40, 1073.37 examples/s]43625 examples [00:40, 1081.38 examples/s]43735 examples [00:40, 1086.13 examples/s]43844 examples [00:40, 1086.88 examples/s]43956 examples [00:40, 1093.73 examples/s]44066 examples [00:41, 1089.89 examples/s]44176 examples [00:41, 1034.60 examples/s]44289 examples [00:41, 1060.35 examples/s]44402 examples [00:41, 1079.10 examples/s]44513 examples [00:41, 1085.95 examples/s]44622 examples [00:41, 1085.93 examples/s]44734 examples [00:41, 1095.03 examples/s]44844 examples [00:41, 1095.35 examples/s]44955 examples [00:41, 1097.19 examples/s]45066 examples [00:41, 1099.26 examples/s]45176 examples [00:42, 1092.42 examples/s]45286 examples [00:42, 1092.74 examples/s]45399 examples [00:42, 1101.62 examples/s]45511 examples [00:42, 1106.40 examples/s]45623 examples [00:42, 1107.99 examples/s]45734 examples [00:42, 1106.20 examples/s]45849 examples [00:42, 1117.26 examples/s]45962 examples [00:42, 1118.87 examples/s]46074 examples [00:42, 1118.54 examples/s]46188 examples [00:42, 1124.58 examples/s]46301 examples [00:43, 1123.84 examples/s]46414 examples [00:43, 1110.68 examples/s]46526 examples [00:43, 1095.54 examples/s]46636 examples [00:43, 1093.23 examples/s]46746 examples [00:43, 1095.12 examples/s]46857 examples [00:43, 1098.30 examples/s]46970 examples [00:43, 1105.77 examples/s]47081 examples [00:43, 1104.16 examples/s]47192 examples [00:43, 1105.34 examples/s]47303 examples [00:43, 1103.50 examples/s]47414 examples [00:44, 1092.93 examples/s]47524 examples [00:44, 1094.45 examples/s]47635 examples [00:44, 1097.67 examples/s]47745 examples [00:44, 1056.28 examples/s]47852 examples [00:44, 1059.49 examples/s]47961 examples [00:44, 1067.88 examples/s]48074 examples [00:44, 1084.49 examples/s]48186 examples [00:44, 1093.11 examples/s]48298 examples [00:44, 1098.94 examples/s]48410 examples [00:45, 1103.87 examples/s]48521 examples [00:45, 1097.32 examples/s]48633 examples [00:45, 1101.32 examples/s]48744 examples [00:45, 1093.96 examples/s]48854 examples [00:45, 1060.51 examples/s]48966 examples [00:45, 1075.22 examples/s]49074 examples [00:45, 1051.80 examples/s]49186 examples [00:45, 1069.52 examples/s]49298 examples [00:45, 1082.89 examples/s]49409 examples [00:45, 1088.85 examples/s]49519 examples [00:46, 1072.15 examples/s]49627 examples [00:46, 1061.16 examples/s]49739 examples [00:46, 1075.78 examples/s]49849 examples [00:46, 1080.96 examples/s]49962 examples [00:46, 1092.72 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6689/50000 [00:00<00:00, 66883.21 examples/s] 39%|███▉      | 19666/50000 [00:00<00:00, 78260.32 examples/s] 67%|██████▋   | 33386/50000 [00:00<00:00, 89837.79 examples/s] 95%|█████████▌| 47523/50000 [00:00<00:00, 100868.17 examples/s]                                                                0 examples [00:00, ? examples/s]95 examples [00:00, 942.06 examples/s]198 examples [00:00, 963.56 examples/s]304 examples [00:00, 988.23 examples/s]418 examples [00:00, 1029.17 examples/s]530 examples [00:00, 1052.58 examples/s]640 examples [00:00, 1064.92 examples/s]752 examples [00:00, 1080.77 examples/s]866 examples [00:00, 1097.87 examples/s]981 examples [00:00, 1111.79 examples/s]1091 examples [00:01, 1106.45 examples/s]1205 examples [00:01, 1114.02 examples/s]1315 examples [00:01, 1108.45 examples/s]1425 examples [00:01, 1079.95 examples/s]1538 examples [00:01, 1092.53 examples/s]1652 examples [00:01, 1102.86 examples/s]1766 examples [00:01, 1113.34 examples/s]1879 examples [00:01, 1117.29 examples/s]1993 examples [00:01, 1121.05 examples/s]2106 examples [00:01, 1105.31 examples/s]2220 examples [00:02, 1114.98 examples/s]2332 examples [00:02, 1095.19 examples/s]2443 examples [00:02, 1097.49 examples/s]2553 examples [00:02, 1079.94 examples/s]2662 examples [00:02, 1079.89 examples/s]2776 examples [00:02, 1094.61 examples/s]2889 examples [00:02, 1102.87 examples/s]3000 examples [00:02, 1092.38 examples/s]3113 examples [00:02, 1102.82 examples/s]3230 examples [00:02, 1119.65 examples/s]3343 examples [00:03, 1097.24 examples/s]3456 examples [00:03, 1106.35 examples/s]3567 examples [00:03, 1095.60 examples/s]3681 examples [00:03, 1106.76 examples/s]3796 examples [00:03, 1117.87 examples/s]3908 examples [00:03, 1109.78 examples/s]4023 examples [00:03, 1121.18 examples/s]4136 examples [00:03, 1116.98 examples/s]4251 examples [00:03, 1126.58 examples/s]4364 examples [00:03, 1084.54 examples/s]4479 examples [00:04, 1101.47 examples/s]4590 examples [00:04, 1102.92 examples/s]4701 examples [00:04, 1100.26 examples/s]4817 examples [00:04, 1115.53 examples/s]4929 examples [00:04, 1084.06 examples/s]5038 examples [00:04, 1072.96 examples/s]5152 examples [00:04, 1089.72 examples/s]5262 examples [00:04, 1081.73 examples/s]5378 examples [00:04, 1102.34 examples/s]5490 examples [00:04, 1105.48 examples/s]5601 examples [00:05, 1092.21 examples/s]5711 examples [00:05, 1091.89 examples/s]5823 examples [00:05, 1099.08 examples/s]5933 examples [00:05, 1091.05 examples/s]6046 examples [00:05, 1100.56 examples/s]6159 examples [00:05, 1107.90 examples/s]6273 examples [00:05, 1116.18 examples/s]6385 examples [00:05, 1101.66 examples/s]6500 examples [00:05, 1115.49 examples/s]6614 examples [00:06, 1121.80 examples/s]6729 examples [00:06, 1129.95 examples/s]6843 examples [00:06, 1123.37 examples/s]6956 examples [00:06, 1113.44 examples/s]7068 examples [00:06, 1103.54 examples/s]7179 examples [00:06, 1104.40 examples/s]7290 examples [00:06, 1091.06 examples/s]7403 examples [00:06, 1099.32 examples/s]7513 examples [00:06, 1098.54 examples/s]7624 examples [00:06, 1100.93 examples/s]7735 examples [00:07, 1098.94 examples/s]7848 examples [00:07, 1105.46 examples/s]7960 examples [00:07, 1107.80 examples/s]8071 examples [00:07, 1105.22 examples/s]8184 examples [00:07, 1110.88 examples/s]8296 examples [00:07, 1106.70 examples/s]8410 examples [00:07, 1113.83 examples/s]8524 examples [00:07, 1119.32 examples/s]8636 examples [00:07, 1118.46 examples/s]8748 examples [00:07, 1091.41 examples/s]8858 examples [00:08, 1089.65 examples/s]8971 examples [00:08, 1099.50 examples/s]9082 examples [00:08, 1094.87 examples/s]9192 examples [00:08, 1093.47 examples/s]9303 examples [00:08, 1096.56 examples/s]9415 examples [00:08, 1102.89 examples/s]9526 examples [00:08, 1079.94 examples/s]9635 examples [00:08, 1071.37 examples/s]9743 examples [00:08, 1055.14 examples/s]9852 examples [00:08, 1063.84 examples/s]9964 examples [00:09, 1079.72 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCCTS52/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCCTS52/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa4b424f950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa4b424f950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa4b424f950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa492d2ff28>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa4725c3208>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa4b424f950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa4b424f950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa4b424f950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa4725c3cc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa492cff4a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fa42cc4c2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fa42cc4c2f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fa42cc4c2f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fa42cc4c400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fa42cc4c400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fa42cc4c400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=18a9b16061286ebdbccfb7c40cbb798a189c94227fccb3a7973c133839fb990a
  Stored in directory: /tmp/pip-ephem-wheel-cache-txnbprwc/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:11:28, 9.90kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:09:59, 14.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:04:11, 19.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:27:22, 28.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.58M/862M [00:01<5:54:14, 40.4kB/s].vector_cache/glove.6B.zip:   1%|          | 7.78M/862M [00:01<4:06:51, 57.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:52:00, 82.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:01<1:59:42, 118kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.7M/862M [00:01<1:23:24, 168kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<58:14, 239kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.7M/862M [00:02<40:35, 341kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.0M/862M [00:02<28:25, 485kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.1M/862M [00:02<19:51, 690kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:02<13:57, 977kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.8M/862M [00:02<09:47, 1.38MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<07:28, 1.81MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<07:07, 1.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<08:57, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<07:07, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:05<05:12, 2.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<08:48, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<07:38, 1.75MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<05:39, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<06:52, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<07:36, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<05:56, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:09<04:19, 3.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<50:41, 261kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<36:48, 359kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<26:03, 506kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<21:17, 618kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<17:34, 748kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<12:51, 1.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:13<09:10, 1.43MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<12:16, 1.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<09:55, 1.32MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<07:15, 1.80MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<05:55, 2.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<8:35:09, 25.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<6:00:29, 36.1kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<4:13:48, 51.1kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<3:00:49, 71.6kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<2:07:09, 102kB/s] .vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:18<1:28:55, 145kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<1:07:43, 190kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<48:48, 264kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<34:23, 374kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<26:50, 478kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<20:06, 637kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:22<14:22, 890kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<13:01, 979kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<11:48, 1.08MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<08:48, 1.45MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<06:19, 2.01MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<10:50, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:53, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:32, 1.93MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<07:31, 1.68MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:49, 1.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:06, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:18, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:42, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:19, 2.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:55, 2.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:24, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:02, 3.08MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:47, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:34, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:07, 2.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:44, 3.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:46, 1.26MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:06, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:59, 2.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:03, 1.74MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:26, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:49, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:02, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:29, 2.22MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:08, 2.93MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:44, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:14, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<03:58, 3.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:37, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:22, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:04, 2.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:29, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:04, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:51, 3.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:27, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:14, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<04:56, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<03:36, 3.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<55:03, 215kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<39:44, 298kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<28:03, 421kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<22:21, 527kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<16:49, 700kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<12:03, 975kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<11:11, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<10:14, 1.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:45, 1.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:17, 1.60MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:19, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:42, 2.46MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:03, 2.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<7:38:17, 25.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<5:20:35, 36.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<3:45:38, 51.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<2:40:20, 71.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<1:52:38, 102kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<1:18:52, 146kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<58:37, 195kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<42:11, 271kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<29:45, 384kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<23:26, 486kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<18:47, 606kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<13:38, 833kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<09:46, 1.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<09:55, 1.14MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<08:07, 1.39MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:55, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:46, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:01, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:23, 2.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:55, 2.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<07:26, 1.50MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:20, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:43, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<05:54, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:23, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:56, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:37, 3.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:05, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:06, 1.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:30, 2.44MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:43, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:14, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<04:50, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:32, 3.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<07:03, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:03, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:30, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:41, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<06:11, 1.75MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:52, 2.22MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:08, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:47, 1.86MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:30, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:18, 3.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<07:04, 1.51MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:03, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:30, 2.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:38, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:03, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<03:48, 2.78MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:08, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:48, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:35, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:53, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:30, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<03:24, 3.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:49, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:31, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:18, 2.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:12, 3.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:26, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:53, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:41, 2.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:57, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:33, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:24, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<03:34, 2.86MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<6:19:26, 27.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<4:25:29, 38.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<3:06:47, 54.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<2:13:40, 76.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:34:04, 108kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<1:05:54, 154kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<48:27, 208kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<35:06, 287kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<24:53, 405kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<17:30, 573kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<18:39, 537kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<15:06, 663kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<11:04, 904kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<09:19, 1.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:32, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<05:29, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:08, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:18, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:50, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<03:29, 2.81MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<08:54, 1.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:14, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:18, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:58, 1.63MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:15, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:53, 1.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<03:31, 2.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<1:11:34, 135kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<51:04, 189kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<35:54, 269kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<27:17, 352kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<21:03, 456kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<15:12, 631kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<12:08, 786kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<09:27, 1.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<06:51, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<07:00, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:49, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:16, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<03:47, 2.48MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<1:19:40, 118kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<56:41, 166kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<39:49, 235kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<29:57, 312kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<22:52, 408kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<16:28, 566kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<11:33, 802kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<9:04:57, 17.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<6:22:11, 24.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<4:26:59, 34.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<3:08:19, 48.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<2:12:39, 69.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<1:32:49, 98.8kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<1:06:54, 137kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<48:39, 188kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<34:24, 265kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<24:07, 376kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<21:00, 431kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<15:39, 579kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<11:07, 812kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<09:52, 911kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<08:43, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<06:33, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<06:00, 1.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:06, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:46, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:15, 2.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<5:57:03, 24.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<4:10:11, 35.4kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<2:54:31, 50.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<2:05:19, 70.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<1:29:44, 98.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<1:03:14, 139kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<44:10, 198kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<38:14, 228kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<27:41, 315kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<19:34, 445kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<15:33, 557kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<12:40, 683kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<09:18, 929kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<06:35, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<1:14:36, 115kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<53:03, 162kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<37:15, 230kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<27:57, 305kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<20:25, 417kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<14:28, 587kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<12:03, 701kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<10:13, 827kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<07:31, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<05:19, 1.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<18:44, 447kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<13:58, 600kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<09:56, 841kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<08:52, 936kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:53, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:52, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<04:12, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<06:44, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:32, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:04, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:46, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:09, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:06, 2.62MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:04, 1.99MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:30, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:29, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:32, 3.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:08, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:07, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:46, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:30, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<03:57, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:58, 2.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<03:56, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:21, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:26, 2.29MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:40, 2.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<03:21, 2.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:32, 3.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:35, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:05, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:11, 2.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:19, 3.32MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<06:51, 1.12MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<05:36, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:06, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:39, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:01, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:00, 2.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:53, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:16, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:18, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:24, 3.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:01, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:17, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:09, 2.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:56, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:30, 2.11MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:38, 2.80MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:17, 3.20MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<5:04:37, 24.2kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<3:33:24, 34.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<2:28:39, 49.2kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<1:47:21, 67.9kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:16:44, 95.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<53:59, 135kB/s]   .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<37:38, 192kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<33:39, 215kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<24:17, 297kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<17:07, 420kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<13:35, 527kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<10:14, 698kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<07:18, 975kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:46, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:27, 1.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:59, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:26, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:48, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:48, 2.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:37, 1.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:58, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:07, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<02:15, 3.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<09:00, 764kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<06:59, 983kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<05:03, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:06, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:57, 1.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:45, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:42, 2.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<05:35, 1.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:36, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:23, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:54, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:24, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:18<02:31, 2.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:19, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:40, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:54, 2.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<03:04, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<02:49, 2.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:06, 3.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<02:59, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<02:44, 2.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:04, 3.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:58, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:42, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:03, 3.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:55, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:20, 1.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:39, 2.38MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:51, 2.19MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<02:37, 2.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<01:59, 3.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:50, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:16, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:33, 2.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<01:51, 3.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:40, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:53, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:52, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:25, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:40, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:52, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:04, 2.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<23:07, 259kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<16:47, 357kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<11:49, 504kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<08:41, 683kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<3:54:10, 25.3kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<2:43:31, 36.1kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<1:54:39, 51.1kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<1:21:34, 71.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<57:19, 102kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<39:53, 145kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<33:35, 172kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<24:07, 240kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<16:58, 339kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<13:03, 438kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<10:24, 550kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<07:34, 754kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<05:19, 1.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<12:13, 462kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<09:07, 619kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<06:29, 865kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:49, 958kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:37, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:22, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:39, 1.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:41, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:48, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:02, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:33, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:02, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:14, 2.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:49, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:04, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:23, 2.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:44, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:43, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:08, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:19, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:50, 1.85MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:31, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<01:53, 2.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:32, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:18, 2.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:44, 2.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:24, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:43, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:08, 2.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<01:33, 3.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<20:25, 247kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<14:47, 340kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<10:26, 480kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<08:24, 591kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<06:53, 720kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<05:01, 985kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<03:34, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<04:40, 1.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:46, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:45, 1.77MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:02, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:36, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:56, 2.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:28, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:41, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:04, 2.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:31, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:36, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:18, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<01:43, 2.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:17, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:32, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<01:59, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:25, 3.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<4:23:49, 17.3kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<3:04:52, 24.6kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<2:08:46, 35.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<1:30:28, 49.6kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<1:04:12, 69.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<45:02, 99.3kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<31:26, 141kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<3:08:19, 23.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<2:11:23, 33.6kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<1:31:44, 47.5kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<1:05:11, 66.8kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<45:44, 95.0kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<31:45, 136kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<24:55, 172kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<17:51, 240kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<12:30, 340kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<09:41, 435kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<07:38, 552kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<05:32, 759kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<03:52, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<4:03:06, 17.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<2:50:20, 24.3kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<1:58:35, 34.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<1:23:15, 49.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<59:04, 69.1kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<41:26, 98.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<28:41, 140kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<48:45, 82.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<34:28, 116kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<24:04, 166kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<17:37, 224kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<13:07, 300kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<09:19, 421kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<06:29, 598kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<11:28, 338kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<08:24, 460kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<05:57, 646kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<05:01, 759kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<04:17, 887kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:09, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<02:15, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<03:00, 1.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:29, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:48, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:06, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:15, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:45, 2.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:15, 2.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<26:36, 135kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<18:58, 190kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<13:17, 269kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<09:14, 383kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<24:01, 147kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<17:31, 202kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<12:22, 284kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<08:39, 403kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<07:14, 479kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<05:25, 639kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:51, 891kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:27, 982kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:45, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:00, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:10, 1.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:51, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:22, 2.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:43, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:32, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:08, 2.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:32, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:23, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:02, 3.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:27, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:39, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:17, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<00:57, 3.21MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:34, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:24, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:02, 2.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<00:55, 3.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<2:00:35, 24.8kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<1:24:18, 35.4kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<58:13, 50.5kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<42:00, 69.6kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<30:01, 97.3kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<21:05, 138kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<14:33, 197kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<13:13, 216kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<09:33, 298kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<06:42, 421kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<05:14, 532kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<04:13, 658kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<03:05, 897kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:33, 1.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:04, 1.31MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:29, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:39, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:42, 1.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:18, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<00:55, 2.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:05, 833kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:25, 1.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:44, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:48, 1.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:46, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:22, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<00:57, 2.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<48:08, 50.8kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<33:51, 72.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<23:30, 103kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<16:45, 142kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<11:56, 198kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<08:19, 281kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<06:17, 367kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<04:51, 474kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<03:29, 658kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<02:25, 928kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:59, 750kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:18, 966kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:39, 1.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:39, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:35, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:12, 1.79MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:51, 2.48MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:00, 1.05MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:37, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:09, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:16, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:05, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:47, 2.50MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:01, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:54, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:40, 2.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:54, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:50, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:37, 2.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:51, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:58, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:45, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:32, 3.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:09, 1.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:59, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:43, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:53, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:57, 1.75MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:44, 2.26MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:32, 3.03MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:51, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:45, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:33, 2.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:45, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:50, 1.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:39, 2.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:31, 2.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<54:23, 27.4kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<37:37, 39.2kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<25:43, 55.3kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<18:18, 77.6kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<12:48, 110kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<08:42, 157kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<06:54, 196kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<04:56, 272kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<03:25, 386kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:38, 487kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:58, 650kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<01:22, 910kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:13, 992kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:06, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:49, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:34, 2.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<02:06, 543kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:35, 719kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:06, 1.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:00, 1.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:55, 1.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:41, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:28, 2.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<06:48, 148kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<04:50, 207kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<03:19, 294kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<02:27, 382kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:54, 492kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:21, 682kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:55, 963kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:17, 678kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:59, 880kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:41, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:39, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:36, 1.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:27, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:18, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<05:29, 134kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<03:53, 188kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<02:38, 267kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:54, 350kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:28, 454kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:02, 629kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:40, 888kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<03:38, 165kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<02:34, 230kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<01:44, 326kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<01:15, 420kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:59, 535kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:41, 740kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:27, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:40, 690kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:30, 896kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:20, 1.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:18, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:18, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:08, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<01:12, 271kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:51, 372kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:33, 525kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:24, 635kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:19, 766kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:13, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:08, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:12, 903kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:09, 1.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.88MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.78MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 395/400000 [00:00<01:41, 3949.53it/s]  0%|          | 1259/400000 [00:00<01:24, 4717.67it/s]  1%|          | 2139/400000 [00:00<01:12, 5480.07it/s]  1%|          | 3031/400000 [00:00<01:04, 6196.70it/s]  1%|          | 3926/400000 [00:00<00:58, 6826.02it/s]  1%|          | 4826/400000 [00:00<00:53, 7357.31it/s]  1%|▏         | 5707/400000 [00:00<00:50, 7738.89it/s]  2%|▏         | 6559/400000 [00:00<00:49, 7957.71it/s]  2%|▏         | 7463/400000 [00:00<00:47, 8252.24it/s]  2%|▏         | 8371/400000 [00:01<00:46, 8484.05it/s]  2%|▏         | 9290/400000 [00:01<00:44, 8683.66it/s]  3%|▎         | 10179/400000 [00:01<00:44, 8742.39it/s]  3%|▎         | 11094/400000 [00:01<00:43, 8858.22it/s]  3%|▎         | 11986/400000 [00:01<00:43, 8854.92it/s]  3%|▎         | 12876/400000 [00:01<00:43, 8842.63it/s]  3%|▎         | 13776/400000 [00:01<00:43, 8888.59it/s]  4%|▎         | 14667/400000 [00:01<00:43, 8815.18it/s]  4%|▍         | 15567/400000 [00:01<00:43, 8869.57it/s]  4%|▍         | 16476/400000 [00:01<00:42, 8932.51it/s]  4%|▍         | 17371/400000 [00:02<00:43, 8857.39it/s]  5%|▍         | 18258/400000 [00:02<00:44, 8592.47it/s]  5%|▍         | 19120/400000 [00:02<00:44, 8539.91it/s]  5%|▍         | 19976/400000 [00:02<00:44, 8525.95it/s]  5%|▌         | 20871/400000 [00:02<00:43, 8648.07it/s]  5%|▌         | 21807/400000 [00:02<00:42, 8848.52it/s]  6%|▌         | 22699/400000 [00:02<00:42, 8868.07it/s]  6%|▌         | 23588/400000 [00:02<00:42, 8822.47it/s]  6%|▌         | 24487/400000 [00:02<00:42, 8870.10it/s]  6%|▋         | 25389/400000 [00:02<00:42, 8912.91it/s]  7%|▋         | 26283/400000 [00:03<00:41, 8920.25it/s]  7%|▋         | 27195/400000 [00:03<00:41, 8978.81it/s]  7%|▋         | 28094/400000 [00:03<00:41, 8904.40it/s]  7%|▋         | 28992/400000 [00:03<00:41, 8924.06it/s]  7%|▋         | 29901/400000 [00:03<00:41, 8973.03it/s]  8%|▊         | 30799/400000 [00:03<00:41, 8962.03it/s]  8%|▊         | 31723/400000 [00:03<00:40, 9041.33it/s]  8%|▊         | 32628/400000 [00:03<00:40, 9004.74it/s]  8%|▊         | 33532/400000 [00:03<00:40, 9015.19it/s]  9%|▊         | 34451/400000 [00:03<00:40, 9066.14it/s]  9%|▉         | 35358/400000 [00:04<00:40, 9057.15it/s]  9%|▉         | 36274/400000 [00:04<00:40, 9085.24it/s]  9%|▉         | 37183/400000 [00:04<00:40, 8961.26it/s] 10%|▉         | 38086/400000 [00:04<00:40, 8979.32it/s] 10%|▉         | 38985/400000 [00:04<00:40, 8893.25it/s] 10%|▉         | 39875/400000 [00:04<00:40, 8858.45it/s] 10%|█         | 40762/400000 [00:04<00:40, 8809.95it/s] 10%|█         | 41644/400000 [00:04<00:40, 8746.04it/s] 11%|█         | 42528/400000 [00:04<00:40, 8771.28it/s] 11%|█         | 43427/400000 [00:04<00:40, 8833.22it/s] 11%|█         | 44357/400000 [00:05<00:39, 8967.54it/s] 11%|█▏        | 45255/400000 [00:05<00:39, 8942.86it/s] 12%|█▏        | 46150/400000 [00:05<00:39, 8911.66it/s] 12%|█▏        | 47069/400000 [00:05<00:39, 8992.48it/s] 12%|█▏        | 47991/400000 [00:05<00:38, 9058.03it/s] 12%|█▏        | 48898/400000 [00:05<00:38, 9033.53it/s] 12%|█▏        | 49802/400000 [00:05<00:39, 8959.44it/s] 13%|█▎        | 50699/400000 [00:05<00:40, 8708.61it/s] 13%|█▎        | 51586/400000 [00:05<00:39, 8755.58it/s] 13%|█▎        | 52478/400000 [00:05<00:39, 8802.73it/s] 13%|█▎        | 53360/400000 [00:06<00:40, 8623.34it/s] 14%|█▎        | 54276/400000 [00:06<00:39, 8776.32it/s] 14%|█▍        | 55170/400000 [00:06<00:39, 8823.90it/s] 14%|█▍        | 56076/400000 [00:06<00:38, 8892.25it/s] 14%|█▍        | 57005/400000 [00:06<00:38, 9005.62it/s] 14%|█▍        | 57907/400000 [00:06<00:37, 9009.37it/s] 15%|█▍        | 58812/400000 [00:06<00:37, 9020.71it/s] 15%|█▍        | 59715/400000 [00:06<00:38, 8932.75it/s] 15%|█▌        | 60612/400000 [00:06<00:37, 8942.35it/s] 15%|█▌        | 61553/400000 [00:06<00:37, 9075.09it/s] 16%|█▌        | 62492/400000 [00:07<00:36, 9166.25it/s] 16%|█▌        | 63410/400000 [00:07<00:37, 9084.63it/s] 16%|█▌        | 64320/400000 [00:07<00:37, 8940.63it/s] 16%|█▋        | 65216/400000 [00:07<00:37, 8902.50it/s] 17%|█▋        | 66112/400000 [00:07<00:37, 8919.52it/s] 17%|█▋        | 67005/400000 [00:07<00:37, 8869.10it/s] 17%|█▋        | 67918/400000 [00:07<00:37, 8943.53it/s] 17%|█▋        | 68844/400000 [00:07<00:36, 9033.22it/s] 17%|█▋        | 69779/400000 [00:07<00:36, 9124.04it/s] 18%|█▊        | 70692/400000 [00:07<00:36, 9090.22it/s] 18%|█▊        | 71606/400000 [00:08<00:36, 9103.14it/s] 18%|█▊        | 72525/400000 [00:08<00:35, 9128.36it/s] 18%|█▊        | 73439/400000 [00:08<00:36, 9007.05it/s] 19%|█▊        | 74341/400000 [00:08<00:36, 8984.14it/s] 19%|█▉        | 75245/400000 [00:08<00:36, 8999.28it/s] 19%|█▉        | 76163/400000 [00:08<00:35, 9049.97it/s] 19%|█▉        | 77069/400000 [00:08<00:36, 8937.79it/s] 19%|█▉        | 77964/400000 [00:08<00:36, 8775.35it/s] 20%|█▉        | 78880/400000 [00:08<00:36, 8886.39it/s] 20%|█▉        | 79781/400000 [00:09<00:35, 8922.51it/s] 20%|██        | 80682/400000 [00:09<00:35, 8947.32it/s] 20%|██        | 81582/400000 [00:09<00:35, 8962.13it/s] 21%|██        | 82490/400000 [00:09<00:35, 8996.13it/s] 21%|██        | 83390/400000 [00:09<00:35, 8968.66it/s] 21%|██        | 84306/400000 [00:09<00:34, 9022.61it/s] 21%|██▏       | 85241/400000 [00:09<00:34, 9116.31it/s] 22%|██▏       | 86160/400000 [00:09<00:34, 9137.01it/s] 22%|██▏       | 87074/400000 [00:09<00:34, 9116.86it/s] 22%|██▏       | 87986/400000 [00:09<00:34, 9112.39it/s] 22%|██▏       | 88898/400000 [00:10<00:34, 9006.96it/s] 22%|██▏       | 89829/400000 [00:10<00:34, 9094.75it/s] 23%|██▎       | 90739/400000 [00:10<00:34, 9041.87it/s] 23%|██▎       | 91644/400000 [00:10<00:34, 9011.35it/s] 23%|██▎       | 92549/400000 [00:10<00:34, 9020.30it/s] 23%|██▎       | 93452/400000 [00:10<00:34, 9001.30it/s] 24%|██▎       | 94353/400000 [00:10<00:34, 8958.21it/s] 24%|██▍       | 95249/400000 [00:10<00:34, 8922.46it/s] 24%|██▍       | 96142/400000 [00:10<00:34, 8918.76it/s] 24%|██▍       | 97034/400000 [00:10<00:34, 8900.96it/s] 24%|██▍       | 97943/400000 [00:11<00:33, 8954.24it/s] 25%|██▍       | 98862/400000 [00:11<00:33, 9023.36it/s] 25%|██▍       | 99765/400000 [00:11<00:33, 8960.55it/s] 25%|██▌       | 100662/400000 [00:11<00:33, 8892.53it/s] 25%|██▌       | 101555/400000 [00:11<00:33, 8901.37it/s] 26%|██▌       | 102446/400000 [00:11<00:33, 8886.87it/s] 26%|██▌       | 103344/400000 [00:11<00:33, 8911.94it/s] 26%|██▌       | 104236/400000 [00:11<00:33, 8757.30it/s] 26%|██▋       | 105113/400000 [00:11<00:33, 8694.94it/s] 27%|██▋       | 106016/400000 [00:11<00:33, 8790.17it/s] 27%|██▋       | 106931/400000 [00:12<00:32, 8894.76it/s] 27%|██▋       | 107860/400000 [00:12<00:32, 9007.69it/s] 27%|██▋       | 108768/400000 [00:12<00:32, 9027.84it/s] 27%|██▋       | 109672/400000 [00:12<00:32, 8916.26it/s] 28%|██▊       | 110565/400000 [00:12<00:32, 8818.95it/s] 28%|██▊       | 111450/400000 [00:12<00:32, 8826.58it/s] 28%|██▊       | 112346/400000 [00:12<00:32, 8863.44it/s] 28%|██▊       | 113233/400000 [00:12<00:32, 8809.68it/s] 29%|██▊       | 114116/400000 [00:12<00:32, 8815.09it/s] 29%|██▉       | 115024/400000 [00:12<00:32, 8892.34it/s] 29%|██▉       | 115947/400000 [00:13<00:31, 8990.72it/s] 29%|██▉       | 116869/400000 [00:13<00:31, 9057.43it/s] 29%|██▉       | 117776/400000 [00:13<00:31, 8974.06it/s] 30%|██▉       | 118692/400000 [00:13<00:31, 9026.29it/s] 30%|██▉       | 119617/400000 [00:13<00:30, 9092.18it/s] 30%|███       | 120527/400000 [00:13<00:30, 9024.26it/s] 30%|███       | 121456/400000 [00:13<00:30, 9100.74it/s] 31%|███       | 122367/400000 [00:13<00:30, 9097.02it/s] 31%|███       | 123278/400000 [00:13<00:30, 9052.73it/s] 31%|███       | 124184/400000 [00:13<00:30, 9033.68it/s] 31%|███▏      | 125088/400000 [00:14<00:30, 8999.06it/s] 32%|███▏      | 126000/400000 [00:14<00:30, 9034.04it/s] 32%|███▏      | 126913/400000 [00:14<00:30, 9061.66it/s] 32%|███▏      | 127820/400000 [00:14<00:30, 9038.78it/s] 32%|███▏      | 128726/400000 [00:14<00:29, 9044.00it/s] 32%|███▏      | 129634/400000 [00:14<00:29, 9052.49it/s] 33%|███▎      | 130551/400000 [00:14<00:29, 9086.51it/s] 33%|███▎      | 131460/400000 [00:14<00:29, 9001.45it/s] 33%|███▎      | 132361/400000 [00:14<00:29, 8964.85it/s] 33%|███▎      | 133267/400000 [00:14<00:29, 8991.64it/s] 34%|███▎      | 134167/400000 [00:15<00:29, 8932.20it/s] 34%|███▍      | 135070/400000 [00:15<00:29, 8959.78it/s] 34%|███▍      | 135967/400000 [00:15<00:29, 8946.91it/s] 34%|███▍      | 136863/400000 [00:15<00:29, 8948.87it/s] 34%|███▍      | 137758/400000 [00:15<00:29, 8862.54it/s] 35%|███▍      | 138645/400000 [00:15<00:29, 8818.39it/s] 35%|███▍      | 139528/400000 [00:15<00:29, 8806.85it/s] 35%|███▌      | 140409/400000 [00:15<00:30, 8486.83it/s] 35%|███▌      | 141282/400000 [00:15<00:30, 8555.67it/s] 36%|███▌      | 142183/400000 [00:15<00:29, 8686.89it/s] 36%|███▌      | 143073/400000 [00:16<00:29, 8749.56it/s] 36%|███▌      | 143982/400000 [00:16<00:28, 8847.36it/s] 36%|███▌      | 144883/400000 [00:16<00:28, 8895.47it/s] 36%|███▋      | 145774/400000 [00:16<00:28, 8817.70it/s] 37%|███▋      | 146681/400000 [00:16<00:28, 8891.55it/s] 37%|███▋      | 147578/400000 [00:16<00:28, 8913.71it/s] 37%|███▋      | 148470/400000 [00:16<00:28, 8865.52it/s] 37%|███▋      | 149364/400000 [00:16<00:28, 8885.40it/s] 38%|███▊      | 150257/400000 [00:16<00:28, 8895.86it/s] 38%|███▊      | 151147/400000 [00:16<00:28, 8849.42it/s] 38%|███▊      | 152033/400000 [00:17<00:28, 8746.13it/s] 38%|███▊      | 152949/400000 [00:17<00:27, 8864.50it/s] 38%|███▊      | 153840/400000 [00:17<00:27, 8877.46it/s] 39%|███▊      | 154729/400000 [00:17<00:27, 8864.86it/s] 39%|███▉      | 155617/400000 [00:17<00:27, 8867.45it/s] 39%|███▉      | 156504/400000 [00:17<00:28, 8536.32it/s] 39%|███▉      | 157388/400000 [00:17<00:28, 8625.00it/s] 40%|███▉      | 158259/400000 [00:17<00:27, 8647.79it/s] 40%|███▉      | 159131/400000 [00:17<00:27, 8669.06it/s] 40%|████      | 160009/400000 [00:18<00:27, 8699.51it/s] 40%|████      | 160914/400000 [00:18<00:27, 8799.31it/s] 40%|████      | 161820/400000 [00:18<00:26, 8875.09it/s] 41%|████      | 162733/400000 [00:18<00:26, 8950.04it/s] 41%|████      | 163652/400000 [00:18<00:26, 9019.89it/s] 41%|████      | 164555/400000 [00:18<00:26, 8883.85it/s] 41%|████▏     | 165454/400000 [00:18<00:26, 8912.97it/s] 42%|████▏     | 166346/400000 [00:18<00:26, 8721.58it/s] 42%|████▏     | 167252/400000 [00:18<00:26, 8818.95it/s] 42%|████▏     | 168173/400000 [00:18<00:25, 8932.12it/s] 42%|████▏     | 169068/400000 [00:19<00:26, 8878.38it/s] 42%|████▏     | 169984/400000 [00:19<00:25, 8958.75it/s] 43%|████▎     | 170898/400000 [00:19<00:25, 9011.41it/s] 43%|████▎     | 171800/400000 [00:19<00:25, 8996.72it/s] 43%|████▎     | 172719/400000 [00:19<00:25, 9051.74it/s] 43%|████▎     | 173625/400000 [00:19<00:25, 8957.85it/s] 44%|████▎     | 174539/400000 [00:19<00:25, 9010.64it/s] 44%|████▍     | 175471/400000 [00:19<00:24, 9100.81it/s] 44%|████▍     | 176401/400000 [00:19<00:24, 9156.92it/s] 44%|████▍     | 177318/400000 [00:19<00:24, 9160.71it/s] 45%|████▍     | 178235/400000 [00:20<00:24, 8945.01it/s] 45%|████▍     | 179131/400000 [00:20<00:24, 8928.95it/s] 45%|████▌     | 180050/400000 [00:20<00:24, 9003.07it/s] 45%|████▌     | 180958/400000 [00:20<00:24, 9023.58it/s] 45%|████▌     | 181871/400000 [00:20<00:24, 9055.00it/s] 46%|████▌     | 182777/400000 [00:20<00:23, 9054.98it/s] 46%|████▌     | 183683/400000 [00:20<00:24, 8910.28it/s] 46%|████▌     | 184575/400000 [00:20<00:24, 8808.96it/s] 46%|████▋     | 185457/400000 [00:20<00:24, 8693.26it/s] 47%|████▋     | 186368/400000 [00:20<00:24, 8811.72it/s] 47%|████▋     | 187262/400000 [00:21<00:24, 8848.62it/s] 47%|████▋     | 188148/400000 [00:21<00:23, 8832.34it/s] 47%|████▋     | 189070/400000 [00:21<00:23, 8943.90it/s] 47%|████▋     | 189966/400000 [00:21<00:23, 8940.91it/s] 48%|████▊     | 190861/400000 [00:21<00:23, 8917.36it/s] 48%|████▊     | 191754/400000 [00:21<00:23, 8779.52it/s] 48%|████▊     | 192637/400000 [00:21<00:23, 8792.34it/s] 48%|████▊     | 193561/400000 [00:21<00:23, 8920.55it/s] 49%|████▊     | 194454/400000 [00:21<00:23, 8873.70it/s] 49%|████▉     | 195354/400000 [00:21<00:22, 8908.96it/s] 49%|████▉     | 196252/400000 [00:22<00:22, 8927.40it/s] 49%|████▉     | 197167/400000 [00:22<00:22, 8991.99it/s] 50%|████▉     | 198075/400000 [00:22<00:22, 9017.52it/s] 50%|████▉     | 198978/400000 [00:22<00:22, 8982.31it/s] 50%|████▉     | 199877/400000 [00:22<00:22, 8973.79it/s] 50%|█████     | 200775/400000 [00:22<00:22, 8931.37it/s] 50%|█████     | 201669/400000 [00:22<00:22, 8748.06it/s] 51%|█████     | 202545/400000 [00:22<00:23, 8521.88it/s] 51%|█████     | 203427/400000 [00:22<00:22, 8609.08it/s] 51%|█████     | 204329/400000 [00:22<00:22, 8726.74it/s] 51%|█████▏    | 205244/400000 [00:23<00:22, 8849.43it/s] 52%|█████▏    | 206152/400000 [00:23<00:21, 8916.13it/s] 52%|█████▏    | 207045/400000 [00:23<00:21, 8833.78it/s] 52%|█████▏    | 207957/400000 [00:23<00:21, 8917.08it/s] 52%|█████▏    | 208897/400000 [00:23<00:21, 9056.14it/s] 52%|█████▏    | 209804/400000 [00:23<00:21, 9020.91it/s] 53%|█████▎    | 210707/400000 [00:23<00:21, 9012.69it/s] 53%|█████▎    | 211628/400000 [00:23<00:20, 9069.84it/s] 53%|█████▎    | 212546/400000 [00:23<00:20, 9100.30it/s] 53%|█████▎    | 213458/400000 [00:23<00:20, 9103.50it/s] 54%|█████▎    | 214369/400000 [00:24<00:20, 9075.67it/s] 54%|█████▍    | 215283/400000 [00:24<00:20, 9093.86it/s] 54%|█████▍    | 216207/400000 [00:24<00:20, 9135.10it/s] 54%|█████▍    | 217121/400000 [00:24<00:20, 9112.75it/s] 55%|█████▍    | 218033/400000 [00:24<00:19, 9111.60it/s] 55%|█████▍    | 218945/400000 [00:24<00:19, 9076.09it/s] 55%|█████▍    | 219855/400000 [00:24<00:19, 9082.65it/s] 55%|█████▌    | 220765/400000 [00:24<00:19, 9087.50it/s] 55%|█████▌    | 221680/400000 [00:24<00:19, 9103.27it/s] 56%|█████▌    | 222601/400000 [00:24<00:19, 9133.28it/s] 56%|█████▌    | 223515/400000 [00:25<00:19, 9061.44it/s] 56%|█████▌    | 224422/400000 [00:25<00:19, 9001.08it/s] 56%|█████▋    | 225332/400000 [00:25<00:19, 9029.36it/s] 57%|█████▋    | 226248/400000 [00:25<00:19, 9067.40it/s] 57%|█████▋    | 227164/400000 [00:25<00:19, 9094.12it/s] 57%|█████▋    | 228074/400000 [00:25<00:19, 9047.02it/s] 57%|█████▋    | 228979/400000 [00:25<00:18, 9042.23it/s] 57%|█████▋    | 229886/400000 [00:25<00:18, 9047.63it/s] 58%|█████▊    | 230791/400000 [00:25<00:19, 8861.17it/s] 58%|█████▊    | 231706/400000 [00:26<00:18, 8943.80it/s] 58%|█████▊    | 232616/400000 [00:26<00:18, 8987.92it/s] 58%|█████▊    | 233527/400000 [00:26<00:18, 9023.85it/s] 59%|█████▊    | 234430/400000 [00:26<00:18, 8997.02it/s] 59%|█████▉    | 235354/400000 [00:26<00:18, 9067.44it/s] 59%|█████▉    | 236265/400000 [00:26<00:18, 9078.06it/s] 59%|█████▉    | 237174/400000 [00:26<00:18, 9026.41it/s] 60%|█████▉    | 238077/400000 [00:26<00:18, 8901.30it/s] 60%|█████▉    | 238968/400000 [00:26<00:18, 8694.55it/s] 60%|█████▉    | 239874/400000 [00:26<00:18, 8800.46it/s] 60%|██████    | 240801/400000 [00:27<00:17, 8934.00it/s] 60%|██████    | 241696/400000 [00:27<00:17, 8928.60it/s] 61%|██████    | 242590/400000 [00:27<00:17, 8927.39it/s] 61%|██████    | 243484/400000 [00:27<00:17, 8929.33it/s] 61%|██████    | 244378/400000 [00:27<00:17, 8919.62it/s] 61%|██████▏   | 245271/400000 [00:27<00:17, 8896.77it/s] 62%|██████▏   | 246161/400000 [00:27<00:17, 8697.83it/s] 62%|██████▏   | 247067/400000 [00:27<00:17, 8802.08it/s] 62%|██████▏   | 247990/400000 [00:27<00:17, 8925.76it/s] 62%|██████▏   | 248884/400000 [00:27<00:16, 8909.69it/s] 62%|██████▏   | 249776/400000 [00:28<00:16, 8846.91it/s] 63%|██████▎   | 250662/400000 [00:28<00:16, 8827.09it/s] 63%|██████▎   | 251546/400000 [00:28<00:17, 8686.61it/s] 63%|██████▎   | 252463/400000 [00:28<00:16, 8824.23it/s] 63%|██████▎   | 253347/400000 [00:28<00:16, 8806.64it/s] 64%|██████▎   | 254270/400000 [00:28<00:16, 8926.64it/s] 64%|██████▍   | 255164/400000 [00:28<00:16, 8879.38it/s] 64%|██████▍   | 256053/400000 [00:28<00:16, 8796.46it/s] 64%|██████▍   | 256949/400000 [00:28<00:16, 8843.69it/s] 64%|██████▍   | 257872/400000 [00:28<00:15, 8955.91it/s] 65%|██████▍   | 258803/400000 [00:29<00:15, 9056.62it/s] 65%|██████▍   | 259727/400000 [00:29<00:15, 9108.48it/s] 65%|██████▌   | 260643/400000 [00:29<00:15, 9122.42it/s] 65%|██████▌   | 261556/400000 [00:29<00:15, 9086.52it/s] 66%|██████▌   | 262465/400000 [00:29<00:15, 8873.59it/s] 66%|██████▌   | 263377/400000 [00:29<00:15, 8944.78it/s] 66%|██████▌   | 264276/400000 [00:29<00:15, 8956.27it/s] 66%|██████▋   | 265191/400000 [00:29<00:14, 9011.13it/s] 67%|██████▋   | 266123/400000 [00:29<00:14, 9098.45it/s] 67%|██████▋   | 267053/400000 [00:29<00:14, 9155.54it/s] 67%|██████▋   | 267970/400000 [00:30<00:14, 9044.77it/s] 67%|██████▋   | 268881/400000 [00:30<00:14, 9062.79it/s] 67%|██████▋   | 269788/400000 [00:30<00:14, 9034.02it/s] 68%|██████▊   | 270707/400000 [00:30<00:14, 9079.35it/s] 68%|██████▊   | 271616/400000 [00:30<00:14, 9035.93it/s] 68%|██████▊   | 272528/400000 [00:30<00:14, 9060.35it/s] 68%|██████▊   | 273435/400000 [00:30<00:14, 9030.23it/s] 69%|██████▊   | 274339/400000 [00:30<00:13, 9004.12it/s] 69%|██████▉   | 275240/400000 [00:30<00:13, 8988.15it/s] 69%|██████▉   | 276140/400000 [00:30<00:13, 8991.38it/s] 69%|██████▉   | 277066/400000 [00:31<00:13, 9068.76it/s] 69%|██████▉   | 277974/400000 [00:31<00:13, 9006.36it/s] 70%|██████▉   | 278886/400000 [00:31<00:13, 9039.80it/s] 70%|██████▉   | 279811/400000 [00:31<00:13, 9099.29it/s] 70%|███████   | 280722/400000 [00:31<00:13, 9097.97it/s] 70%|███████   | 281648/400000 [00:31<00:12, 9144.51it/s] 71%|███████   | 282563/400000 [00:31<00:12, 9093.85it/s] 71%|███████   | 283479/400000 [00:31<00:12, 9112.49it/s] 71%|███████   | 284391/400000 [00:31<00:12, 9058.69it/s] 71%|███████▏  | 285308/400000 [00:31<00:12, 9089.41it/s] 72%|███████▏  | 286218/400000 [00:32<00:12, 9049.99it/s] 72%|███████▏  | 287135/400000 [00:32<00:12, 9083.95it/s] 72%|███████▏  | 288044/400000 [00:32<00:12, 8747.15it/s] 72%|███████▏  | 288922/400000 [00:32<00:12, 8748.54it/s] 72%|███████▏  | 289823/400000 [00:32<00:12, 8823.80it/s] 73%|███████▎  | 290749/400000 [00:32<00:12, 8947.77it/s] 73%|███████▎  | 291646/400000 [00:32<00:12, 8899.03it/s] 73%|███████▎  | 292541/400000 [00:32<00:12, 8913.59it/s] 73%|███████▎  | 293434/400000 [00:32<00:12, 8820.03it/s] 74%|███████▎  | 294332/400000 [00:32<00:11, 8865.41it/s] 74%|███████▍  | 295249/400000 [00:33<00:11, 8953.53it/s] 74%|███████▍  | 296145/400000 [00:33<00:12, 8624.19it/s] 74%|███████▍  | 297011/400000 [00:33<00:12, 8515.83it/s] 74%|███████▍  | 297885/400000 [00:33<00:11, 8579.70it/s] 75%|███████▍  | 298745/400000 [00:33<00:11, 8540.59it/s] 75%|███████▍  | 299641/400000 [00:33<00:11, 8661.74it/s] 75%|███████▌  | 300538/400000 [00:33<00:11, 8750.92it/s] 75%|███████▌  | 301415/400000 [00:33<00:11, 8743.78it/s] 76%|███████▌  | 302315/400000 [00:33<00:11, 8816.71it/s] 76%|███████▌  | 303220/400000 [00:34<00:10, 8883.21it/s] 76%|███████▌  | 304152/400000 [00:34<00:10, 9007.76it/s] 76%|███████▋  | 305054/400000 [00:34<00:10, 9000.02it/s] 76%|███████▋  | 305955/400000 [00:34<00:10, 8852.21it/s] 77%|███████▋  | 306842/400000 [00:34<00:10, 8814.13it/s] 77%|███████▋  | 307745/400000 [00:34<00:10, 8875.64it/s] 77%|███████▋  | 308645/400000 [00:34<00:10, 8909.76it/s] 77%|███████▋  | 309549/400000 [00:34<00:10, 8948.10it/s] 78%|███████▊  | 310445/400000 [00:34<00:10, 8885.82it/s] 78%|███████▊  | 311335/400000 [00:34<00:09, 8889.87it/s] 78%|███████▊  | 312225/400000 [00:35<00:09, 8792.08it/s] 78%|███████▊  | 313105/400000 [00:35<00:10, 8674.49it/s] 78%|███████▊  | 313997/400000 [00:35<00:09, 8745.79it/s] 79%|███████▊  | 314880/400000 [00:35<00:09, 8770.74it/s] 79%|███████▉  | 315773/400000 [00:35<00:09, 8815.20it/s] 79%|███████▉  | 316690/400000 [00:35<00:09, 8918.37it/s] 79%|███████▉  | 317595/400000 [00:35<00:09, 8954.59it/s] 80%|███████▉  | 318496/400000 [00:35<00:09, 8968.92it/s] 80%|███████▉  | 319394/400000 [00:35<00:09, 8862.20it/s] 80%|████████  | 320281/400000 [00:35<00:09, 8612.19it/s] 80%|████████  | 321158/400000 [00:36<00:09, 8658.09it/s] 81%|████████  | 322065/400000 [00:36<00:08, 8777.48it/s] 81%|████████  | 322975/400000 [00:36<00:08, 8871.63it/s] 81%|████████  | 323888/400000 [00:36<00:08, 8945.78it/s] 81%|████████  | 324806/400000 [00:36<00:08, 9012.23it/s] 81%|████████▏ | 325709/400000 [00:36<00:08, 8858.52it/s] 82%|████████▏ | 326597/400000 [00:36<00:08, 8850.76it/s] 82%|████████▏ | 327513/400000 [00:36<00:08, 8939.87it/s] 82%|████████▏ | 328408/400000 [00:36<00:08, 8843.42it/s] 82%|████████▏ | 329301/400000 [00:36<00:07, 8867.17it/s] 83%|████████▎ | 330216/400000 [00:37<00:07, 8949.27it/s] 83%|████████▎ | 331145/400000 [00:37<00:07, 9047.58it/s] 83%|████████▎ | 332066/400000 [00:37<00:07, 9095.47it/s] 83%|████████▎ | 332977/400000 [00:37<00:07, 9012.09it/s] 83%|████████▎ | 333889/400000 [00:37<00:07, 9042.28it/s] 84%|████████▎ | 334820/400000 [00:37<00:07, 9118.63it/s] 84%|████████▍ | 335733/400000 [00:37<00:07, 9049.64it/s] 84%|████████▍ | 336639/400000 [00:37<00:07, 9009.08it/s] 84%|████████▍ | 337541/400000 [00:37<00:07, 8900.52it/s] 85%|████████▍ | 338432/400000 [00:37<00:06, 8807.30it/s] 85%|████████▍ | 339354/400000 [00:38<00:06, 8924.82it/s] 85%|████████▌ | 340248/400000 [00:38<00:06, 8743.72it/s] 85%|████████▌ | 341149/400000 [00:38<00:06, 8821.62it/s] 86%|████████▌ | 342033/400000 [00:38<00:06, 8735.20it/s] 86%|████████▌ | 342910/400000 [00:38<00:06, 8743.06it/s] 86%|████████▌ | 343785/400000 [00:38<00:06, 8601.82it/s] 86%|████████▌ | 344690/400000 [00:38<00:06, 8729.89it/s] 86%|████████▋ | 345587/400000 [00:38<00:06, 8799.04it/s] 87%|████████▋ | 346468/400000 [00:38<00:06, 8778.71it/s] 87%|████████▋ | 347366/400000 [00:38<00:05, 8835.31it/s] 87%|████████▋ | 348266/400000 [00:39<00:05, 8882.41it/s] 87%|████████▋ | 349161/400000 [00:39<00:05, 8901.13it/s] 88%|████████▊ | 350071/400000 [00:39<00:05, 8957.15it/s] 88%|████████▊ | 350968/400000 [00:39<00:05, 8941.99it/s] 88%|████████▊ | 351884/400000 [00:39<00:05, 9003.82it/s] 88%|████████▊ | 352813/400000 [00:39<00:05, 9086.88it/s] 88%|████████▊ | 353723/400000 [00:39<00:05, 9024.54it/s] 89%|████████▊ | 354626/400000 [00:39<00:05, 8888.63it/s] 89%|████████▉ | 355521/400000 [00:39<00:04, 8904.82it/s] 89%|████████▉ | 356414/400000 [00:39<00:04, 8910.99it/s] 89%|████████▉ | 357306/400000 [00:40<00:04, 8893.21it/s] 90%|████████▉ | 358199/400000 [00:40<00:04, 8901.96it/s] 90%|████████▉ | 359114/400000 [00:40<00:04, 8973.68it/s] 90%|█████████ | 360019/400000 [00:40<00:04, 8995.69it/s] 90%|█████████ | 360919/400000 [00:40<00:04, 8966.20it/s] 90%|█████████ | 361816/400000 [00:40<00:04, 8963.99it/s] 91%|█████████ | 362713/400000 [00:40<00:04, 8958.70it/s] 91%|█████████ | 363609/400000 [00:40<00:04, 8921.61it/s] 91%|█████████ | 364502/400000 [00:40<00:03, 8910.46it/s] 91%|█████████▏| 365394/400000 [00:41<00:03, 8880.52it/s] 92%|█████████▏| 366304/400000 [00:41<00:03, 8942.70it/s] 92%|█████████▏| 367218/400000 [00:41<00:03, 8999.03it/s] 92%|█████████▏| 368119/400000 [00:41<00:03, 8995.99it/s] 92%|█████████▏| 369028/400000 [00:41<00:03, 9022.65it/s] 92%|█████████▏| 369933/400000 [00:41<00:03, 9030.12it/s] 93%|█████████▎| 370837/400000 [00:41<00:03, 9018.99it/s] 93%|█████████▎| 371760/400000 [00:41<00:03, 9080.51it/s] 93%|█████████▎| 372669/400000 [00:41<00:03, 9005.42it/s] 93%|█████████▎| 373573/400000 [00:41<00:02, 9014.93it/s] 94%|█████████▎| 374475/400000 [00:42<00:02, 8972.32it/s] 94%|█████████▍| 375377/400000 [00:42<00:02, 8985.42it/s] 94%|█████████▍| 376276/400000 [00:42<00:02, 8967.29it/s] 94%|█████████▍| 377183/400000 [00:42<00:02, 8996.19it/s] 95%|█████████▍| 378103/400000 [00:42<00:02, 9055.53it/s] 95%|█████████▍| 379009/400000 [00:42<00:02, 8951.79it/s] 95%|█████████▍| 379905/400000 [00:42<00:02, 8839.72it/s] 95%|█████████▌| 380802/400000 [00:42<00:02, 8876.20it/s] 95%|█████████▌| 381691/400000 [00:42<00:02, 8837.12it/s] 96%|█████████▌| 382591/400000 [00:42<00:01, 8884.78it/s] 96%|█████████▌| 383483/400000 [00:43<00:01, 8893.91it/s] 96%|█████████▌| 384395/400000 [00:43<00:01, 8958.39it/s] 96%|█████████▋| 385311/400000 [00:43<00:01, 9016.61it/s] 97%|█████████▋| 386239/400000 [00:43<00:01, 9091.19it/s] 97%|█████████▋| 387149/400000 [00:43<00:01, 9062.82it/s] 97%|█████████▋| 388056/400000 [00:43<00:01, 8713.17it/s] 97%|█████████▋| 388931/400000 [00:43<00:01, 8648.27it/s] 97%|█████████▋| 389859/400000 [00:43<00:01, 8827.10it/s] 98%|█████████▊| 390778/400000 [00:43<00:01, 8932.41it/s] 98%|█████████▊| 391677/400000 [00:43<00:00, 8949.59it/s] 98%|█████████▊| 392585/400000 [00:44<00:00, 8987.85it/s] 98%|█████████▊| 393487/400000 [00:44<00:00, 8996.65it/s] 99%|█████████▊| 394410/400000 [00:44<00:00, 9064.00it/s] 99%|█████████▉| 395332/400000 [00:44<00:00, 9107.55it/s] 99%|█████████▉| 396244/400000 [00:44<00:00, 8985.51it/s] 99%|█████████▉| 397144/400000 [00:44<00:00, 8918.42it/s]100%|█████████▉| 398038/400000 [00:44<00:00, 8922.69it/s]100%|█████████▉| 398934/400000 [00:44<00:00, 8933.64it/s]100%|█████████▉| 399828/400000 [00:44<00:00, 8934.51it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8915.80it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fa42cc68ef0>, <torchtext.data.dataset.TabularDataset object at 0x7fa42cc680b8>, <torchtext.vocab.Vocab object at 0x7fa42cc68f60>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.86 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.65 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.65 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.34 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.34 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.42 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.42 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.58 file/s]2020-07-23 18:19:24.538252: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-23 18:19:24.542528: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-23 18:19:24.542758: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c04afa5050 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-23 18:19:24.542778: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▉       | 2859008/9912422 [00:00<00:00, 28516950.27it/s]9920512it [00:00, 31534210.60it/s]                             
0it [00:00, ?it/s]32768it [00:00, 744281.13it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1038910.84it/s]1654784it [00:00, 12655174.45it/s]                           
0it [00:00, ?it/s]8192it [00:00, 164980.07it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
