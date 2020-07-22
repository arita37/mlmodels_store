
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f26ae3459d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f26ae3459d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f2718faf510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f2718faf510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f27381deea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f27381deea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f26c62e1950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f26c62e1950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f26c62e1950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 26%|██▋       | 2613248/9912422 [00:00<00:00, 25897321.70it/s]9920512it [00:00, 28372023.19it/s]                             
0it [00:00, ?it/s]32768it [00:00, 649479.49it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 985943.12it/s]1654784it [00:00, 11696518.26it/s]                          
0it [00:00, ?it/s]8192it [00:00, 219052.75it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26c48395f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26c4853898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f26c62e1598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f26c62e1598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f26c62e1598> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:53,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:53,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:37,  3.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
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

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:36,  3.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:12, 10.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:12, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:12, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:12, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:08, 13.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 17.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 17.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 17.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 17.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 17.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 22.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 22.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 22.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 22.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 22.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 22.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 22.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 22.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 28.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 35.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 35.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 35.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 35.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 42.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 42.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 42.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 42.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 49.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 53.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 53.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 53.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 53.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 53.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 53.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 55.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 55.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 55.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 55.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 59.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 59.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 59.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 59.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 59.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 59.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 59.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 59.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 58.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 58.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 58.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 58.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 58.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 58.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 58.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 58.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 60.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 60.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 60.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 60.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 60.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 60.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 60.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 60.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 61.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 61.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 61.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 61.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 61.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 61.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 61.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 61.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 62.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 62.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 62.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 62.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 62.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 62.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 62.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 62.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 62.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.97s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.97s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.97s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 62.35 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.11s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  2.97s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 62.35 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.11s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.11s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 26.50 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.11s/ url]
0 examples [00:00, ? examples/s]2020-07-22 00:10:37.892667: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-22 00:10:37.907064: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-07-22 00:10:37.907298: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a64733f1f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-22 00:10:37.907316: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
15 examples [00:00, 149.81 examples/s]94 examples [00:00, 197.91 examples/s]183 examples [00:00, 257.86 examples/s]273 examples [00:00, 327.77 examples/s]361 examples [00:00, 403.73 examples/s]442 examples [00:00, 475.00 examples/s]516 examples [00:00, 531.96 examples/s]598 examples [00:00, 593.67 examples/s]683 examples [00:00, 650.93 examples/s]766 examples [00:01, 694.20 examples/s]861 examples [00:01, 753.59 examples/s]945 examples [00:01, 757.48 examples/s]1029 examples [00:01, 778.84 examples/s]1114 examples [00:01, 796.59 examples/s]1202 examples [00:01, 817.30 examples/s]1286 examples [00:01, 820.83 examples/s]1370 examples [00:01, 811.61 examples/s]1453 examples [00:01, 799.64 examples/s]1538 examples [00:01, 811.92 examples/s]1620 examples [00:02, 809.25 examples/s]1704 examples [00:02, 815.43 examples/s]1786 examples [00:02, 796.65 examples/s]1876 examples [00:02, 824.10 examples/s]1964 examples [00:02, 838.00 examples/s]2049 examples [00:02, 834.10 examples/s]2140 examples [00:02, 853.83 examples/s]2227 examples [00:02, 856.62 examples/s]2313 examples [00:02, 817.41 examples/s]2401 examples [00:02, 835.03 examples/s]2487 examples [00:03, 842.24 examples/s]2577 examples [00:03, 856.45 examples/s]2669 examples [00:03, 872.61 examples/s]2757 examples [00:03, 853.69 examples/s]2843 examples [00:03, 825.12 examples/s]2933 examples [00:03, 843.12 examples/s]3018 examples [00:03, 837.79 examples/s]3105 examples [00:03, 844.85 examples/s]3192 examples [00:03, 851.29 examples/s]3281 examples [00:03, 861.90 examples/s]3368 examples [00:04, 837.84 examples/s]3454 examples [00:04, 842.95 examples/s]3540 examples [00:04, 847.22 examples/s]3628 examples [00:04, 855.91 examples/s]3715 examples [00:04, 858.20 examples/s]3802 examples [00:04, 860.01 examples/s]3889 examples [00:04, 857.52 examples/s]3982 examples [00:04, 875.84 examples/s]4070 examples [00:04, 873.11 examples/s]4158 examples [00:05, 867.13 examples/s]4245 examples [00:05, 861.39 examples/s]4332 examples [00:05, 856.55 examples/s]4418 examples [00:05, 851.69 examples/s]4504 examples [00:05, 839.88 examples/s]4592 examples [00:05, 850.09 examples/s]4678 examples [00:05, 848.58 examples/s]4767 examples [00:05, 858.12 examples/s]4856 examples [00:05, 858.83 examples/s]4942 examples [00:05, 844.11 examples/s]5027 examples [00:06, 831.11 examples/s]5111 examples [00:06, 795.45 examples/s]5201 examples [00:06, 823.01 examples/s]5289 examples [00:06, 837.75 examples/s]5374 examples [00:06, 826.94 examples/s]5464 examples [00:06, 845.62 examples/s]5549 examples [00:06, 840.38 examples/s]5634 examples [00:06, 837.71 examples/s]5723 examples [00:06, 841.98 examples/s]5812 examples [00:06, 855.20 examples/s]5899 examples [00:07, 857.94 examples/s]5985 examples [00:07, 847.84 examples/s]6071 examples [00:07, 850.38 examples/s]6159 examples [00:07, 857.29 examples/s]6250 examples [00:07, 871.02 examples/s]6338 examples [00:07, 872.14 examples/s]6426 examples [00:07, 870.33 examples/s]6514 examples [00:07, 846.27 examples/s]6604 examples [00:07, 860.88 examples/s]6694 examples [00:08, 871.95 examples/s]6786 examples [00:08, 885.28 examples/s]6875 examples [00:08, 860.68 examples/s]6963 examples [00:08, 864.41 examples/s]7053 examples [00:08, 872.54 examples/s]7145 examples [00:08, 884.90 examples/s]7237 examples [00:08, 894.90 examples/s]7328 examples [00:08, 895.73 examples/s]7418 examples [00:08, 869.19 examples/s]7507 examples [00:08, 873.41 examples/s]7595 examples [00:09, 873.60 examples/s]7684 examples [00:09, 876.16 examples/s]7772 examples [00:09, 872.88 examples/s]7860 examples [00:09, 861.07 examples/s]7950 examples [00:09, 871.26 examples/s]8038 examples [00:09, 868.14 examples/s]8125 examples [00:09, 847.97 examples/s]8210 examples [00:09, 825.53 examples/s]8295 examples [00:09, 832.52 examples/s]8383 examples [00:09, 844.41 examples/s]8472 examples [00:10, 855.14 examples/s]8561 examples [00:10, 862.33 examples/s]8651 examples [00:10, 873.29 examples/s]8741 examples [00:10, 879.45 examples/s]8830 examples [00:10, 879.52 examples/s]8921 examples [00:10, 886.97 examples/s]9012 examples [00:10, 892.16 examples/s]9102 examples [00:10, 892.61 examples/s]9194 examples [00:10, 898.43 examples/s]9285 examples [00:10, 900.53 examples/s]9376 examples [00:11, 900.08 examples/s]9467 examples [00:11, 883.74 examples/s]9556 examples [00:11, 879.58 examples/s]9645 examples [00:11, 871.85 examples/s]9733 examples [00:11, 815.85 examples/s]9817 examples [00:11, 821.45 examples/s]9902 examples [00:11, 828.59 examples/s]9986 examples [00:11, 830.40 examples/s]10070 examples [00:11, 788.04 examples/s]10156 examples [00:12, 806.41 examples/s]10242 examples [00:12, 819.29 examples/s]10325 examples [00:12, 819.26 examples/s]10411 examples [00:12, 829.16 examples/s]10495 examples [00:12, 829.61 examples/s]10584 examples [00:12, 844.53 examples/s]10669 examples [00:12, 844.55 examples/s]10754 examples [00:12, 821.76 examples/s]10837 examples [00:12, 809.66 examples/s]10924 examples [00:12, 826.47 examples/s]11015 examples [00:13, 848.80 examples/s]11104 examples [00:13, 858.88 examples/s]11191 examples [00:13, 858.08 examples/s]11277 examples [00:13, 843.90 examples/s]11362 examples [00:13, 835.23 examples/s]11446 examples [00:13, 818.06 examples/s]11534 examples [00:13, 833.91 examples/s]11623 examples [00:13, 849.04 examples/s]11711 examples [00:13, 857.08 examples/s]11802 examples [00:13, 870.24 examples/s]11890 examples [00:14, 853.70 examples/s]11976 examples [00:14, 849.15 examples/s]12062 examples [00:14, 841.27 examples/s]12147 examples [00:14, 836.53 examples/s]12232 examples [00:14, 839.92 examples/s]12322 examples [00:14, 854.84 examples/s]12412 examples [00:14, 865.28 examples/s]12499 examples [00:14, 851.76 examples/s]12585 examples [00:14, 830.06 examples/s]12669 examples [00:15, 831.64 examples/s]12756 examples [00:15, 840.12 examples/s]12842 examples [00:15, 843.36 examples/s]12930 examples [00:15, 852.76 examples/s]13016 examples [00:15, 835.17 examples/s]13100 examples [00:15, 828.61 examples/s]13188 examples [00:15, 840.98 examples/s]13276 examples [00:15, 843.31 examples/s]13368 examples [00:15, 863.09 examples/s]13456 examples [00:15, 868.02 examples/s]13546 examples [00:16, 875.81 examples/s]13637 examples [00:16, 883.68 examples/s]13726 examples [00:16, 861.76 examples/s]13813 examples [00:16, 842.37 examples/s]13898 examples [00:16, 791.15 examples/s]13982 examples [00:16, 802.95 examples/s]14067 examples [00:16, 814.31 examples/s]14152 examples [00:16, 824.07 examples/s]14239 examples [00:16, 836.89 examples/s]14323 examples [00:16, 809.25 examples/s]14409 examples [00:17, 822.22 examples/s]14498 examples [00:17, 841.29 examples/s]14583 examples [00:17, 833.90 examples/s]14673 examples [00:17, 851.13 examples/s]14763 examples [00:17, 863.32 examples/s]14855 examples [00:17, 878.78 examples/s]14946 examples [00:17, 887.05 examples/s]15038 examples [00:17, 895.54 examples/s]15128 examples [00:17, 887.52 examples/s]15217 examples [00:17, 881.08 examples/s]15306 examples [00:18, 877.81 examples/s]15394 examples [00:18, 876.86 examples/s]15482 examples [00:18, 872.23 examples/s]15570 examples [00:18, 866.41 examples/s]15657 examples [00:18, 862.28 examples/s]15746 examples [00:18, 868.44 examples/s]15833 examples [00:18, 866.30 examples/s]15920 examples [00:18, 848.23 examples/s]16005 examples [00:18, 824.92 examples/s]16088 examples [00:19, 821.93 examples/s]16176 examples [00:19, 834.73 examples/s]16260 examples [00:19, 835.63 examples/s]16349 examples [00:19, 850.00 examples/s]16435 examples [00:19, 852.36 examples/s]16521 examples [00:19, 854.22 examples/s]16611 examples [00:19, 864.46 examples/s]16698 examples [00:19, 865.33 examples/s]16789 examples [00:19, 876.33 examples/s]16877 examples [00:19, 874.51 examples/s]16969 examples [00:20, 884.09 examples/s]17059 examples [00:20, 886.82 examples/s]17148 examples [00:20, 868.59 examples/s]17238 examples [00:20, 875.63 examples/s]17326 examples [00:20, 861.30 examples/s]17413 examples [00:20, 857.62 examples/s]17499 examples [00:20, 850.82 examples/s]17585 examples [00:20, 831.20 examples/s]17671 examples [00:20, 836.51 examples/s]17758 examples [00:20, 844.83 examples/s]17845 examples [00:21, 851.93 examples/s]17931 examples [00:21, 842.64 examples/s]18016 examples [00:21, 825.49 examples/s]18101 examples [00:21, 831.39 examples/s]18185 examples [00:21, 822.00 examples/s]18269 examples [00:21, 826.37 examples/s]18355 examples [00:21, 834.30 examples/s]18439 examples [00:21, 831.50 examples/s]18523 examples [00:21, 827.43 examples/s]18606 examples [00:21, 821.64 examples/s]18695 examples [00:22, 838.85 examples/s]18780 examples [00:22, 840.16 examples/s]18868 examples [00:22, 850.15 examples/s]18954 examples [00:22, 818.80 examples/s]19045 examples [00:22, 842.57 examples/s]19136 examples [00:22, 860.83 examples/s]19223 examples [00:22, 848.36 examples/s]19309 examples [00:22, 844.19 examples/s]19397 examples [00:22, 852.12 examples/s]19484 examples [00:23, 856.62 examples/s]19573 examples [00:23, 865.14 examples/s]19661 examples [00:23, 868.59 examples/s]19752 examples [00:23, 879.75 examples/s]19841 examples [00:23, 882.68 examples/s]19930 examples [00:23, 865.09 examples/s]20017 examples [00:23, 818.32 examples/s]20109 examples [00:23, 844.94 examples/s]20197 examples [00:23, 854.17 examples/s]20286 examples [00:23, 860.62 examples/s]20373 examples [00:24, 817.24 examples/s]20456 examples [00:24, 820.20 examples/s]20539 examples [00:24, 820.81 examples/s]20626 examples [00:24, 832.84 examples/s]20711 examples [00:24, 836.56 examples/s]20800 examples [00:24, 851.21 examples/s]20890 examples [00:24, 863.91 examples/s]20980 examples [00:24, 874.17 examples/s]21069 examples [00:24, 876.48 examples/s]21157 examples [00:24, 865.69 examples/s]21244 examples [00:25, 825.34 examples/s]21331 examples [00:25, 834.45 examples/s]21420 examples [00:25, 848.29 examples/s]21508 examples [00:25, 855.34 examples/s]21599 examples [00:25, 870.87 examples/s]21687 examples [00:25, 865.82 examples/s]21774 examples [00:25, 863.47 examples/s]21861 examples [00:25, 860.48 examples/s]21948 examples [00:25, 862.87 examples/s]22035 examples [00:26, 862.97 examples/s]22124 examples [00:26, 870.02 examples/s]22213 examples [00:26, 875.49 examples/s]22301 examples [00:26, 876.75 examples/s]22391 examples [00:26, 877.24 examples/s]22479 examples [00:26, 870.46 examples/s]22567 examples [00:26, 841.47 examples/s]22652 examples [00:26, 826.91 examples/s]22737 examples [00:26, 815.67 examples/s]22819 examples [00:26, 803.69 examples/s]22905 examples [00:27, 819.43 examples/s]22991 examples [00:27, 830.75 examples/s]23077 examples [00:27, 836.48 examples/s]23164 examples [00:27, 843.52 examples/s]23249 examples [00:27, 833.37 examples/s]23334 examples [00:27, 835.91 examples/s]23420 examples [00:27, 841.69 examples/s]23505 examples [00:27, 838.34 examples/s]23589 examples [00:27, 803.76 examples/s]23670 examples [00:27, 798.98 examples/s]23752 examples [00:28, 804.03 examples/s]23833 examples [00:28, 803.52 examples/s]23914 examples [00:28, 796.33 examples/s]23999 examples [00:28, 810.59 examples/s]24084 examples [00:28, 819.90 examples/s]24171 examples [00:28, 831.54 examples/s]24258 examples [00:28, 840.59 examples/s]24346 examples [00:28, 850.13 examples/s]24437 examples [00:28, 865.01 examples/s]24524 examples [00:28, 854.68 examples/s]24610 examples [00:29, 842.93 examples/s]24695 examples [00:29, 841.28 examples/s]24780 examples [00:29, 840.40 examples/s]24865 examples [00:29, 825.75 examples/s]24952 examples [00:29, 838.36 examples/s]25043 examples [00:29, 857.11 examples/s]25133 examples [00:29, 869.10 examples/s]25221 examples [00:29, 865.01 examples/s]25308 examples [00:29, 856.69 examples/s]25396 examples [00:30, 862.22 examples/s]25483 examples [00:30, 862.63 examples/s]25576 examples [00:30, 880.57 examples/s]25665 examples [00:30, 859.84 examples/s]25754 examples [00:30, 866.99 examples/s]25841 examples [00:30, 827.87 examples/s]25925 examples [00:30, 816.52 examples/s]26009 examples [00:30, 821.32 examples/s]26092 examples [00:30, 816.73 examples/s]26177 examples [00:30, 825.71 examples/s]26260 examples [00:31, 825.90 examples/s]26343 examples [00:31, 821.68 examples/s]26429 examples [00:31, 826.87 examples/s]26517 examples [00:31, 841.96 examples/s]26608 examples [00:31, 860.53 examples/s]26699 examples [00:31, 872.31 examples/s]26787 examples [00:31, 873.98 examples/s]26876 examples [00:31, 878.31 examples/s]26968 examples [00:31, 888.58 examples/s]27057 examples [00:31, 856.97 examples/s]27148 examples [00:32, 869.73 examples/s]27236 examples [00:32, 866.93 examples/s]27323 examples [00:32, 863.17 examples/s]27411 examples [00:32, 866.71 examples/s]27498 examples [00:32, 855.75 examples/s]27589 examples [00:32, 865.00 examples/s]27677 examples [00:32, 868.47 examples/s]27764 examples [00:32, 856.91 examples/s]27852 examples [00:32, 863.19 examples/s]27941 examples [00:32, 869.00 examples/s]28028 examples [00:33, 865.61 examples/s]28116 examples [00:33, 867.48 examples/s]28204 examples [00:33, 869.42 examples/s]28291 examples [00:33, 835.06 examples/s]28375 examples [00:33, 831.61 examples/s]28464 examples [00:33, 847.88 examples/s]28553 examples [00:33, 859.83 examples/s]28644 examples [00:33, 872.41 examples/s]28732 examples [00:33, 865.09 examples/s]28822 examples [00:34, 872.57 examples/s]28910 examples [00:34, 845.66 examples/s]28995 examples [00:34, 844.83 examples/s]29080 examples [00:34, 841.17 examples/s]29165 examples [00:34, 834.65 examples/s]29249 examples [00:34, 809.57 examples/s]29331 examples [00:34, 809.49 examples/s]29413 examples [00:34, 807.53 examples/s]29494 examples [00:34, 800.43 examples/s]29575 examples [00:34, 802.29 examples/s]29657 examples [00:35, 805.03 examples/s]29743 examples [00:35, 819.17 examples/s]29828 examples [00:35, 825.94 examples/s]29917 examples [00:35, 842.87 examples/s]30002 examples [00:35, 797.12 examples/s]30088 examples [00:35, 813.98 examples/s]30175 examples [00:35, 829.18 examples/s]30265 examples [00:35, 847.51 examples/s]30354 examples [00:35, 858.65 examples/s]30441 examples [00:35, 825.75 examples/s]30525 examples [00:36, 815.36 examples/s]30610 examples [00:36, 825.09 examples/s]30696 examples [00:36, 832.88 examples/s]30780 examples [00:36, 834.44 examples/s]30864 examples [00:36, 825.99 examples/s]30947 examples [00:36, 820.50 examples/s]31030 examples [00:36, 805.30 examples/s]31111 examples [00:36, 800.87 examples/s]31205 examples [00:36, 835.77 examples/s]31290 examples [00:37, 832.40 examples/s]31375 examples [00:37, 836.46 examples/s]31461 examples [00:37, 840.96 examples/s]31548 examples [00:37, 848.32 examples/s]31633 examples [00:37, 845.76 examples/s]31722 examples [00:37, 857.42 examples/s]31809 examples [00:37, 860.04 examples/s]31896 examples [00:37, 860.65 examples/s]31983 examples [00:37, 858.75 examples/s]32070 examples [00:37, 861.26 examples/s]32160 examples [00:38, 870.66 examples/s]32249 examples [00:38, 875.82 examples/s]32337 examples [00:38, 876.12 examples/s]32426 examples [00:38, 877.78 examples/s]32515 examples [00:38, 880.03 examples/s]32606 examples [00:38, 887.01 examples/s]32698 examples [00:38, 896.57 examples/s]32788 examples [00:38, 875.44 examples/s]32878 examples [00:38, 880.66 examples/s]32967 examples [00:38, 858.06 examples/s]33054 examples [00:39, 851.74 examples/s]33140 examples [00:39, 847.52 examples/s]33225 examples [00:39, 839.16 examples/s]33312 examples [00:39, 845.48 examples/s]33399 examples [00:39, 852.20 examples/s]33485 examples [00:39, 827.25 examples/s]33569 examples [00:39, 831.02 examples/s]33655 examples [00:39, 838.16 examples/s]33739 examples [00:39, 824.83 examples/s]33822 examples [00:39, 819.26 examples/s]33911 examples [00:40, 838.66 examples/s]33996 examples [00:40, 841.34 examples/s]34089 examples [00:40, 865.80 examples/s]34176 examples [00:40, 864.94 examples/s]34267 examples [00:40, 876.18 examples/s]34358 examples [00:40, 883.41 examples/s]34447 examples [00:40, 864.39 examples/s]34534 examples [00:40, 861.17 examples/s]34624 examples [00:40, 869.95 examples/s]34712 examples [00:41, 856.20 examples/s]34799 examples [00:41, 860.08 examples/s]34886 examples [00:41, 845.21 examples/s]34975 examples [00:41, 856.00 examples/s]35061 examples [00:41, 840.08 examples/s]35151 examples [00:41, 855.18 examples/s]35241 examples [00:41, 866.92 examples/s]35328 examples [00:41, 864.66 examples/s]35415 examples [00:41, 849.29 examples/s]35502 examples [00:41, 855.12 examples/s]35588 examples [00:42, 853.42 examples/s]35674 examples [00:42, 853.14 examples/s]35760 examples [00:42, 838.94 examples/s]35844 examples [00:42, 818.11 examples/s]35927 examples [00:42, 821.29 examples/s]36019 examples [00:42, 846.33 examples/s]36105 examples [00:42, 845.68 examples/s]36190 examples [00:42, 842.91 examples/s]36275 examples [00:42, 821.20 examples/s]36361 examples [00:42, 829.32 examples/s]36449 examples [00:43, 841.51 examples/s]36536 examples [00:43, 849.15 examples/s]36624 examples [00:43, 855.32 examples/s]36712 examples [00:43, 860.29 examples/s]36799 examples [00:43, 852.46 examples/s]36885 examples [00:43, 848.90 examples/s]36970 examples [00:43, 843.39 examples/s]37059 examples [00:43, 855.78 examples/s]37149 examples [00:43, 865.86 examples/s]37240 examples [00:43, 875.46 examples/s]37332 examples [00:44, 885.74 examples/s]37421 examples [00:44, 812.62 examples/s]37508 examples [00:44, 828.77 examples/s]37593 examples [00:44, 833.58 examples/s]37683 examples [00:44, 850.43 examples/s]37770 examples [00:44, 855.29 examples/s]37858 examples [00:44, 859.02 examples/s]37949 examples [00:44, 872.32 examples/s]38037 examples [00:44, 874.15 examples/s]38129 examples [00:45, 885.45 examples/s]38218 examples [00:45, 872.30 examples/s]38306 examples [00:45, 857.88 examples/s]38392 examples [00:45, 844.24 examples/s]38478 examples [00:45, 847.44 examples/s]38568 examples [00:45, 861.15 examples/s]38659 examples [00:45, 872.84 examples/s]38751 examples [00:45, 886.30 examples/s]38840 examples [00:45, 861.78 examples/s]38927 examples [00:45, 851.55 examples/s]39018 examples [00:46, 867.59 examples/s]39105 examples [00:46, 867.64 examples/s]39192 examples [00:46, 862.68 examples/s]39279 examples [00:46, 847.44 examples/s]39373 examples [00:46, 871.17 examples/s]39461 examples [00:46, 872.25 examples/s]39549 examples [00:46, 867.89 examples/s]39638 examples [00:46, 872.97 examples/s]39726 examples [00:46, 859.25 examples/s]39814 examples [00:46, 863.08 examples/s]39902 examples [00:47, 867.12 examples/s]39992 examples [00:47, 876.63 examples/s]40080 examples [00:47, 822.66 examples/s]40169 examples [00:47, 840.84 examples/s]40265 examples [00:47, 871.96 examples/s]40356 examples [00:47, 882.99 examples/s]40445 examples [00:47, 880.90 examples/s]40536 examples [00:47, 888.34 examples/s]40626 examples [00:47, 886.72 examples/s]40715 examples [00:48, 869.92 examples/s]40805 examples [00:48, 876.64 examples/s]40893 examples [00:48, 863.61 examples/s]40980 examples [00:48, 847.19 examples/s]41065 examples [00:48, 818.28 examples/s]41151 examples [00:48, 829.77 examples/s]41235 examples [00:48, 832.75 examples/s]41321 examples [00:48, 839.66 examples/s]41406 examples [00:48, 826.22 examples/s]41494 examples [00:48, 841.54 examples/s]41586 examples [00:49, 863.07 examples/s]41675 examples [00:49, 870.72 examples/s]41763 examples [00:49, 853.75 examples/s]41849 examples [00:49, 851.51 examples/s]41935 examples [00:49, 812.28 examples/s]42017 examples [00:49, 742.71 examples/s]42101 examples [00:49, 767.31 examples/s]42189 examples [00:49, 796.95 examples/s]42274 examples [00:49, 811.28 examples/s]42357 examples [00:49, 816.39 examples/s]42441 examples [00:50, 820.86 examples/s]42524 examples [00:50, 807.22 examples/s]42608 examples [00:50, 816.02 examples/s]42694 examples [00:50, 826.00 examples/s]42777 examples [00:50, 811.97 examples/s]42866 examples [00:50, 832.24 examples/s]42951 examples [00:50, 835.35 examples/s]43037 examples [00:50, 840.23 examples/s]43126 examples [00:50, 852.77 examples/s]43215 examples [00:51, 862.36 examples/s]43302 examples [00:51, 859.14 examples/s]43389 examples [00:51, 856.17 examples/s]43477 examples [00:51, 861.75 examples/s]43569 examples [00:51, 877.50 examples/s]43657 examples [00:51, 873.78 examples/s]43745 examples [00:51, 860.81 examples/s]43833 examples [00:51, 863.95 examples/s]43923 examples [00:51, 872.44 examples/s]44011 examples [00:51, 868.14 examples/s]44099 examples [00:52, 870.09 examples/s]44187 examples [00:52, 868.11 examples/s]44274 examples [00:52, 842.04 examples/s]44359 examples [00:52, 839.94 examples/s]44444 examples [00:52, 841.41 examples/s]44529 examples [00:52, 838.42 examples/s]44615 examples [00:52, 842.26 examples/s]44706 examples [00:52, 859.75 examples/s]44795 examples [00:52, 867.79 examples/s]44882 examples [00:52, 863.43 examples/s]44969 examples [00:53, 840.35 examples/s]45054 examples [00:53, 841.99 examples/s]45139 examples [00:53, 833.29 examples/s]45225 examples [00:53, 838.43 examples/s]45312 examples [00:53, 845.33 examples/s]45397 examples [00:53, 843.24 examples/s]45486 examples [00:53, 855.49 examples/s]45572 examples [00:53, 855.02 examples/s]45658 examples [00:53, 854.09 examples/s]45747 examples [00:53, 864.03 examples/s]45836 examples [00:54, 870.39 examples/s]45924 examples [00:54, 867.01 examples/s]46012 examples [00:54, 869.15 examples/s]46099 examples [00:54, 855.54 examples/s]46187 examples [00:54, 860.85 examples/s]46277 examples [00:54, 870.43 examples/s]46365 examples [00:54, 870.73 examples/s]46453 examples [00:54, 870.33 examples/s]46541 examples [00:54, 845.07 examples/s]46626 examples [00:55, 808.00 examples/s]46714 examples [00:55, 825.98 examples/s]46800 examples [00:55, 833.78 examples/s]46889 examples [00:55, 847.95 examples/s]46977 examples [00:55, 856.03 examples/s]47065 examples [00:55, 861.44 examples/s]47152 examples [00:55, 858.01 examples/s]47238 examples [00:55, 858.12 examples/s]47329 examples [00:55, 871.61 examples/s]47418 examples [00:55, 875.65 examples/s]47509 examples [00:56, 883.04 examples/s]47598 examples [00:56, 879.34 examples/s]47686 examples [00:56, 877.97 examples/s]47775 examples [00:56, 880.88 examples/s]47866 examples [00:56, 888.15 examples/s]47961 examples [00:56, 903.82 examples/s]48052 examples [00:56, 894.96 examples/s]48143 examples [00:56, 899.22 examples/s]48234 examples [00:56, 898.64 examples/s]48325 examples [00:56, 900.62 examples/s]48416 examples [00:57, 898.20 examples/s]48506 examples [00:57, 876.89 examples/s]48594 examples [00:57, 857.53 examples/s]48680 examples [00:57, 851.26 examples/s]48766 examples [00:57, 839.49 examples/s]48851 examples [00:57, 832.23 examples/s]48935 examples [00:57, 799.75 examples/s]49028 examples [00:57, 832.67 examples/s]49112 examples [00:57, 815.97 examples/s]49198 examples [00:57, 827.88 examples/s]49285 examples [00:58, 837.97 examples/s]49370 examples [00:58, 837.60 examples/s]49456 examples [00:58, 843.54 examples/s]49544 examples [00:58, 851.57 examples/s]49632 examples [00:58, 857.10 examples/s]49721 examples [00:58, 865.92 examples/s]49808 examples [00:58, 861.86 examples/s]49896 examples [00:58, 866.67 examples/s]49983 examples [00:58, 864.26 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6584/50000 [00:00<00:00, 65835.04 examples/s] 34%|███▍      | 17040/50000 [00:00<00:00, 74064.13 examples/s] 56%|█████▌    | 27796/50000 [00:00<00:00, 81695.25 examples/s] 77%|███████▋  | 38297/50000 [00:00<00:00, 87525.02 examples/s] 99%|█████████▉| 49412/50000 [00:00<00:00, 93484.74 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 699.57 examples/s]155 examples [00:00, 738.45 examples/s]240 examples [00:00, 767.05 examples/s]328 examples [00:00, 796.52 examples/s]411 examples [00:00, 804.46 examples/s]500 examples [00:00, 827.98 examples/s]593 examples [00:00, 855.83 examples/s]674 examples [00:00, 827.89 examples/s]760 examples [00:00, 834.65 examples/s]843 examples [00:01, 831.45 examples/s]935 examples [00:01, 855.67 examples/s]1024 examples [00:01, 864.62 examples/s]1115 examples [00:01, 875.37 examples/s]1203 examples [00:01, 866.75 examples/s]1290 examples [00:01, 838.94 examples/s]1374 examples [00:01, 827.17 examples/s]1461 examples [00:01, 837.38 examples/s]1546 examples [00:01, 840.58 examples/s]1634 examples [00:01, 850.44 examples/s]1721 examples [00:02, 854.44 examples/s]1809 examples [00:02, 861.74 examples/s]1899 examples [00:02, 871.24 examples/s]1990 examples [00:02, 881.02 examples/s]2080 examples [00:02, 885.76 examples/s]2169 examples [00:02, 864.37 examples/s]2260 examples [00:02, 875.13 examples/s]2350 examples [00:02, 881.08 examples/s]2444 examples [00:02, 896.66 examples/s]2537 examples [00:02, 903.86 examples/s]2628 examples [00:03, 901.16 examples/s]2719 examples [00:03, 891.40 examples/s]2812 examples [00:03, 900.04 examples/s]2904 examples [00:03, 904.96 examples/s]2996 examples [00:03, 906.73 examples/s]3088 examples [00:03, 909.40 examples/s]3179 examples [00:03, 852.81 examples/s]3266 examples [00:03, 851.24 examples/s]3352 examples [00:03, 853.41 examples/s]3438 examples [00:03, 847.64 examples/s]3526 examples [00:04, 855.57 examples/s]3614 examples [00:04, 860.82 examples/s]3701 examples [00:04, 844.53 examples/s]3795 examples [00:04, 869.32 examples/s]3883 examples [00:04, 871.96 examples/s]3974 examples [00:04, 881.97 examples/s]4063 examples [00:04, 876.09 examples/s]4151 examples [00:04, 853.85 examples/s]4237 examples [00:04, 847.61 examples/s]4325 examples [00:05, 854.39 examples/s]4411 examples [00:05, 854.14 examples/s]4497 examples [00:05, 831.70 examples/s]4581 examples [00:05, 834.07 examples/s]4665 examples [00:05, 821.65 examples/s]4748 examples [00:05, 815.19 examples/s]4833 examples [00:05, 824.23 examples/s]4916 examples [00:05, 801.13 examples/s]4997 examples [00:05, 791.76 examples/s]5088 examples [00:05, 822.54 examples/s]5173 examples [00:06, 829.48 examples/s]5264 examples [00:06, 851.36 examples/s]5353 examples [00:06, 860.45 examples/s]5440 examples [00:06, 862.29 examples/s]5527 examples [00:06, 863.72 examples/s]5616 examples [00:06, 868.87 examples/s]5705 examples [00:06, 874.05 examples/s]5795 examples [00:06, 881.10 examples/s]5884 examples [00:06, 882.08 examples/s]5973 examples [00:06, 882.18 examples/s]6062 examples [00:07, 880.32 examples/s]6151 examples [00:07, 872.27 examples/s]6241 examples [00:07, 878.32 examples/s]6329 examples [00:07, 870.33 examples/s]6417 examples [00:07, 868.86 examples/s]6504 examples [00:07, 856.00 examples/s]6594 examples [00:07, 867.72 examples/s]6681 examples [00:07, 852.06 examples/s]6773 examples [00:07, 869.75 examples/s]6861 examples [00:07, 872.64 examples/s]6954 examples [00:08, 888.49 examples/s]7044 examples [00:08, 886.42 examples/s]7133 examples [00:08, 881.77 examples/s]7222 examples [00:08, 859.96 examples/s]7309 examples [00:08, 793.23 examples/s]7396 examples [00:08, 814.78 examples/s]7480 examples [00:08, 821.00 examples/s]7576 examples [00:08, 856.66 examples/s]7663 examples [00:08, 847.15 examples/s]7749 examples [00:09, 828.13 examples/s]7833 examples [00:09, 796.01 examples/s]7915 examples [00:09, 802.56 examples/s]8002 examples [00:09, 819.38 examples/s]8091 examples [00:09, 837.50 examples/s]8176 examples [00:09, 840.62 examples/s]8265 examples [00:09, 854.63 examples/s]8351 examples [00:09, 849.87 examples/s]8437 examples [00:09, 848.87 examples/s]8523 examples [00:09, 851.81 examples/s]8609 examples [00:10, 851.92 examples/s]8696 examples [00:10, 857.19 examples/s]8784 examples [00:10, 861.59 examples/s]8871 examples [00:10, 853.63 examples/s]8961 examples [00:10, 865.20 examples/s]9056 examples [00:10, 887.46 examples/s]9145 examples [00:10, 879.48 examples/s]9236 examples [00:10, 886.96 examples/s]9325 examples [00:10, 843.57 examples/s]9410 examples [00:11, 806.69 examples/s]9495 examples [00:11, 819.01 examples/s]9578 examples [00:11, 793.78 examples/s]9658 examples [00:11, 781.57 examples/s]9740 examples [00:11, 791.91 examples/s]9831 examples [00:11, 822.77 examples/s]9921 examples [00:11, 844.04 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 94%|█████████▍| 9382/10000 [00:00<00:00, 93816.29 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFZEDI3/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFZEDI3/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f26c62e1950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f26c62e1950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f26c62e1950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26a5697c18>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2684659d30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f26c62e1950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f26c62e1950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f26c62e1950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26846592e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26c48534a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f263ec96ea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f263ec96ea0> 

  function with postional parmater data_info <function split_train_valid at 0x7f263ec96ea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f264e861048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f264e861048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f264e861048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=b7b2120fd3182d3fe3e49e630d2e4cb1d4a34fc2b2663edea2666f4d774e8b91
  Stored in directory: /tmp/pip-ephem-wheel-cache-lmaael_a/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:06:25, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:18:23, 16.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:04:07, 23.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:03:24, 33.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.63M/862M [00:01<4:55:37, 48.4kB/s].vector_cache/glove.6B.zip:   1%|          | 8.04M/862M [00:01<3:25:58, 69.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:23:32, 98.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.1M/862M [00:01<1:39:53, 141kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:09:48, 201kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.9M/862M [00:01<48:36, 286kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<34:03, 407kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.5M/862M [00:01<23:45, 580kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:02<16:42, 822kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.7M/862M [00:02<11:41, 1.17MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:02<08:16, 1.64MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<06:01, 2.24MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<06:06, 2.20MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<07:54, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:20, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:04<04:39, 2.87MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<08:49, 1.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<07:45, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<05:48, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<04:48, 2.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:08<9:23:03, 23.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<6:34:24, 33.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:08<4:35:38, 48.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:10<3:17:13, 67.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<2:20:42, 94.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<1:38:57, 134kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:10<1:09:16, 190kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:12<54:31, 242kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<39:30, 333kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:12<27:56, 470kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:14<22:35, 580kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<18:27, 710kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<13:34, 964kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:16<11:34, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<09:25, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:16<06:54, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:18<07:52, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<08:07, 1.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<06:15, 2.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:18<04:31, 2.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:20<11:55, 1.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<09:40, 1.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<07:05, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:22<07:57, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<08:10, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<06:21, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:24<06:30, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<07:08, 1.79MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<05:33, 2.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<04:02, 3.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<11:33, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:10, 1.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:43, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<07:41, 1.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<07:56, 1.59MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:11, 2.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:21, 1.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:59, 1.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:26, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<03:57, 3.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<09:26, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<07:51, 1.59MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:49, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:57, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<07:22, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:42, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<04:08, 2.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<10:40, 1.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<08:45, 1.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:23, 1.93MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<07:20, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:43, 1.59MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:55, 2.07MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:19, 2.83MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<08:20, 1.46MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<07:05, 1.72MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:39<05:16, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:30, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:00, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<05:25, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<03:58, 3.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<07:18, 1.65MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:21, 1.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<04:44, 2.54MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<06:08, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<06:43, 1.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:13, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<03:48, 3.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<09:47, 1.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<07:52, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<05:44, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<04:12, 2.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<53:31, 222kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<39:51, 298kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<28:28, 416kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<20:23, 579kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<8:19:18, 23.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<5:49:54, 33.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<4:04:12, 48.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<2:57:53, 66.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<2:07:19, 92.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<1:29:36, 131kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<1:02:43, 186kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<48:05, 243kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<34:52, 334kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<24:37, 473kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<19:48, 585kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<16:12, 715kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<11:55, 972kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<10:10, 1.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<08:07, 1.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:56, 1.94MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<04:17, 2.67MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<34:32, 332kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<26:30, 432kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<19:07, 598kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<13:27, 846kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<53:21, 213kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<38:29, 296kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<27:10, 418kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<21:36, 524kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<17:25, 650kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<12:45, 887kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<10:42, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<08:39, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:17, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<07:01, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<07:11, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:30, 2.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<04:00, 2.78MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<07:45, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:22, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:44, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:55, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:24, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:02, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:17, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:50, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<03:36, 3.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:05, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:47, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:32, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:18, 3.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<09:34, 1.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<07:48, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:43, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:31, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:45, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:13, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:47, 2.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<07:28, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<06:20, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:42, 2.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:45, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:12, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:48, 2.21MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:28, 3.04MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<12:01, 879kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<09:29, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<06:51, 1.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<07:15, 1.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<07:13, 1.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:31, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<03:57, 2.63MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<13:46, 757kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<10:42, 974kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<07:42, 1.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<07:48, 1.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<07:33, 1.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:48, 1.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:43, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:02, 2.04MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:47, 2.71MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:17, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<03:54, 2.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:22, 3.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<7:00:35, 24.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<4:54:39, 34.4kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<3:25:27, 49.2kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<2:31:56, 66.4kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:48:21, 93.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<1:16:15, 132kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<53:13, 188kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<10:18:10, 16.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<7:13:29, 23.1kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<5:02:54, 33.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<3:33:36, 46.6kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<2:31:28, 65.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<1:46:23, 93.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<1:15:41, 131kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<53:57, 183kB/s]  .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<37:55, 260kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<28:45, 341kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<22:06, 444kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<15:52, 618kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<11:10, 873kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<15:12, 641kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<11:38, 836kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<08:20, 1.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<08:06, 1.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<07:37, 1.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:49, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:37, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:55, 1.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:40, 2.60MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:47, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<05:17, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:06, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<02:59, 3.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<06:58, 1.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:49, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:18, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:13, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:34, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:26, 2.73MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:36, 2.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:00, 2.32MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:02, 3.06MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:18, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:52, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:49, 2.42MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:46, 3.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<08:16, 1.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<06:43, 1.37MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:55, 1.86MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:34, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:44, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:28, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:44, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:31, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:05, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:05, 2.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:16, 2.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:48, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<03:45, 2.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:43, 3.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<09:12, 969kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<07:20, 1.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<05:19, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:48, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:51, 1.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<04:27, 1.98MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<03:14, 2.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:23, 1.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:22, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<03:58, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:49, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:08, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:57, 2.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<02:51, 3.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<05:07, 1.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<6:00:44, 24.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<4:12:43, 34.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<2:56:08, 48.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<2:08:19, 66.9kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<1:31:32, 93.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<1:04:24, 133kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<46:12, 184kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<33:11, 256kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<23:23, 363kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<18:16, 463kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<14:29, 583kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<10:28, 805kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<07:28, 1.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:51, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:21, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:39, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:09, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:20, 1.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:14, 2.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:10, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:40, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:38, 2.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:38, 3.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<06:40, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:32, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:02, 2.01MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:41, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:59, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:54, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<02:48, 2.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<40:37, 198kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<29:15, 275kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<20:36, 389kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<16:11, 493kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<12:07, 657kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<08:40, 916kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<07:53, 1.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<07:07, 1.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:20, 1.48MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<03:48, 2.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<09:34, 818kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<07:30, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:26, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:36, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:33, 1.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:14, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<03:03, 2.52MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:52, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<04:53, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:35, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:15, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:38, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:44, 2.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:40, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:22, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:32, 2.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:30, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:14, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:25, 3.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:24, 2.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:09, 2.35MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:23, 3.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:21, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:54, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:07, 2.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:58<02:15, 3.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<19:47, 368kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<14:36, 499kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<10:22, 699kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<08:53, 812kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<07:44, 931kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<05:44, 1.26MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<04:06, 1.75MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:38, 1.97MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<5:01:24, 23.7kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<3:30:58, 33.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<2:27:00, 48.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<1:45:50, 67.0kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<1:15:33, 93.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<53:06, 133kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<37:03, 190kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<30:15, 232kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<21:54, 320kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<15:26, 453kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<12:20, 563kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<10:05, 688kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:25, 935kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<05:14, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<25:33, 269kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<18:35, 370kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<13:07, 522kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<10:41, 637kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<08:11, 832kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:52, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:39, 1.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:19, 1.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:03, 1.65MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<02:54, 2.29MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<49:49, 134kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<35:31, 188kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<24:56, 266kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<18:54, 349kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<14:37, 452kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<10:30, 627kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<07:23, 886kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<09:02, 724kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<06:59, 934kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:02, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:50<28:23, 228kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:50<26:09, 247kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:50<19:33, 331kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:50<14:30, 445kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:50<10:18, 625kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:50<07:16, 881kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:53<13:13, 484kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:53<15:29, 413kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:53<12:19, 519kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:53<08:55, 715kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:53<06:19, 1.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:55<06:23, 990kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:55<05:10, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:55<03:46, 1.67MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:57<03:59, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:57<04:07, 1.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:57<03:10, 1.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:57<02:20, 2.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:59<03:21, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:59<02:59, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:59<02:13, 2.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:01<02:58, 2.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:01<02:36, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [04:01<01:57, 3.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [04:01<01:26, 4.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [04:03<14:07, 429kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [04:03<10:31, 575kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [04:03<07:28, 806kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [04:05<06:34, 910kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [04:05<05:14, 1.14MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [04:05<03:47, 1.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:07<03:59, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:07<03:25, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [04:07<02:30, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [04:09<03:06, 1.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [04:09<02:46, 2.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [04:09<02:04, 2.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [04:10<02:47, 2.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [04:11<03:10, 1.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [04:11<02:28, 2.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [04:11<01:47, 3.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [04:12<04:36, 1.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [04:13<03:49, 1.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [04:13<02:49, 2.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [04:14<03:15, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [04:15<03:28, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [04:15<02:40, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [04:15<01:57, 2.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:16<03:12, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:16<02:44, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [04:17<02:03, 2.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:18<02:42, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:18<03:00, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:19<02:20, 2.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:19<01:42, 3.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:20<03:27, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:20<02:53, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:21<02:07, 2.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:21<01:33, 3.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:22<15:12, 353kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:22<11:46, 456kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:23<08:30, 629kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:23<05:58, 888kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:24<20:07, 263kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:24<14:37, 362kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:24<10:19, 510kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:31<43:38, 121kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:31<30:09, 172kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:31<23:23, 222kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:31<16:52, 307kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:32<11:54, 433kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:32<08:18, 615kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:32<06:58, 732kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:32<04:53, 1.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:34<06:06, 823kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:34<05:20, 941kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:35<03:57, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:35<02:50, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:36<03:40, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:36<03:04, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:37<02:16, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:38<02:42, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:38<02:53, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:38<02:16, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:39<01:37, 2.98MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:40<13:56, 346kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:40<10:14, 470kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:40<07:15, 660kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:42<06:08, 775kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:42<05:18, 895kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:42<03:54, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:42<02:47, 1.68MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:44<03:39, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:44<02:58, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:44<02:11, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:46<02:35, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:46<02:17, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:46<01:42, 2.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [05:12<17:08, 265kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [05:12<16:26, 277kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [05:12<12:32, 362kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [05:12<09:01, 502kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [05:12<06:19, 710kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [05:13<07:55, 565kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [05:14<05:46, 774kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [05:14<04:03, 1.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [05:15<08:30, 519kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [05:16<06:50, 644kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [05:16<04:57, 885kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [05:16<03:34, 1.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [05:17<03:33, 1.22MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [05:17<02:56, 1.47MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [05:18<02:09, 2.00MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:19<02:29, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:19<02:38, 1.61MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [05:20<02:02, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [05:20<01:28, 2.86MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:21<02:40, 1.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:21<02:19, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [05:22<01:42, 2.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [05:23<02:07, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [05:23<01:50, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [05:23<01:25, 2.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [05:24<01:02, 3.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [05:25<08:23, 485kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [05:25<06:41, 607kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [05:25<04:52, 830kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [05:26<03:25, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [05:27<05:33, 720kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [05:27<04:17, 929kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [05:27<03:05, 1.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [05:29<03:03, 1.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [05:29<02:32, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [05:29<01:51, 2.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [05:31<02:09, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [05:31<01:55, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [05:31<01:25, 2.69MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [05:33<01:51, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [05:33<01:41, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [05:33<01:16, 2.96MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:35<01:45, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:35<01:37, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [05:35<01:13, 3.01MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:37<01:41, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:37<01:33, 2.33MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [05:37<01:10, 3.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [05:39<01:38, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [05:39<01:31, 2.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [05:39<01:08, 3.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [05:41<01:36, 2.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [05:41<01:28, 2.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [05:41<01:06, 3.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [05:43<01:34, 2.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:43<01:23, 2.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:43<01:03, 3.26MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:43<00:46, 4.38MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:45<08:43, 387kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [05:45<06:48, 496kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:45<04:54, 686kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:45<03:26, 967kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:47<03:57, 835kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:47<03:07, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:47<02:14, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:49<02:16, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:49<02:14, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:49<01:43, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:49<01:13, 2.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:51<1:02:03, 51.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:51<43:41, 72.5kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:51<30:24, 103kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:53<21:45, 143kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:53<15:51, 196kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:53<11:12, 276kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:53<07:46, 392kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:55<07:41, 395kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:55<05:41, 533kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:55<04:00, 748kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:57<03:28, 855kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:57<03:03, 971kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:57<02:17, 1.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:57<01:36, 1.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:59<08:07, 357kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:59<05:55, 489kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:59<04:11, 686kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:59<02:56, 966kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [06:01<12:36, 225kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [06:01<09:24, 301kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [06:01<06:41, 421kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [06:01<04:41, 595kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [06:03<04:07, 670kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [06:03<03:09, 871kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [06:03<02:15, 1.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 701M/862M [06:04<02:11, 1.23MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [06:05<02:04, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [06:05<01:35, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [06:05<01:07, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [06:06<19:33, 134kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [06:07<13:55, 188kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [06:07<09:43, 267kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [06:08<07:18, 350kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [06:09<05:38, 452kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [06:09<04:04, 625kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [06:09<02:49, 882kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [06:10<09:32, 261kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [06:11<06:55, 358kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [06:11<04:51, 505kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [06:12<03:54, 619kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [06:12<02:58, 810kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [06:13<02:07, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [06:14<02:00, 1.17MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [06:14<01:54, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [06:15<01:26, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [06:15<01:01, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [06:16<07:35, 301kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [06:16<05:32, 411kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [06:17<03:52, 580kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [06:18<03:10, 696kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [06:18<02:41, 820kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [06:18<01:59, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [06:19<01:23, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [06:20<06:07, 350kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [06:20<04:29, 476kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [06:20<03:09, 668kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [06:22<02:39, 782kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [06:22<02:16, 911kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [06:22<01:40, 1.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [06:22<01:10, 1.71MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [06:29<04:34, 439kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [06:29<04:49, 416kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [06:29<03:51, 519kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [06:29<02:48, 710kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [06:30<01:56, 1.00MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [06:31<03:04, 631kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [06:31<02:21, 818kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [06:31<01:40, 1.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [06:33<01:33, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [06:33<01:28, 1.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [06:33<01:06, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [06:33<00:47, 2.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [06:35<01:23, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [06:35<01:09, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [06:35<00:50, 2.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [06:37<00:58, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [06:37<00:51, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [06:37<00:38, 2.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [06:39<00:49, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [06:39<00:55, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [06:39<00:42, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [06:39<00:30, 3.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [06:41<01:01, 1.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [06:41<00:52, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [06:41<00:38, 2.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [06:43<00:47, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [06:43<00:52, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [06:43<00:41, 2.19MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [06:43<00:29, 3.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [06:45<03:48, 383kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [06:45<02:48, 517kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:45<01:58, 725kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:47<01:39, 836kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:47<01:18, 1.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:47<00:55, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:49<00:56, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:49<00:55, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:49<00:41, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:49<00:29, 2.58MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:51<00:56, 1.33MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:51<00:47, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:51<00:34, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:53<00:39, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:53<00:34, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:53<00:25, 2.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:55<00:33, 2.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:55<00:37, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:55<00:28, 2.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:55<00:20, 3.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:56<01:01, 1.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:57<00:48, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:57<00:34, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:58<00:37, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:59<00:38, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:59<00:29, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:59<00:20, 2.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [07:00<00:45, 1.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [07:01<00:37, 1.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [07:01<00:26, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [07:02<00:29, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [07:03<00:31, 1.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [07:03<00:23, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [07:03<00:16, 2.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [07:04<00:35, 1.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [07:04<00:29, 1.55MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [07:05<00:21, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [07:06<00:23, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [07:06<00:25, 1.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [07:07<00:19, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [07:07<00:13, 2.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [07:08<04:42, 135kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [07:08<03:20, 189kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [07:09<02:15, 269kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [07:10<01:36, 353kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [07:10<01:14, 456kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [07:10<00:52, 632kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [07:11<00:34, 893kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [07:55<08:27, 59.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [07:56<06:24, 78.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [07:56<04:34, 109kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [07:56<03:09, 154kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [07:56<02:00, 219kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [07:57<01:44, 247kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:58<01:14, 340kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [07:58<00:49, 481kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [07:59<00:36, 593kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [08:00<00:27, 779kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [08:00<00:18, 1.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [08:01<00:15, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [08:01<00:12, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [08:02<00:08, 1.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [08:02<00:05, 2.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [08:03<00:29, 466kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [08:03<00:22, 583kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [08:04<00:15, 798kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [08:04<00:08, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [08:05<01:13, 128kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [08:05<00:50, 180kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [08:06<00:29, 255kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [08:07<00:15, 336kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [08:07<00:11, 436kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [08:08<00:07, 605kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [08:08<00:02, 854kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [08:09<00:01, 787kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [08:09<00:00, 1.00MB/s].vector_cache/glove.6B.zip: 862MB [08:09, 1.76MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<80:48:25,  1.38it/s]  0%|          | 624/400000 [00:00<56:28:55,  1.96it/s]  0%|          | 1213/400000 [00:00<39:29:05,  2.81it/s]  0%|          | 1831/400000 [00:01<27:36:07,  4.01it/s]  1%|          | 2505/400000 [00:01<19:17:36,  5.72it/s]  1%|          | 3218/400000 [00:01<13:29:09,  8.17it/s]  1%|          | 3887/400000 [00:01<9:25:45, 11.67it/s]   1%|          | 4507/400000 [00:01<6:35:43, 16.66it/s]  1%|▏         | 5136/400000 [00:01<4:36:52, 23.77it/s]  1%|▏         | 5860/400000 [00:01<3:13:43, 33.91it/s]  2%|▏         | 6546/400000 [00:01<2:15:39, 48.34it/s]  2%|▏         | 7222/400000 [00:01<1:35:05, 68.84it/s]  2%|▏         | 7898/400000 [00:01<1:06:44, 97.92it/s]  2%|▏         | 8579/400000 [00:02<46:55, 139.02it/s]   2%|▏         | 9282/400000 [00:02<33:03, 196.94it/s]  2%|▎         | 10000/400000 [00:02<23:22, 278.07it/s]  3%|▎         | 10707/400000 [00:02<16:36, 390.65it/s]  3%|▎         | 11403/400000 [00:02<11:53, 544.88it/s]  3%|▎         | 12113/400000 [00:02<08:34, 753.61it/s]  3%|▎         | 12813/400000 [00:02<06:16, 1027.86it/s]  3%|▎         | 13522/400000 [00:02<04:39, 1382.36it/s]  4%|▎         | 14224/400000 [00:02<03:31, 1821.07it/s]  4%|▎         | 14924/400000 [00:02<02:45, 2333.47it/s]  4%|▍         | 15618/400000 [00:03<02:13, 2879.81it/s]  4%|▍         | 16294/400000 [00:03<01:51, 3452.12it/s]  4%|▍         | 17014/400000 [00:03<01:33, 4090.19it/s]  4%|▍         | 17709/400000 [00:03<01:21, 4665.97it/s]  5%|▍         | 18418/400000 [00:03<01:13, 5198.94it/s]  5%|▍         | 19123/400000 [00:03<01:07, 5642.02it/s]  5%|▍         | 19852/400000 [00:03<01:02, 6050.67it/s]  5%|▌         | 20558/400000 [00:03<01:01, 6212.95it/s]  5%|▌         | 21258/400000 [00:03<00:58, 6428.82it/s]  5%|▌         | 21975/400000 [00:03<00:56, 6633.00it/s]  6%|▌         | 22677/400000 [00:04<00:56, 6718.81it/s]  6%|▌         | 23376/400000 [00:04<00:55, 6751.67it/s]  6%|▌         | 24077/400000 [00:04<00:55, 6824.82it/s]  6%|▌         | 24773/400000 [00:04<00:54, 6839.91it/s]  6%|▋         | 25489/400000 [00:04<00:54, 6932.29it/s]  7%|▋         | 26217/400000 [00:04<00:53, 7032.44it/s]  7%|▋         | 26947/400000 [00:04<00:52, 7110.53it/s]  7%|▋         | 27662/400000 [00:04<00:52, 7112.98it/s]  7%|▋         | 28382/400000 [00:04<00:52, 7136.92it/s]  7%|▋         | 29098/400000 [00:04<00:52, 7126.52it/s]  7%|▋         | 29812/400000 [00:05<00:52, 7055.57it/s]  8%|▊         | 30519/400000 [00:05<00:52, 7053.48it/s]  8%|▊         | 31232/400000 [00:05<00:52, 7075.43it/s]  8%|▊         | 31941/400000 [00:05<00:52, 7074.24it/s]  8%|▊         | 32649/400000 [00:05<00:53, 6930.24it/s]  8%|▊         | 33343/400000 [00:05<00:54, 6773.80it/s]  9%|▊         | 34022/400000 [00:05<00:55, 6557.78it/s]  9%|▊         | 34681/400000 [00:05<00:55, 6559.36it/s]  9%|▉         | 35339/400000 [00:05<00:55, 6563.96it/s]  9%|▉         | 36036/400000 [00:05<00:54, 6678.56it/s]  9%|▉         | 36725/400000 [00:06<00:53, 6740.06it/s]  9%|▉         | 37466/400000 [00:06<00:52, 6926.79it/s] 10%|▉         | 38161/400000 [00:06<00:52, 6905.00it/s] 10%|▉         | 38853/400000 [00:06<00:53, 6791.75it/s] 10%|▉         | 39534/400000 [00:06<00:53, 6763.04it/s] 10%|█         | 40221/400000 [00:06<00:52, 6793.71it/s] 10%|█         | 40925/400000 [00:06<00:52, 6863.67it/s] 10%|█         | 41646/400000 [00:06<00:51, 6962.82it/s] 11%|█         | 42360/400000 [00:06<00:50, 7014.61it/s] 11%|█         | 43063/400000 [00:07<00:50, 7001.06it/s] 11%|█         | 43764/400000 [00:07<00:51, 6920.49it/s] 11%|█         | 44457/400000 [00:07<00:53, 6702.89it/s] 11%|█▏        | 45131/400000 [00:07<00:52, 6711.83it/s] 11%|█▏        | 45856/400000 [00:07<00:51, 6864.38it/s] 12%|█▏        | 46561/400000 [00:07<00:51, 6917.59it/s] 12%|█▏        | 47259/400000 [00:07<00:50, 6935.75it/s] 12%|█▏        | 47967/400000 [00:07<00:50, 6978.21it/s] 12%|█▏        | 48669/400000 [00:07<00:50, 6990.49it/s] 12%|█▏        | 49379/400000 [00:07<00:49, 7022.31it/s] 13%|█▎        | 50082/400000 [00:08<00:50, 6924.66it/s] 13%|█▎        | 50789/400000 [00:08<00:50, 6966.47it/s] 13%|█▎        | 51491/400000 [00:08<00:49, 6980.20it/s] 13%|█▎        | 52226/400000 [00:08<00:49, 7085.96it/s] 13%|█▎        | 52936/400000 [00:08<00:50, 6892.26it/s] 13%|█▎        | 53665/400000 [00:08<00:49, 7006.26it/s] 14%|█▎        | 54368/400000 [00:08<00:50, 6878.55it/s] 14%|█▍        | 55072/400000 [00:08<00:49, 6923.70it/s] 14%|█▍        | 55766/400000 [00:08<00:50, 6827.48it/s] 14%|█▍        | 56472/400000 [00:08<00:49, 6893.28it/s] 14%|█▍        | 57175/400000 [00:09<00:49, 6932.24it/s] 14%|█▍        | 57869/400000 [00:09<00:49, 6891.25it/s] 15%|█▍        | 58593/400000 [00:09<00:48, 6991.79it/s] 15%|█▍        | 59293/400000 [00:09<00:48, 6971.96it/s] 15%|█▍        | 59991/400000 [00:09<00:49, 6876.35it/s] 15%|█▌        | 60680/400000 [00:09<00:50, 6725.71it/s] 15%|█▌        | 61381/400000 [00:09<00:49, 6807.64it/s] 16%|█▌        | 62098/400000 [00:09<00:48, 6910.73it/s] 16%|█▌        | 62822/400000 [00:09<00:48, 7002.56it/s] 16%|█▌        | 63524/400000 [00:09<00:48, 7001.30it/s] 16%|█▌        | 64225/400000 [00:10<00:48, 6979.84it/s] 16%|█▌        | 64934/400000 [00:10<00:47, 7010.84it/s] 16%|█▋        | 65636/400000 [00:10<00:47, 6994.84it/s] 17%|█▋        | 66336/400000 [00:10<00:49, 6770.80it/s] 17%|█▋        | 67032/400000 [00:10<00:48, 6824.26it/s] 17%|█▋        | 67726/400000 [00:10<00:48, 6857.09it/s] 17%|█▋        | 68413/400000 [00:10<00:48, 6836.81it/s] 17%|█▋        | 69143/400000 [00:10<00:47, 6964.25it/s] 17%|█▋        | 69875/400000 [00:10<00:46, 7066.45it/s] 18%|█▊        | 70583/400000 [00:10<00:47, 6891.07it/s] 18%|█▊        | 71276/400000 [00:11<00:47, 6901.51it/s] 18%|█▊        | 71968/400000 [00:11<00:48, 6782.64it/s] 18%|█▊        | 72681/400000 [00:11<00:47, 6881.54it/s] 18%|█▊        | 73371/400000 [00:11<00:47, 6884.43it/s] 19%|█▊        | 74067/400000 [00:11<00:47, 6906.66it/s] 19%|█▊        | 74764/400000 [00:11<00:46, 6924.70it/s] 19%|█▉        | 75494/400000 [00:11<00:46, 7031.25it/s] 19%|█▉        | 76198/400000 [00:11<00:46, 6972.60it/s] 19%|█▉        | 76896/400000 [00:11<00:46, 6958.33it/s] 19%|█▉        | 77593/400000 [00:11<00:46, 6933.76it/s] 20%|█▉        | 78305/400000 [00:12<00:46, 6988.02it/s] 20%|█▉        | 79005/400000 [00:12<00:46, 6869.48it/s] 20%|█▉        | 79725/400000 [00:12<00:45, 6964.33it/s] 20%|██        | 80434/400000 [00:12<00:45, 7001.24it/s] 20%|██        | 81154/400000 [00:12<00:45, 7057.39it/s] 20%|██        | 81889/400000 [00:12<00:44, 7140.55it/s] 21%|██        | 82604/400000 [00:12<00:46, 6856.45it/s] 21%|██        | 83308/400000 [00:12<00:45, 6908.41it/s] 21%|██        | 84001/400000 [00:12<00:46, 6798.36it/s] 21%|██        | 84683/400000 [00:13<00:46, 6770.71it/s] 21%|██▏       | 85362/400000 [00:13<00:46, 6741.20it/s] 22%|██▏       | 86078/400000 [00:13<00:45, 6860.28it/s] 22%|██▏       | 86766/400000 [00:13<00:46, 6773.18it/s] 22%|██▏       | 87490/400000 [00:13<00:45, 6904.43it/s] 22%|██▏       | 88182/400000 [00:13<00:45, 6855.23it/s] 22%|██▏       | 88888/400000 [00:13<00:44, 6914.67it/s] 22%|██▏       | 89593/400000 [00:13<00:44, 6952.38it/s] 23%|██▎       | 90289/400000 [00:13<00:45, 6800.07it/s] 23%|██▎       | 90971/400000 [00:13<00:45, 6783.66it/s] 23%|██▎       | 91658/400000 [00:14<00:45, 6801.18it/s] 23%|██▎       | 92364/400000 [00:14<00:44, 6874.71it/s] 23%|██▎       | 93078/400000 [00:14<00:44, 6950.74it/s] 23%|██▎       | 93804/400000 [00:14<00:43, 7039.06it/s] 24%|██▎       | 94509/400000 [00:14<00:43, 7017.60it/s] 24%|██▍       | 95253/400000 [00:14<00:42, 7136.49it/s] 24%|██▍       | 95995/400000 [00:14<00:42, 7218.77it/s] 24%|██▍       | 96718/400000 [00:14<00:42, 7086.31it/s] 24%|██▍       | 97428/400000 [00:14<00:42, 7050.38it/s] 25%|██▍       | 98134/400000 [00:14<00:43, 6959.47it/s] 25%|██▍       | 98831/400000 [00:15<00:43, 6877.25it/s] 25%|██▍       | 99560/400000 [00:15<00:42, 6992.65it/s] 25%|██▌       | 100277/400000 [00:15<00:42, 7042.64it/s] 25%|██▌       | 101013/400000 [00:15<00:41, 7132.35it/s] 25%|██▌       | 101758/400000 [00:15<00:41, 7222.21it/s] 26%|██▌       | 102482/400000 [00:15<00:41, 7176.35it/s] 26%|██▌       | 103201/400000 [00:15<00:43, 6881.96it/s] 26%|██▌       | 103893/400000 [00:15<00:43, 6809.88it/s] 26%|██▌       | 104577/400000 [00:15<00:43, 6806.43it/s] 26%|██▋       | 105287/400000 [00:15<00:42, 6891.73it/s] 27%|██▋       | 106018/400000 [00:16<00:41, 7011.29it/s] 27%|██▋       | 106745/400000 [00:16<00:41, 7085.89it/s] 27%|██▋       | 107455/400000 [00:16<00:41, 7007.16it/s] 27%|██▋       | 108168/400000 [00:16<00:41, 7041.74it/s] 27%|██▋       | 108873/400000 [00:16<00:42, 6856.99it/s] 27%|██▋       | 109561/400000 [00:16<00:43, 6657.53it/s] 28%|██▊       | 110248/400000 [00:16<00:43, 6717.80it/s] 28%|██▊       | 110945/400000 [00:16<00:42, 6790.16it/s] 28%|██▊       | 111626/400000 [00:16<00:42, 6759.50it/s] 28%|██▊       | 112303/400000 [00:17<00:42, 6724.90it/s] 28%|██▊       | 112977/400000 [00:17<00:42, 6685.91it/s] 28%|██▊       | 113674/400000 [00:17<00:42, 6768.39it/s] 29%|██▊       | 114352/400000 [00:17<00:42, 6709.25it/s] 29%|██▉       | 115077/400000 [00:17<00:41, 6862.18it/s] 29%|██▉       | 115770/400000 [00:17<00:41, 6879.05it/s] 29%|██▉       | 116473/400000 [00:17<00:40, 6921.22it/s] 29%|██▉       | 117182/400000 [00:17<00:40, 6970.63it/s] 29%|██▉       | 117903/400000 [00:17<00:40, 7039.89it/s] 30%|██▉       | 118632/400000 [00:17<00:39, 7111.31it/s] 30%|██▉       | 119344/400000 [00:18<00:39, 7105.12it/s] 30%|███       | 120055/400000 [00:18<00:39, 7089.52it/s] 30%|███       | 120771/400000 [00:18<00:39, 7109.96it/s] 30%|███       | 121483/400000 [00:18<00:39, 7097.04it/s] 31%|███       | 122193/400000 [00:18<00:39, 7042.56it/s] 31%|███       | 122911/400000 [00:18<00:39, 7081.92it/s] 31%|███       | 123654/400000 [00:18<00:38, 7181.94it/s] 31%|███       | 124373/400000 [00:18<00:38, 7163.11it/s] 31%|███▏      | 125095/400000 [00:18<00:38, 7177.39it/s] 31%|███▏      | 125826/400000 [00:18<00:37, 7215.15it/s] 32%|███▏      | 126548/400000 [00:19<00:39, 6995.94it/s] 32%|███▏      | 127270/400000 [00:19<00:38, 7059.91it/s] 32%|███▏      | 127981/400000 [00:19<00:38, 7070.54it/s] 32%|███▏      | 128689/400000 [00:19<00:39, 6908.30it/s] 32%|███▏      | 129405/400000 [00:19<00:38, 6980.71it/s] 33%|███▎      | 130139/400000 [00:19<00:38, 7084.05it/s] 33%|███▎      | 130849/400000 [00:19<00:38, 7070.65it/s] 33%|███▎      | 131557/400000 [00:19<00:38, 6916.12it/s] 33%|███▎      | 132250/400000 [00:19<00:38, 6883.08it/s] 33%|███▎      | 132972/400000 [00:19<00:38, 6978.17it/s] 33%|███▎      | 133697/400000 [00:20<00:37, 7056.04it/s] 34%|███▎      | 134404/400000 [00:20<00:38, 6947.40it/s] 34%|███▍      | 135100/400000 [00:20<00:39, 6773.47it/s] 34%|███▍      | 135811/400000 [00:20<00:38, 6869.67it/s] 34%|███▍      | 136522/400000 [00:20<00:37, 6937.77it/s] 34%|███▍      | 137258/400000 [00:20<00:37, 7058.55it/s] 35%|███▍      | 138002/400000 [00:20<00:36, 7167.56it/s] 35%|███▍      | 138721/400000 [00:20<00:36, 7152.75it/s] 35%|███▍      | 139444/400000 [00:20<00:36, 7174.60it/s] 35%|███▌      | 140163/400000 [00:20<00:36, 7160.02it/s] 35%|███▌      | 140900/400000 [00:21<00:35, 7219.41it/s] 35%|███▌      | 141623/400000 [00:21<00:35, 7213.92it/s] 36%|███▌      | 142345/400000 [00:21<00:36, 7140.06it/s] 36%|███▌      | 143060/400000 [00:21<00:37, 6769.44it/s] 36%|███▌      | 143753/400000 [00:21<00:37, 6814.78it/s] 36%|███▌      | 144438/400000 [00:21<00:37, 6760.90it/s] 36%|███▋      | 145117/400000 [00:21<00:37, 6725.12it/s] 36%|███▋      | 145792/400000 [00:21<00:38, 6613.48it/s] 37%|███▋      | 146510/400000 [00:21<00:37, 6772.05it/s] 37%|███▋      | 147225/400000 [00:22<00:36, 6879.52it/s] 37%|███▋      | 147915/400000 [00:22<00:36, 6879.41it/s] 37%|███▋      | 148641/400000 [00:22<00:35, 6987.98it/s] 37%|███▋      | 149342/400000 [00:22<00:36, 6841.89it/s] 38%|███▊      | 150054/400000 [00:22<00:36, 6922.41it/s] 38%|███▊      | 150802/400000 [00:22<00:35, 7080.61it/s] 38%|███▊      | 151531/400000 [00:22<00:34, 7141.24it/s] 38%|███▊      | 152253/400000 [00:22<00:34, 7161.79it/s] 38%|███▊      | 152971/400000 [00:22<00:35, 7031.13it/s] 38%|███▊      | 153683/400000 [00:22<00:34, 7054.60it/s] 39%|███▊      | 154390/400000 [00:23<00:35, 6999.14it/s] 39%|███▉      | 155091/400000 [00:23<00:35, 6905.63it/s] 39%|███▉      | 155783/400000 [00:23<00:35, 6857.91it/s] 39%|███▉      | 156470/400000 [00:23<00:35, 6779.06it/s] 39%|███▉      | 157185/400000 [00:23<00:35, 6885.65it/s] 39%|███▉      | 157875/400000 [00:23<00:35, 6745.92it/s] 40%|███▉      | 158551/400000 [00:23<00:35, 6728.76it/s] 40%|███▉      | 159278/400000 [00:23<00:34, 6882.37it/s] 40%|███▉      | 159996/400000 [00:23<00:34, 6962.73it/s] 40%|████      | 160700/400000 [00:23<00:34, 6983.79it/s] 40%|████      | 161439/400000 [00:24<00:33, 7100.53it/s] 41%|████      | 162178/400000 [00:24<00:33, 7182.51it/s] 41%|████      | 162937/400000 [00:24<00:32, 7298.01it/s] 41%|████      | 163668/400000 [00:24<00:33, 7058.58it/s] 41%|████      | 164377/400000 [00:24<00:34, 6831.21it/s] 41%|████▏     | 165064/400000 [00:24<00:34, 6749.83it/s] 41%|████▏     | 165742/400000 [00:24<00:34, 6702.48it/s] 42%|████▏     | 166462/400000 [00:24<00:34, 6843.49it/s] 42%|████▏     | 167206/400000 [00:24<00:33, 7010.77it/s] 42%|████▏     | 167923/400000 [00:24<00:32, 7055.26it/s] 42%|████▏     | 168652/400000 [00:25<00:32, 7122.12it/s] 42%|████▏     | 169366/400000 [00:25<00:32, 7123.36it/s] 43%|████▎     | 170101/400000 [00:25<00:31, 7189.26it/s] 43%|████▎     | 170821/400000 [00:25<00:33, 6913.44it/s] 43%|████▎     | 171516/400000 [00:25<00:33, 6832.36it/s] 43%|████▎     | 172202/400000 [00:25<00:33, 6725.03it/s] 43%|████▎     | 172877/400000 [00:25<00:33, 6698.14it/s] 43%|████▎     | 173549/400000 [00:25<00:34, 6649.13it/s] 44%|████▎     | 174253/400000 [00:25<00:33, 6759.68it/s] 44%|████▎     | 174937/400000 [00:26<00:33, 6783.35it/s] 44%|████▍     | 175663/400000 [00:26<00:32, 6918.05it/s] 44%|████▍     | 176357/400000 [00:26<00:32, 6812.04it/s] 44%|████▍     | 177074/400000 [00:26<00:32, 6915.51it/s] 44%|████▍     | 177781/400000 [00:26<00:31, 6959.44it/s] 45%|████▍     | 178478/400000 [00:26<00:32, 6771.71it/s] 45%|████▍     | 179165/400000 [00:26<00:32, 6799.74it/s] 45%|████▍     | 179847/400000 [00:26<00:32, 6760.32it/s] 45%|████▌     | 180524/400000 [00:26<00:32, 6716.59it/s] 45%|████▌     | 181197/400000 [00:26<00:33, 6596.42it/s] 45%|████▌     | 181910/400000 [00:27<00:32, 6745.48it/s] 46%|████▌     | 182601/400000 [00:27<00:32, 6783.66it/s] 46%|████▌     | 183295/400000 [00:27<00:31, 6827.99it/s] 46%|████▌     | 183988/400000 [00:27<00:31, 6856.06it/s] 46%|████▌     | 184675/400000 [00:27<00:31, 6840.33it/s] 46%|████▋     | 185360/400000 [00:27<00:32, 6691.67it/s] 47%|████▋     | 186031/400000 [00:27<00:32, 6631.11it/s] 47%|████▋     | 186695/400000 [00:27<00:32, 6618.37it/s] 47%|████▋     | 187370/400000 [00:27<00:31, 6649.75it/s] 47%|████▋     | 188077/400000 [00:27<00:31, 6770.21it/s] 47%|████▋     | 188815/400000 [00:28<00:30, 6940.20it/s] 47%|████▋     | 189533/400000 [00:28<00:30, 7009.24it/s] 48%|████▊     | 190267/400000 [00:28<00:29, 7104.87it/s] 48%|████▊     | 190979/400000 [00:28<00:29, 6982.57it/s] 48%|████▊     | 191679/400000 [00:28<00:29, 6972.54it/s] 48%|████▊     | 192396/400000 [00:28<00:29, 7030.45it/s] 48%|████▊     | 193116/400000 [00:28<00:29, 7078.84it/s] 48%|████▊     | 193852/400000 [00:28<00:28, 7160.54it/s] 49%|████▊     | 194569/400000 [00:28<00:29, 6956.39it/s] 49%|████▉     | 195267/400000 [00:28<00:29, 6873.47it/s] 49%|████▉     | 195956/400000 [00:29<00:29, 6871.49it/s] 49%|████▉     | 196658/400000 [00:29<00:29, 6913.39it/s] 49%|████▉     | 197379/400000 [00:29<00:28, 6999.11it/s] 50%|████▉     | 198080/400000 [00:29<00:29, 6860.47it/s] 50%|████▉     | 198768/400000 [00:29<00:29, 6775.33it/s] 50%|████▉     | 199447/400000 [00:29<00:29, 6714.52it/s] 50%|█████     | 200120/400000 [00:29<00:30, 6587.73it/s] 50%|█████     | 200782/400000 [00:29<00:30, 6594.83it/s] 50%|█████     | 201443/400000 [00:29<00:30, 6551.44it/s] 51%|█████     | 202156/400000 [00:29<00:29, 6712.79it/s] 51%|█████     | 202878/400000 [00:30<00:28, 6856.84it/s] 51%|█████     | 203607/400000 [00:30<00:28, 6979.54it/s] 51%|█████     | 204307/400000 [00:30<00:29, 6720.24it/s] 51%|█████▏    | 205022/400000 [00:30<00:28, 6817.89it/s] 51%|█████▏    | 205707/400000 [00:30<00:28, 6795.06it/s] 52%|█████▏    | 206459/400000 [00:30<00:27, 6997.01it/s] 52%|█████▏    | 207162/400000 [00:30<00:27, 6940.09it/s] 52%|█████▏    | 207881/400000 [00:30<00:27, 7012.52it/s] 52%|█████▏    | 208600/400000 [00:30<00:27, 7063.81it/s] 52%|█████▏    | 209336/400000 [00:31<00:26, 7147.40it/s] 53%|█████▎    | 210068/400000 [00:31<00:26, 7192.48it/s] 53%|█████▎    | 210789/400000 [00:31<00:26, 7186.77it/s] 53%|█████▎    | 211509/400000 [00:31<00:26, 7082.09it/s] 53%|█████▎    | 212218/400000 [00:31<00:26, 7065.63it/s] 53%|█████▎    | 212926/400000 [00:31<00:26, 6945.10it/s] 53%|█████▎    | 213646/400000 [00:31<00:26, 7019.61it/s] 54%|█████▎    | 214349/400000 [00:31<00:26, 6954.02it/s] 54%|█████▍    | 215058/400000 [00:31<00:26, 6992.47it/s] 54%|█████▍    | 215758/400000 [00:31<00:26, 6825.41it/s] 54%|█████▍    | 216465/400000 [00:32<00:26, 6894.76it/s] 54%|█████▍    | 217174/400000 [00:32<00:26, 6949.57it/s] 54%|█████▍    | 217882/400000 [00:32<00:26, 6975.91it/s] 55%|█████▍    | 218581/400000 [00:32<00:26, 6964.59it/s] 55%|█████▍    | 219318/400000 [00:32<00:25, 7081.35it/s] 55%|█████▌    | 220027/400000 [00:32<00:25, 7032.59it/s] 55%|█████▌    | 220731/400000 [00:32<00:25, 6947.19it/s] 55%|█████▌    | 221440/400000 [00:32<00:25, 6988.06it/s] 56%|█████▌    | 222140/400000 [00:32<00:26, 6746.68it/s] 56%|█████▌    | 222817/400000 [00:32<00:27, 6514.58it/s] 56%|█████▌    | 223486/400000 [00:33<00:26, 6552.02it/s] 56%|█████▌    | 224176/400000 [00:33<00:26, 6651.08it/s] 56%|█████▌    | 224883/400000 [00:33<00:25, 6770.21it/s] 56%|█████▋    | 225570/400000 [00:33<00:25, 6797.72it/s] 57%|█████▋    | 226293/400000 [00:33<00:25, 6921.27it/s] 57%|█████▋    | 227006/400000 [00:33<00:24, 6981.83it/s] 57%|█████▋    | 227714/400000 [00:33<00:24, 7008.34it/s] 57%|█████▋    | 228416/400000 [00:33<00:24, 6976.01it/s] 57%|█████▋    | 229115/400000 [00:33<00:25, 6832.46it/s] 57%|█████▋    | 229800/400000 [00:33<00:25, 6731.32it/s] 58%|█████▊    | 230475/400000 [00:34<00:26, 6471.53it/s] 58%|█████▊    | 231155/400000 [00:34<00:25, 6564.26it/s] 58%|█████▊    | 231861/400000 [00:34<00:25, 6703.85it/s] 58%|█████▊    | 232586/400000 [00:34<00:24, 6857.08it/s] 58%|█████▊    | 233304/400000 [00:34<00:23, 6948.61it/s] 59%|█████▊    | 234001/400000 [00:34<00:23, 6950.29it/s] 59%|█████▊    | 234724/400000 [00:34<00:23, 7031.06it/s] 59%|█████▉    | 235440/400000 [00:34<00:23, 7065.98it/s] 59%|█████▉    | 236148/400000 [00:34<00:23, 6996.59it/s] 59%|█████▉    | 236899/400000 [00:34<00:22, 7138.50it/s] 59%|█████▉    | 237615/400000 [00:35<00:22, 7126.29it/s] 60%|█████▉    | 238334/400000 [00:35<00:22, 7145.15it/s] 60%|█████▉    | 239050/400000 [00:35<00:22, 7066.76it/s] 60%|█████▉    | 239758/400000 [00:35<00:23, 6915.00it/s] 60%|██████    | 240451/400000 [00:35<00:23, 6745.99it/s] 60%|██████    | 241149/400000 [00:35<00:23, 6812.43it/s] 60%|██████    | 241883/400000 [00:35<00:22, 6961.52it/s] 61%|██████    | 242599/400000 [00:35<00:22, 7018.43it/s] 61%|██████    | 243303/400000 [00:35<00:22, 6818.86it/s] 61%|██████    | 243988/400000 [00:36<00:23, 6666.76it/s] 61%|██████    | 244687/400000 [00:36<00:22, 6759.61it/s] 61%|██████▏   | 245407/400000 [00:36<00:22, 6884.44it/s] 62%|██████▏   | 246107/400000 [00:36<00:22, 6915.58it/s] 62%|██████▏   | 246804/400000 [00:36<00:22, 6931.74it/s] 62%|██████▏   | 247499/400000 [00:36<00:22, 6809.33it/s] 62%|██████▏   | 248182/400000 [00:36<00:22, 6771.74it/s] 62%|██████▏   | 248931/400000 [00:36<00:21, 6971.30it/s] 62%|██████▏   | 249673/400000 [00:36<00:21, 7098.62it/s] 63%|██████▎   | 250385/400000 [00:36<00:21, 6827.72it/s] 63%|██████▎   | 251072/400000 [00:37<00:22, 6728.18it/s] 63%|██████▎   | 251768/400000 [00:37<00:21, 6794.79it/s] 63%|██████▎   | 252463/400000 [00:37<00:21, 6840.04it/s] 63%|██████▎   | 253190/400000 [00:37<00:21, 6961.25it/s] 63%|██████▎   | 253888/400000 [00:37<00:21, 6765.24it/s] 64%|██████▎   | 254567/400000 [00:37<00:21, 6688.24it/s] 64%|██████▍   | 255238/400000 [00:37<00:21, 6685.58it/s] 64%|██████▍   | 255947/400000 [00:37<00:21, 6801.06it/s] 64%|██████▍   | 256662/400000 [00:37<00:20, 6901.43it/s] 64%|██████▍   | 257354/400000 [00:37<00:20, 6893.80it/s] 65%|██████▍   | 258063/400000 [00:38<00:20, 6950.94it/s] 65%|██████▍   | 258789/400000 [00:38<00:20, 7039.36it/s] 65%|██████▍   | 259494/400000 [00:38<00:20, 6994.48it/s] 65%|██████▌   | 260195/400000 [00:38<00:20, 6855.75it/s] 65%|██████▌   | 260882/400000 [00:38<00:20, 6715.47it/s] 65%|██████▌   | 261576/400000 [00:38<00:20, 6781.09it/s] 66%|██████▌   | 262304/400000 [00:38<00:19, 6922.41it/s] 66%|██████▌   | 263013/400000 [00:38<00:19, 6970.57it/s] 66%|██████▌   | 263712/400000 [00:38<00:19, 6858.70it/s] 66%|██████▌   | 264400/400000 [00:39<00:21, 6444.92it/s] 66%|██████▋   | 265114/400000 [00:39<00:20, 6637.46it/s] 66%|██████▋   | 265800/400000 [00:39<00:20, 6700.87it/s] 67%|██████▋   | 266511/400000 [00:39<00:19, 6816.53it/s] 67%|██████▋   | 267205/400000 [00:39<00:19, 6851.81it/s] 67%|██████▋   | 267895/400000 [00:39<00:19, 6864.15it/s] 67%|██████▋   | 268584/400000 [00:39<00:20, 6464.33it/s] 67%|██████▋   | 269248/400000 [00:39<00:20, 6513.98it/s] 67%|██████▋   | 269943/400000 [00:39<00:19, 6637.22it/s] 68%|██████▊   | 270627/400000 [00:39<00:19, 6694.97it/s] 68%|██████▊   | 271300/400000 [00:40<00:19, 6588.80it/s] 68%|██████▊   | 271988/400000 [00:40<00:19, 6672.67it/s] 68%|██████▊   | 272703/400000 [00:40<00:18, 6807.13it/s] 68%|██████▊   | 273395/400000 [00:40<00:18, 6838.43it/s] 69%|██████▊   | 274081/400000 [00:40<00:18, 6801.26it/s] 69%|██████▊   | 274763/400000 [00:40<00:18, 6794.76it/s] 69%|██████▉   | 275444/400000 [00:40<00:18, 6762.50it/s] 69%|██████▉   | 276121/400000 [00:40<00:18, 6705.11it/s] 69%|██████▉   | 276793/400000 [00:40<00:18, 6708.94it/s] 69%|██████▉   | 277510/400000 [00:40<00:17, 6838.95it/s] 70%|██████▉   | 278248/400000 [00:41<00:17, 6989.61it/s] 70%|██████▉   | 278957/400000 [00:41<00:17, 7018.52it/s] 70%|██████▉   | 279684/400000 [00:41<00:16, 7089.71it/s] 70%|███████   | 280394/400000 [00:41<00:17, 6984.16it/s] 70%|███████   | 281094/400000 [00:41<00:17, 6878.39it/s] 70%|███████   | 281783/400000 [00:41<00:17, 6593.02it/s] 71%|███████   | 282446/400000 [00:41<00:17, 6576.74it/s] 71%|███████   | 283182/400000 [00:41<00:17, 6792.31it/s] 71%|███████   | 283904/400000 [00:41<00:16, 6914.55it/s] 71%|███████   | 284605/400000 [00:41<00:16, 6941.99it/s] 71%|███████▏  | 285308/400000 [00:42<00:16, 6966.99it/s] 72%|███████▏  | 286007/400000 [00:42<00:16, 6948.16it/s] 72%|███████▏  | 286716/400000 [00:42<00:16, 6987.59it/s] 72%|███████▏  | 287416/400000 [00:42<00:16, 6886.23it/s] 72%|███████▏  | 288129/400000 [00:42<00:16, 6956.61it/s] 72%|███████▏  | 288833/400000 [00:42<00:15, 6978.84it/s] 72%|███████▏  | 289549/400000 [00:42<00:15, 7032.01it/s] 73%|███████▎  | 290257/400000 [00:42<00:15, 7025.06it/s] 73%|███████▎  | 290960/400000 [00:42<00:15, 7021.90it/s] 73%|███████▎  | 291667/400000 [00:42<00:15, 7033.86it/s] 73%|███████▎  | 292371/400000 [00:43<00:15, 7029.69it/s] 73%|███████▎  | 293075/400000 [00:43<00:15, 7001.51it/s] 73%|███████▎  | 293776/400000 [00:43<00:15, 6878.78it/s] 74%|███████▎  | 294465/400000 [00:43<00:15, 6814.31it/s] 74%|███████▍  | 295148/400000 [00:43<00:15, 6818.49it/s] 74%|███████▍  | 295831/400000 [00:43<00:15, 6775.57it/s] 74%|███████▍  | 296513/400000 [00:43<00:15, 6787.55it/s] 74%|███████▍  | 297238/400000 [00:43<00:14, 6917.72it/s] 74%|███████▍  | 297935/400000 [00:43<00:14, 6930.38it/s] 75%|███████▍  | 298650/400000 [00:44<00:14, 6994.71it/s] 75%|███████▍  | 299373/400000 [00:44<00:14, 7062.98it/s] 75%|███████▌  | 300080/400000 [00:44<00:14, 7022.14it/s] 75%|███████▌  | 300783/400000 [00:44<00:14, 6708.02it/s] 75%|███████▌  | 301472/400000 [00:44<00:14, 6759.03it/s] 76%|███████▌  | 302151/400000 [00:44<00:14, 6637.36it/s] 76%|███████▌  | 302862/400000 [00:44<00:14, 6770.79it/s] 76%|███████▌  | 303542/400000 [00:44<00:14, 6624.64it/s] 76%|███████▌  | 304207/400000 [00:44<00:14, 6606.28it/s] 76%|███████▌  | 304870/400000 [00:44<00:14, 6609.80it/s] 76%|███████▋  | 305597/400000 [00:45<00:13, 6794.89it/s] 77%|███████▋  | 306293/400000 [00:45<00:13, 6841.47it/s] 77%|███████▋  | 307030/400000 [00:45<00:13, 6991.13it/s] 77%|███████▋  | 307731/400000 [00:45<00:13, 6941.22it/s] 77%|███████▋  | 308443/400000 [00:45<00:13, 6993.11it/s] 77%|███████▋  | 309176/400000 [00:45<00:12, 7089.40it/s] 77%|███████▋  | 309887/400000 [00:45<00:12, 7056.03it/s] 78%|███████▊  | 310594/400000 [00:45<00:12, 6990.65it/s] 78%|███████▊  | 311326/400000 [00:45<00:12, 7085.87it/s] 78%|███████▊  | 312048/400000 [00:45<00:12, 7122.71it/s] 78%|███████▊  | 312761/400000 [00:46<00:12, 7081.53it/s] 78%|███████▊  | 313470/400000 [00:46<00:12, 7043.71it/s] 79%|███████▊  | 314175/400000 [00:46<00:12, 6896.26it/s] 79%|███████▊  | 314866/400000 [00:46<00:12, 6719.78it/s] 79%|███████▉  | 315540/400000 [00:46<00:12, 6636.93it/s] 79%|███████▉  | 316206/400000 [00:46<00:12, 6607.32it/s] 79%|███████▉  | 316942/400000 [00:46<00:12, 6813.87it/s] 79%|███████▉  | 317651/400000 [00:46<00:11, 6892.15it/s] 80%|███████▉  | 318381/400000 [00:46<00:11, 7007.43it/s] 80%|███████▉  | 319084/400000 [00:46<00:11, 7006.98it/s] 80%|███████▉  | 319799/400000 [00:47<00:11, 7046.38it/s] 80%|████████  | 320524/400000 [00:47<00:11, 7098.16it/s] 80%|████████  | 321235/400000 [00:47<00:11, 7026.41it/s] 80%|████████  | 321939/400000 [00:47<00:11, 6885.84it/s] 81%|████████  | 322652/400000 [00:47<00:11, 6954.85it/s] 81%|████████  | 323349/400000 [00:47<00:11, 6737.56it/s] 81%|████████  | 324025/400000 [00:47<00:11, 6719.88it/s] 81%|████████  | 324699/400000 [00:47<00:11, 6619.72it/s] 81%|████████▏ | 325364/400000 [00:47<00:11, 6626.86it/s] 82%|████████▏ | 326036/400000 [00:48<00:11, 6653.22it/s] 82%|████████▏ | 326733/400000 [00:48<00:10, 6743.88it/s] 82%|████████▏ | 327409/400000 [00:48<00:10, 6728.60it/s] 82%|████████▏ | 328117/400000 [00:48<00:10, 6829.46it/s] 82%|████████▏ | 328837/400000 [00:48<00:10, 6935.96it/s] 82%|████████▏ | 329566/400000 [00:48<00:10, 7036.18it/s] 83%|████████▎ | 330280/400000 [00:48<00:09, 7065.18it/s] 83%|████████▎ | 331005/400000 [00:48<00:09, 7118.53it/s] 83%|████████▎ | 331718/400000 [00:48<00:09, 7014.63it/s] 83%|████████▎ | 332458/400000 [00:48<00:09, 7122.75it/s] 83%|████████▎ | 333172/400000 [00:49<00:09, 7031.38it/s] 83%|████████▎ | 333911/400000 [00:49<00:09, 7132.76it/s] 84%|████████▎ | 334634/400000 [00:49<00:09, 7159.80it/s] 84%|████████▍ | 335351/400000 [00:49<00:09, 7062.67it/s] 84%|████████▍ | 336059/400000 [00:49<00:09, 7038.30it/s] 84%|████████▍ | 336776/400000 [00:49<00:08, 7075.65it/s] 84%|████████▍ | 337485/400000 [00:49<00:08, 7027.11it/s] 85%|████████▍ | 338198/400000 [00:49<00:08, 7056.66it/s] 85%|████████▍ | 338919/400000 [00:49<00:08, 7100.04it/s] 85%|████████▍ | 339643/400000 [00:49<00:08, 7139.41it/s] 85%|████████▌ | 340358/400000 [00:50<00:08, 7109.81it/s] 85%|████████▌ | 341109/400000 [00:50<00:08, 7224.57it/s] 85%|████████▌ | 341833/400000 [00:50<00:08, 7200.94it/s] 86%|████████▌ | 342554/400000 [00:50<00:08, 7115.84it/s] 86%|████████▌ | 343292/400000 [00:50<00:07, 7175.64it/s] 86%|████████▌ | 344011/400000 [00:50<00:07, 7137.98it/s] 86%|████████▌ | 344726/400000 [00:50<00:07, 6967.12it/s] 86%|████████▋ | 345424/400000 [00:50<00:07, 6850.33it/s] 87%|████████▋ | 346111/400000 [00:50<00:08, 6683.94it/s] 87%|████████▋ | 346782/400000 [00:50<00:08, 6631.60it/s] 87%|████████▋ | 347478/400000 [00:51<00:07, 6725.80it/s] 87%|████████▋ | 348152/400000 [00:51<00:07, 6682.13it/s] 87%|████████▋ | 348822/400000 [00:51<00:07, 6631.37it/s] 87%|████████▋ | 349486/400000 [00:51<00:08, 6219.28it/s] 88%|████████▊ | 350213/400000 [00:51<00:07, 6500.25it/s] 88%|████████▊ | 350934/400000 [00:51<00:07, 6686.53it/s] 88%|████████▊ | 351662/400000 [00:51<00:07, 6853.28it/s] 88%|████████▊ | 352415/400000 [00:51<00:06, 7043.15it/s] 88%|████████▊ | 353125/400000 [00:51<00:06, 6990.27it/s] 88%|████████▊ | 353854/400000 [00:51<00:06, 7075.00it/s] 89%|████████▊ | 354571/400000 [00:52<00:06, 7100.54it/s] 89%|████████▉ | 355284/400000 [00:52<00:06, 7100.68it/s] 89%|████████▉ | 355996/400000 [00:52<00:06, 6924.94it/s] 89%|████████▉ | 356691/400000 [00:52<00:06, 6704.98it/s] 89%|████████▉ | 357365/400000 [00:52<00:06, 6566.75it/s] 90%|████████▉ | 358079/400000 [00:52<00:06, 6726.86it/s] 90%|████████▉ | 358786/400000 [00:52<00:06, 6825.27it/s] 90%|████████▉ | 359471/400000 [00:52<00:05, 6780.99it/s] 90%|█████████ | 360151/400000 [00:52<00:05, 6673.76it/s] 90%|█████████ | 360820/400000 [00:53<00:05, 6612.76it/s] 90%|█████████ | 361483/400000 [00:53<00:05, 6520.75it/s] 91%|█████████ | 362207/400000 [00:53<00:05, 6718.61it/s] 91%|█████████ | 362892/400000 [00:53<00:05, 6755.46it/s] 91%|█████████ | 363586/400000 [00:53<00:05, 6808.58it/s] 91%|█████████ | 364269/400000 [00:53<00:05, 6714.15it/s] 91%|█████████ | 364942/400000 [00:53<00:05, 6665.93it/s] 91%|█████████▏| 365679/400000 [00:53<00:05, 6858.39it/s] 92%|█████████▏| 366412/400000 [00:53<00:04, 6991.41it/s] 92%|█████████▏| 367123/400000 [00:53<00:04, 7024.42it/s] 92%|█████████▏| 367827/400000 [00:54<00:04, 6956.38it/s] 92%|█████████▏| 368549/400000 [00:54<00:04, 7032.86it/s] 92%|█████████▏| 369273/400000 [00:54<00:04, 7093.03it/s] 92%|█████████▏| 369984/400000 [00:54<00:04, 7083.74it/s] 93%|█████████▎| 370705/400000 [00:54<00:04, 7121.05it/s] 93%|█████████▎| 371434/400000 [00:54<00:03, 7169.54it/s] 93%|█████████▎| 372152/400000 [00:54<00:03, 7155.55it/s] 93%|█████████▎| 372882/400000 [00:54<00:03, 7197.58it/s] 93%|█████████▎| 373634/400000 [00:54<00:03, 7289.58it/s] 94%|█████████▎| 374364/400000 [00:54<00:03, 7099.98it/s] 94%|█████████▍| 375076/400000 [00:55<00:03, 7089.79it/s] 94%|█████████▍| 375821/400000 [00:55<00:03, 7194.02it/s] 94%|█████████▍| 376567/400000 [00:55<00:03, 7271.29it/s] 94%|█████████▍| 377296/400000 [00:55<00:03, 7057.53it/s] 95%|█████████▍| 378004/400000 [00:55<00:03, 7057.42it/s] 95%|█████████▍| 378712/400000 [00:55<00:03, 7010.30it/s] 95%|█████████▍| 379415/400000 [00:55<00:02, 6900.16it/s] 95%|█████████▌| 380157/400000 [00:55<00:02, 7045.92it/s] 95%|█████████▌| 380875/400000 [00:55<00:02, 7084.09it/s] 95%|█████████▌| 381608/400000 [00:55<00:02, 7155.00it/s] 96%|█████████▌| 382330/400000 [00:56<00:02, 7173.91it/s] 96%|█████████▌| 383049/400000 [00:56<00:02, 7178.28it/s] 96%|█████████▌| 383768/400000 [00:56<00:02, 7160.36it/s] 96%|█████████▌| 384485/400000 [00:56<00:02, 7004.56it/s] 96%|█████████▋| 385252/400000 [00:56<00:02, 7191.05it/s] 96%|█████████▋| 385974/400000 [00:56<00:02, 6972.76it/s] 97%|█████████▋| 386701/400000 [00:56<00:01, 7057.07it/s] 97%|█████████▋| 387419/400000 [00:56<00:01, 7092.51it/s] 97%|█████████▋| 388132/400000 [00:56<00:01, 7102.54it/s] 97%|█████████▋| 388866/400000 [00:57<00:01, 7171.10it/s] 97%|█████████▋| 389585/400000 [00:57<00:01, 7034.93it/s] 98%|█████████▊| 390290/400000 [00:57<00:01, 6997.58it/s] 98%|█████████▊| 391004/400000 [00:57<00:01, 7039.36it/s] 98%|█████████▊| 391709/400000 [00:57<00:01, 7036.97it/s] 98%|█████████▊| 392421/400000 [00:57<00:01, 7059.47it/s] 98%|█████████▊| 393128/400000 [00:57<00:00, 6876.62it/s] 98%|█████████▊| 393817/400000 [00:57<00:00, 6818.68it/s] 99%|█████████▊| 394517/400000 [00:57<00:00, 6870.79it/s] 99%|█████████▉| 395217/400000 [00:57<00:00, 6906.90it/s] 99%|█████████▉| 395909/400000 [00:58<00:00, 6818.55it/s] 99%|█████████▉| 396655/400000 [00:58<00:00, 6996.70it/s] 99%|█████████▉| 397377/400000 [00:58<00:00, 7060.48it/s]100%|█████████▉| 398085/400000 [00:58<00:00, 6733.08it/s]100%|█████████▉| 398763/400000 [00:58<00:00, 6729.83it/s]100%|█████████▉| 399467/400000 [00:58<00:00, 6819.00it/s]100%|█████████▉| 399999/400000 [00:58<00:00, 6821.42it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f264e880d30>, <torchtext.data.dataset.TabularDataset object at 0x7f26840c6b70>, <torchtext.vocab.Vocab object at 0x7f264e880d68>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.59 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.33 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.72 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.72 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.36 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.36 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.70 file/s]2020-07-22 00:21:57.183617: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-22 00:21:57.188678: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-07-22 00:21:57.189217: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557b3fc51320 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-22 00:21:57.189576: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▋      | 3612672/9912422 [00:00<00:00, 35836002.91it/s]9920512it [00:00, 34412479.54it/s]                             
0it [00:00, ?it/s]32768it [00:00, 518772.64it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470430.02it/s]1654784it [00:00, 11981065.44it/s]                         
0it [00:00, ?it/s]8192it [00:00, 270478.84it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
