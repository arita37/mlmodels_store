
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f224acf3ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f224acf3ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f22b5968488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f22b5968488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f22d4b94ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f22d4b94ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2262d0f9d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2262d0f9d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f2262d0f9d8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:20, 123556.85it/s] 17%|█▋        | 1687552/9912422 [00:00<00:46, 175921.79it/s] 51%|█████     | 5062656/9912422 [00:00<00:19, 250748.52it/s]9920512it [00:00, 23976339.89it/s]                           
0it [00:00, ?it/s]32768it [00:00, 611122.27it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 478545.67it/s]1654784it [00:00, 12151836.34it/s]                         
0it [00:00, ?it/s]8192it [00:00, 194613.20it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f22417336d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2241741978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f2262d0f620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f2262d0f620> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f2262d0f620> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:18,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:17,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:16,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:16,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:53,  2.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:53,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:52,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:51,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:51,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:51,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:50,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:50,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:33,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:33,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:33,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  7.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 10.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 25.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 33.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 49.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 64.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 64.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 71.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 79.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 85.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 85.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 85.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 85.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.23s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.23s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.23s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.53 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.45s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.23s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 85.53 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.45s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.45s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.38 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.45s/ url]
0 examples [00:00, ? examples/s]2020-07-24 12:09:17.463797: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-24 12:09:17.479086: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095139999 Hz
2020-07-24 12:09:17.479371: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56058a6a5ba0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 12:09:17.479405: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.53 examples/s]111 examples [00:00,  5.04 examples/s]219 examples [00:00,  7.18 examples/s]322 examples [00:00, 10.22 examples/s]429 examples [00:00, 14.55 examples/s]532 examples [00:00, 20.66 examples/s]641 examples [00:00, 29.27 examples/s]751 examples [00:00, 41.34 examples/s]861 examples [00:01, 58.12 examples/s]963 examples [00:01, 81.02 examples/s]1071 examples [00:01, 112.13 examples/s]1175 examples [00:01, 152.64 examples/s]1285 examples [00:01, 205.79 examples/s]1389 examples [00:01, 270.76 examples/s]1493 examples [00:01, 347.84 examples/s]1597 examples [00:01, 433.12 examples/s]1704 examples [00:01, 527.13 examples/s]1811 examples [00:02, 620.73 examples/s]1918 examples [00:02, 710.17 examples/s]2027 examples [00:02, 792.91 examples/s]2137 examples [00:02, 864.09 examples/s]2245 examples [00:02, 908.64 examples/s]2352 examples [00:02, 947.96 examples/s]2460 examples [00:02, 983.21 examples/s]2569 examples [00:02, 1011.16 examples/s]2676 examples [00:02, 1019.71 examples/s]2783 examples [00:02, 1030.72 examples/s]2892 examples [00:03, 1045.98 examples/s]2999 examples [00:03, 1021.77 examples/s]3103 examples [00:03, 1025.65 examples/s]3208 examples [00:03, 1032.48 examples/s]3316 examples [00:03, 1045.24 examples/s]3424 examples [00:03, 1054.31 examples/s]3534 examples [00:03, 1066.15 examples/s]3643 examples [00:03, 1072.16 examples/s]3751 examples [00:03, 1060.33 examples/s]3858 examples [00:03, 1032.13 examples/s]3968 examples [00:04, 1051.53 examples/s]4075 examples [00:04, 1055.77 examples/s]4181 examples [00:04, 1044.30 examples/s]4286 examples [00:04, 1030.10 examples/s]4390 examples [00:04, 998.98 examples/s] 4491 examples [00:04, 993.64 examples/s]4591 examples [00:04, 982.68 examples/s]4699 examples [00:04, 1009.98 examples/s]4802 examples [00:04, 1013.30 examples/s]4907 examples [00:04, 1022.10 examples/s]5010 examples [00:05, 1018.76 examples/s]5119 examples [00:05, 1037.46 examples/s]5228 examples [00:05, 1050.65 examples/s]5334 examples [00:05, 1033.65 examples/s]5438 examples [00:05, 1029.03 examples/s]5542 examples [00:05, 1017.68 examples/s]5644 examples [00:05, 1013.25 examples/s]5748 examples [00:05, 1019.52 examples/s]5857 examples [00:05, 1037.45 examples/s]5962 examples [00:05, 1038.13 examples/s]6066 examples [00:06, 1023.72 examples/s]6172 examples [00:06, 1032.11 examples/s]6276 examples [00:06, 1024.89 examples/s]6379 examples [00:06, 1007.45 examples/s]6480 examples [00:06, 986.32 examples/s] 6585 examples [00:06, 1004.36 examples/s]6686 examples [00:06, 1004.42 examples/s]6793 examples [00:06, 1022.71 examples/s]6898 examples [00:06, 1030.52 examples/s]7006 examples [00:07, 1043.68 examples/s]7115 examples [00:07, 1055.33 examples/s]7223 examples [00:07, 1061.22 examples/s]7330 examples [00:07, 1033.74 examples/s]7434 examples [00:07, 1018.73 examples/s]7537 examples [00:07, 1007.56 examples/s]7639 examples [00:07, 1009.99 examples/s]7743 examples [00:07, 1017.08 examples/s]7851 examples [00:07, 1033.60 examples/s]7957 examples [00:07, 1040.23 examples/s]8064 examples [00:08, 1048.67 examples/s]8169 examples [00:08, 1034.40 examples/s]8278 examples [00:08, 1048.11 examples/s]8383 examples [00:08, 1046.91 examples/s]8491 examples [00:08, 1052.74 examples/s]8597 examples [00:08, 1045.46 examples/s]8702 examples [00:08, 1040.83 examples/s]8807 examples [00:08, 1034.68 examples/s]8911 examples [00:08, 1026.04 examples/s]9018 examples [00:08, 1037.98 examples/s]9124 examples [00:09, 1043.57 examples/s]9232 examples [00:09, 1052.55 examples/s]9338 examples [00:09, 1048.84 examples/s]9443 examples [00:09, 1048.03 examples/s]9548 examples [00:09, 1042.57 examples/s]9653 examples [00:09, 1032.98 examples/s]9762 examples [00:09, 1047.34 examples/s]9867 examples [00:09, 1042.41 examples/s]9972 examples [00:09, 1037.66 examples/s]10076 examples [00:09, 991.23 examples/s]10176 examples [00:10, 986.19 examples/s]10280 examples [00:10, 999.94 examples/s]10383 examples [00:10, 1008.39 examples/s]10492 examples [00:10, 1031.09 examples/s]10601 examples [00:10, 1045.17 examples/s]10708 examples [00:10, 1049.82 examples/s]10817 examples [00:10, 1059.24 examples/s]10924 examples [00:10, 1057.93 examples/s]11030 examples [00:10, 1039.02 examples/s]11138 examples [00:11, 1050.33 examples/s]11244 examples [00:11, 1034.93 examples/s]11349 examples [00:11, 1038.33 examples/s]11455 examples [00:11, 1043.23 examples/s]11564 examples [00:11, 1054.81 examples/s]11672 examples [00:11, 1060.12 examples/s]11781 examples [00:11, 1068.80 examples/s]11890 examples [00:11, 1074.27 examples/s]12000 examples [00:11, 1079.48 examples/s]12109 examples [00:11, 1080.34 examples/s]12218 examples [00:12, 1075.81 examples/s]12326 examples [00:12, 1076.13 examples/s]12434 examples [00:12, 1072.61 examples/s]12544 examples [00:12, 1078.26 examples/s]12655 examples [00:12, 1085.23 examples/s]12764 examples [00:12, 1083.72 examples/s]12873 examples [00:12, 1081.13 examples/s]12982 examples [00:12, 1080.10 examples/s]13091 examples [00:12, 1049.82 examples/s]13200 examples [00:12, 1059.08 examples/s]13311 examples [00:13, 1071.62 examples/s]13419 examples [00:13, 1063.79 examples/s]13528 examples [00:13, 1069.61 examples/s]13637 examples [00:13, 1075.37 examples/s]13746 examples [00:13, 1077.05 examples/s]13854 examples [00:13, 1077.25 examples/s]13962 examples [00:13, 1077.18 examples/s]14070 examples [00:13, 1056.73 examples/s]14180 examples [00:13, 1068.56 examples/s]14287 examples [00:13, 1066.36 examples/s]14395 examples [00:14, 1068.55 examples/s]14504 examples [00:14, 1073.69 examples/s]14612 examples [00:14, 1064.19 examples/s]14719 examples [00:14, 1050.44 examples/s]14826 examples [00:14, 1054.45 examples/s]14935 examples [00:14, 1063.03 examples/s]15044 examples [00:14, 1069.67 examples/s]15152 examples [00:14, 1064.44 examples/s]15262 examples [00:14, 1072.49 examples/s]15372 examples [00:14, 1080.41 examples/s]15481 examples [00:15, 1076.84 examples/s]15590 examples [00:15, 1080.66 examples/s]15699 examples [00:15, 1069.04 examples/s]15808 examples [00:15, 1073.88 examples/s]15916 examples [00:15, 1051.93 examples/s]16024 examples [00:15, 1057.92 examples/s]16132 examples [00:15, 1063.29 examples/s]16239 examples [00:15, 1045.45 examples/s]16344 examples [00:15, 1020.65 examples/s]16450 examples [00:15, 1030.13 examples/s]16556 examples [00:16, 1038.08 examples/s]16665 examples [00:16, 1050.89 examples/s]16774 examples [00:16, 1059.67 examples/s]16883 examples [00:16, 1067.29 examples/s]16993 examples [00:16, 1075.78 examples/s]17101 examples [00:16, 1071.24 examples/s]17209 examples [00:16, 1071.45 examples/s]17317 examples [00:16, 1053.40 examples/s]17423 examples [00:16, 1041.79 examples/s]17532 examples [00:17, 1054.31 examples/s]17638 examples [00:17, 1020.02 examples/s]17744 examples [00:17, 1030.81 examples/s]17854 examples [00:17, 1048.90 examples/s]17963 examples [00:17, 1058.62 examples/s]18073 examples [00:17, 1068.55 examples/s]18181 examples [00:17, 1063.19 examples/s]18288 examples [00:17, 1050.02 examples/s]18398 examples [00:17, 1063.87 examples/s]18509 examples [00:17, 1075.51 examples/s]18619 examples [00:18, 1080.87 examples/s]18728 examples [00:18, 1081.34 examples/s]18838 examples [00:18, 1085.09 examples/s]18948 examples [00:18, 1087.45 examples/s]19057 examples [00:18, 1069.43 examples/s]19165 examples [00:18, 1069.51 examples/s]19273 examples [00:18, 1068.84 examples/s]19380 examples [00:18, 1060.60 examples/s]19488 examples [00:18, 1064.68 examples/s]19597 examples [00:18, 1072.02 examples/s]19706 examples [00:19, 1076.40 examples/s]19814 examples [00:19, 1056.94 examples/s]19920 examples [00:19, 1056.44 examples/s]20026 examples [00:19, 1012.57 examples/s]20137 examples [00:19, 1038.79 examples/s]20247 examples [00:19, 1055.59 examples/s]20357 examples [00:19, 1065.91 examples/s]20466 examples [00:19, 1071.83 examples/s]20574 examples [00:19, 1063.21 examples/s]20684 examples [00:19, 1073.79 examples/s]20793 examples [00:20, 1076.58 examples/s]20902 examples [00:20, 1080.33 examples/s]21012 examples [00:20, 1085.35 examples/s]21121 examples [00:20, 1081.12 examples/s]21230 examples [00:20, 1083.76 examples/s]21339 examples [00:20, 1085.02 examples/s]21448 examples [00:20, 1080.03 examples/s]21557 examples [00:20, 1066.32 examples/s]21667 examples [00:20, 1050.00 examples/s]21777 examples [00:20, 1063.00 examples/s]21887 examples [00:21, 1072.82 examples/s]21996 examples [00:21, 1077.66 examples/s]22106 examples [00:21, 1082.80 examples/s]22216 examples [00:21, 1086.74 examples/s]22325 examples [00:21, 1067.38 examples/s]22432 examples [00:21, 1053.52 examples/s]22539 examples [00:21, 1055.95 examples/s]22647 examples [00:21, 1061.70 examples/s]22756 examples [00:21, 1068.69 examples/s]22865 examples [00:22, 1074.52 examples/s]22974 examples [00:22, 1078.71 examples/s]23082 examples [00:22, 1072.35 examples/s]23190 examples [00:22, 1069.88 examples/s]23300 examples [00:22, 1078.33 examples/s]23408 examples [00:22, 1072.50 examples/s]23518 examples [00:22, 1079.89 examples/s]23627 examples [00:22, 1079.77 examples/s]23736 examples [00:22, 1074.23 examples/s]23844 examples [00:22, 1060.52 examples/s]23953 examples [00:23, 1068.93 examples/s]24064 examples [00:23, 1079.21 examples/s]24172 examples [00:23, 1060.71 examples/s]24283 examples [00:23, 1073.19 examples/s]24393 examples [00:23, 1080.19 examples/s]24502 examples [00:23, 1079.27 examples/s]24611 examples [00:23, 1082.00 examples/s]24720 examples [00:23, 1052.88 examples/s]24828 examples [00:23, 1060.38 examples/s]24935 examples [00:23, 1056.40 examples/s]25045 examples [00:24, 1067.19 examples/s]25155 examples [00:24, 1074.53 examples/s]25263 examples [00:24, 1049.99 examples/s]25373 examples [00:24, 1064.06 examples/s]25484 examples [00:24, 1075.10 examples/s]25594 examples [00:24, 1080.67 examples/s]25703 examples [00:24, 1057.90 examples/s]25809 examples [00:24, 1019.04 examples/s]25912 examples [00:24, 1010.89 examples/s]26016 examples [00:24, 1017.10 examples/s]26118 examples [00:25, 995.20 examples/s] 26225 examples [00:25, 1014.16 examples/s]26332 examples [00:25, 1028.50 examples/s]26443 examples [00:25, 1049.88 examples/s]26552 examples [00:25, 1060.92 examples/s]26661 examples [00:25, 1069.36 examples/s]26769 examples [00:25, 1071.08 examples/s]26877 examples [00:25, 1066.38 examples/s]26985 examples [00:25, 1068.04 examples/s]27093 examples [00:25, 1070.32 examples/s]27201 examples [00:26, 1060.73 examples/s]27308 examples [00:26, 1043.68 examples/s]27413 examples [00:26, 1024.44 examples/s]27519 examples [00:26, 1033.11 examples/s]27624 examples [00:26, 1037.24 examples/s]27728 examples [00:26, 1026.49 examples/s]27831 examples [00:26, 1019.65 examples/s]27934 examples [00:26, 1011.74 examples/s]28036 examples [00:26, 992.21 examples/s] 28136 examples [00:27, 980.68 examples/s]28236 examples [00:27, 984.98 examples/s]28335 examples [00:27, 986.08 examples/s]28436 examples [00:27, 990.71 examples/s]28536 examples [00:27, 979.24 examples/s]28636 examples [00:27, 984.21 examples/s]28738 examples [00:27, 993.21 examples/s]28838 examples [00:27, 988.15 examples/s]28941 examples [00:27, 998.51 examples/s]29046 examples [00:27, 1012.22 examples/s]29154 examples [00:28, 1031.59 examples/s]29258 examples [00:28, 1026.68 examples/s]29361 examples [00:28, 1002.93 examples/s]29466 examples [00:28, 1015.41 examples/s]29568 examples [00:28, 1011.55 examples/s]29671 examples [00:28, 1016.00 examples/s]29773 examples [00:28, 1013.93 examples/s]29879 examples [00:28, 1025.39 examples/s]29982 examples [00:28, 1018.24 examples/s]30084 examples [00:28, 960.52 examples/s] 30194 examples [00:29, 997.22 examples/s]30300 examples [00:29, 1013.33 examples/s]30404 examples [00:29, 1020.16 examples/s]30514 examples [00:29, 1040.66 examples/s]30619 examples [00:29, 1043.16 examples/s]30729 examples [00:29, 1058.73 examples/s]30838 examples [00:29, 1064.88 examples/s]30945 examples [00:29, 1059.27 examples/s]31052 examples [00:29, 1059.35 examples/s]31159 examples [00:29, 1055.16 examples/s]31268 examples [00:30, 1064.83 examples/s]31379 examples [00:30, 1075.91 examples/s]31487 examples [00:30, 1066.04 examples/s]31597 examples [00:30, 1073.28 examples/s]31705 examples [00:30, 1069.56 examples/s]31813 examples [00:30, 1042.95 examples/s]31918 examples [00:30, 1011.94 examples/s]32020 examples [00:30, 1006.14 examples/s]32129 examples [00:30, 1028.75 examples/s]32233 examples [00:31, 1030.75 examples/s]32337 examples [00:31, 1018.07 examples/s]32439 examples [00:31, 1017.36 examples/s]32541 examples [00:31, 977.82 examples/s] 32649 examples [00:31, 1004.43 examples/s]32758 examples [00:31, 1027.32 examples/s]32863 examples [00:31, 1031.67 examples/s]32967 examples [00:31, 1026.31 examples/s]33070 examples [00:31, 988.57 examples/s] 33175 examples [00:31, 1005.83 examples/s]33281 examples [00:32, 1021.29 examples/s]33384 examples [00:32, 1021.49 examples/s]33487 examples [00:32, 1015.79 examples/s]33589 examples [00:32, 1008.21 examples/s]33698 examples [00:32, 1028.06 examples/s]33807 examples [00:32, 1045.35 examples/s]33914 examples [00:32, 1049.98 examples/s]34024 examples [00:32, 1064.42 examples/s]34134 examples [00:32, 1072.76 examples/s]34242 examples [00:32, 1069.34 examples/s]34352 examples [00:33, 1077.29 examples/s]34463 examples [00:33, 1085.59 examples/s]34572 examples [00:33, 1072.01 examples/s]34681 examples [00:33, 1074.56 examples/s]34791 examples [00:33, 1080.81 examples/s]34900 examples [00:33, 1079.45 examples/s]35009 examples [00:33, 1080.67 examples/s]35118 examples [00:33, 1058.64 examples/s]35224 examples [00:33, 1050.79 examples/s]35330 examples [00:33, 1049.42 examples/s]35437 examples [00:34, 1053.18 examples/s]35543 examples [00:34, 1053.44 examples/s]35654 examples [00:34, 1067.61 examples/s]35763 examples [00:34, 1071.45 examples/s]35871 examples [00:34, 1060.78 examples/s]35978 examples [00:34, 1041.34 examples/s]36085 examples [00:34, 1048.74 examples/s]36190 examples [00:34, 1045.29 examples/s]36297 examples [00:34, 1051.35 examples/s]36406 examples [00:35, 1062.12 examples/s]36514 examples [00:35, 1065.45 examples/s]36625 examples [00:35, 1077.43 examples/s]36733 examples [00:35, 1061.78 examples/s]36842 examples [00:35, 1068.10 examples/s]36951 examples [00:35, 1072.30 examples/s]37059 examples [00:35, 1067.27 examples/s]37168 examples [00:35, 1072.06 examples/s]37277 examples [00:35, 1075.67 examples/s]37385 examples [00:35, 1076.17 examples/s]37494 examples [00:36, 1079.26 examples/s]37603 examples [00:36, 1080.18 examples/s]37712 examples [00:36, 1082.20 examples/s]37821 examples [00:36, 1083.05 examples/s]37930 examples [00:36, 1057.55 examples/s]38038 examples [00:36, 1061.21 examples/s]38147 examples [00:36, 1067.40 examples/s]38256 examples [00:36, 1073.94 examples/s]38364 examples [00:36, 1071.51 examples/s]38474 examples [00:36, 1076.97 examples/s]38583 examples [00:37, 1080.65 examples/s]38693 examples [00:37, 1085.40 examples/s]38802 examples [00:37, 1078.68 examples/s]38910 examples [00:37, 1076.97 examples/s]39018 examples [00:37, 1070.42 examples/s]39128 examples [00:37, 1078.27 examples/s]39236 examples [00:37, 1078.40 examples/s]39344 examples [00:37, 1072.06 examples/s]39452 examples [00:37, 1062.67 examples/s]39560 examples [00:37, 1065.37 examples/s]39667 examples [00:38, 1051.18 examples/s]39773 examples [00:38, 1043.77 examples/s]39878 examples [00:38, 1015.37 examples/s]39982 examples [00:38, 1020.46 examples/s]40085 examples [00:38, 972.63 examples/s] 40190 examples [00:38, 992.57 examples/s]40293 examples [00:38, 1002.33 examples/s]40394 examples [00:38, 976.85 examples/s] 40499 examples [00:38, 996.33 examples/s]40608 examples [00:38, 1022.10 examples/s]40714 examples [00:39, 1031.80 examples/s]40823 examples [00:39, 1047.82 examples/s]40932 examples [00:39, 1057.89 examples/s]41039 examples [00:39, 1035.55 examples/s]41143 examples [00:39, 1010.91 examples/s]41252 examples [00:39, 1032.86 examples/s]41361 examples [00:39, 1045.43 examples/s]41466 examples [00:39, 1033.83 examples/s]41571 examples [00:39, 1038.59 examples/s]41676 examples [00:40, 1035.34 examples/s]41780 examples [00:40, 1023.48 examples/s]41884 examples [00:40, 1028.33 examples/s]41992 examples [00:40, 1041.52 examples/s]42102 examples [00:40, 1057.31 examples/s]42208 examples [00:40, 1056.53 examples/s]42317 examples [00:40, 1065.23 examples/s]42428 examples [00:40, 1075.57 examples/s]42536 examples [00:40, 1061.97 examples/s]42645 examples [00:40, 1068.47 examples/s]42752 examples [00:41, 1058.64 examples/s]42858 examples [00:41, 1048.45 examples/s]42969 examples [00:41, 1064.61 examples/s]43076 examples [00:41, 1050.52 examples/s]43184 examples [00:41, 1056.50 examples/s]43291 examples [00:41, 1057.47 examples/s]43397 examples [00:41, 1058.14 examples/s]43505 examples [00:41, 1064.24 examples/s]43613 examples [00:41, 1068.49 examples/s]43721 examples [00:41, 1070.43 examples/s]43829 examples [00:42, 1048.96 examples/s]43937 examples [00:42, 1056.22 examples/s]44045 examples [00:42, 1061.03 examples/s]44152 examples [00:42, 1061.96 examples/s]44261 examples [00:42, 1069.61 examples/s]44369 examples [00:42, 1071.69 examples/s]44477 examples [00:42, 1073.03 examples/s]44585 examples [00:42, 1074.63 examples/s]44693 examples [00:42, 1075.79 examples/s]44803 examples [00:42, 1082.69 examples/s]44912 examples [00:43, 1083.29 examples/s]45021 examples [00:43, 1072.21 examples/s]45129 examples [00:43, 1050.93 examples/s]45237 examples [00:43, 1058.33 examples/s]45343 examples [00:43, 1048.84 examples/s]45448 examples [00:43, 971.79 examples/s] 45547 examples [00:43, 939.52 examples/s]45655 examples [00:43, 975.13 examples/s]45764 examples [00:43, 1005.74 examples/s]45871 examples [00:44, 1023.46 examples/s]45978 examples [00:44, 1035.02 examples/s]46088 examples [00:44, 1051.22 examples/s]46198 examples [00:44, 1063.63 examples/s]46307 examples [00:44, 1070.05 examples/s]46415 examples [00:44, 1038.77 examples/s]46520 examples [00:44, 1040.28 examples/s]46630 examples [00:44, 1052.26 examples/s]46736 examples [00:44, 997.94 examples/s] 46837 examples [00:44, 980.87 examples/s]46947 examples [00:45, 1011.34 examples/s]47055 examples [00:45, 1028.78 examples/s]47165 examples [00:45, 1048.84 examples/s]47275 examples [00:45, 1062.42 examples/s]47383 examples [00:45, 1065.94 examples/s]47492 examples [00:45, 1071.19 examples/s]47603 examples [00:45, 1080.95 examples/s]47712 examples [00:45, 1071.41 examples/s]47820 examples [00:45, 1064.71 examples/s]47930 examples [00:45, 1073.46 examples/s]48041 examples [00:46, 1081.41 examples/s]48150 examples [00:46, 1083.55 examples/s]48259 examples [00:46, 1077.17 examples/s]48368 examples [00:46, 1080.47 examples/s]48477 examples [00:46, 1080.30 examples/s]48586 examples [00:46, 1061.69 examples/s]48694 examples [00:46, 1065.24 examples/s]48802 examples [00:46, 1068.62 examples/s]48912 examples [00:46, 1076.24 examples/s]49023 examples [00:46, 1084.95 examples/s]49134 examples [00:47, 1089.49 examples/s]49244 examples [00:47, 1089.63 examples/s]49354 examples [00:47, 1090.98 examples/s]49464 examples [00:47, 1056.18 examples/s]49573 examples [00:47, 1063.46 examples/s]49682 examples [00:47, 1070.92 examples/s]49791 examples [00:47, 1075.91 examples/s]49899 examples [00:47, 1071.13 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6285/50000 [00:00<00:00, 62849.61 examples/s] 38%|███▊      | 19112/50000 [00:00<00:00, 74202.47 examples/s] 64%|██████▍   | 32220/50000 [00:00<00:00, 85307.07 examples/s] 89%|████████▉ | 44671/50000 [00:00<00:00, 94196.51 examples/s]                                                               0 examples [00:00, ? examples/s]92 examples [00:00, 918.30 examples/s]202 examples [00:00, 964.74 examples/s]309 examples [00:00, 991.07 examples/s]410 examples [00:00, 994.32 examples/s]521 examples [00:00, 1025.24 examples/s]632 examples [00:00, 1048.39 examples/s]744 examples [00:00, 1067.68 examples/s]853 examples [00:00, 1073.45 examples/s]962 examples [00:00, 1077.38 examples/s]1072 examples [00:01, 1082.68 examples/s]1179 examples [00:01, 1076.09 examples/s]1290 examples [00:01, 1083.43 examples/s]1399 examples [00:01, 1084.69 examples/s]1510 examples [00:01, 1091.88 examples/s]1621 examples [00:01, 1097.00 examples/s]1732 examples [00:01, 1099.09 examples/s]1844 examples [00:01, 1102.89 examples/s]1955 examples [00:01, 1096.67 examples/s]2067 examples [00:01, 1101.09 examples/s]2178 examples [00:02, 1097.89 examples/s]2290 examples [00:02, 1102.90 examples/s]2401 examples [00:02, 1093.64 examples/s]2512 examples [00:02, 1097.46 examples/s]2622 examples [00:02, 1094.58 examples/s]2732 examples [00:02, 1092.80 examples/s]2842 examples [00:02, 1094.40 examples/s]2954 examples [00:02, 1101.56 examples/s]3065 examples [00:02, 1103.94 examples/s]3176 examples [00:02, 1104.68 examples/s]3288 examples [00:03, 1108.73 examples/s]3400 examples [00:03, 1109.95 examples/s]3512 examples [00:03, 1111.74 examples/s]3624 examples [00:03, 1102.20 examples/s]3735 examples [00:03, 1104.12 examples/s]3846 examples [00:03, 1101.03 examples/s]3957 examples [00:03, 1097.62 examples/s]4067 examples [00:03, 1097.72 examples/s]4177 examples [00:03, 1095.29 examples/s]4287 examples [00:03, 1096.01 examples/s]4397 examples [00:04, 1093.24 examples/s]4507 examples [00:04, 1083.84 examples/s]4619 examples [00:04, 1092.36 examples/s]4730 examples [00:04, 1094.63 examples/s]4841 examples [00:04, 1099.06 examples/s]4951 examples [00:04, 1093.13 examples/s]5062 examples [00:04, 1096.10 examples/s]5175 examples [00:04, 1103.52 examples/s]5286 examples [00:04, 1088.16 examples/s]5397 examples [00:04, 1092.59 examples/s]5507 examples [00:05, 1082.15 examples/s]5617 examples [00:05, 1084.90 examples/s]5728 examples [00:05, 1090.89 examples/s]5839 examples [00:05, 1094.21 examples/s]5950 examples [00:05, 1098.12 examples/s]6062 examples [00:05, 1101.84 examples/s]6173 examples [00:05, 1089.22 examples/s]6285 examples [00:05, 1096.34 examples/s]6395 examples [00:05, 1095.93 examples/s]6506 examples [00:05, 1099.71 examples/s]6618 examples [00:06, 1104.54 examples/s]6729 examples [00:06, 1092.05 examples/s]6839 examples [00:06, 1085.48 examples/s]6950 examples [00:06, 1091.03 examples/s]7060 examples [00:06, 1086.90 examples/s]7169 examples [00:06, 1073.32 examples/s]7281 examples [00:06, 1084.22 examples/s]7393 examples [00:06, 1092.90 examples/s]7503 examples [00:06, 1092.92 examples/s]7615 examples [00:06, 1098.17 examples/s]7725 examples [00:07, 1096.84 examples/s]7835 examples [00:07, 1094.05 examples/s]7947 examples [00:07, 1099.08 examples/s]8057 examples [00:07, 1099.19 examples/s]8167 examples [00:07, 1078.20 examples/s]8278 examples [00:07, 1085.42 examples/s]8388 examples [00:07, 1087.47 examples/s]8497 examples [00:07, 1077.34 examples/s]8606 examples [00:07, 1078.42 examples/s]8714 examples [00:07, 1070.74 examples/s]8825 examples [00:08, 1079.88 examples/s]8934 examples [00:08, 1082.09 examples/s]9043 examples [00:08, 1079.28 examples/s]9151 examples [00:08, 1071.71 examples/s]9261 examples [00:08, 1079.09 examples/s]9371 examples [00:08, 1084.94 examples/s]9482 examples [00:08, 1090.01 examples/s]9592 examples [00:08, 1088.44 examples/s]9701 examples [00:08, 1082.21 examples/s]9811 examples [00:09, 1085.16 examples/s]9920 examples [00:09, 1084.12 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9GX1QD/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9GX1QD/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2262d0f9d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2262d0f9d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f2262d0f9d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f22ce19ff98>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f21ecfe5f28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2262d0f9d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2262d0f9d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f2262d0f9d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2241741358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f22417411d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f21db6482f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f21db6482f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f21db6482f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f21db648400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f21db648400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f21db648400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=fc80628e2d6870b942fe620758eb996cb79b80f0cded2578848321478f13f00d
  Stored in directory: /tmp/pip-ephem-wheel-cache-_sgz1f14/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:13:27, 11.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:23:30, 16.6kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<10:07:11, 23.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:05:33, 33.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.01M/862M [00:01<4:57:54, 48.1kB/s].vector_cache/glove.6B.zip:   1%|          | 6.73M/862M [00:01<3:27:30, 68.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.5M/862M [00:01<2:24:31, 98.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.0M/862M [00:01<1:40:44, 140kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.7M/862M [00:01<1:10:18, 200kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.8M/862M [00:01<49:01, 285kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.6M/862M [00:01<34:12, 406kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.4M/862M [00:01<23:57, 577kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:02<16:44, 820kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:02<11:48, 1.16MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.3M/862M [00:02<08:17, 1.64MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:02<05:54, 2.29MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<05:19, 2.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<05:38, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<05:31, 2.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:04<04:14, 3.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<03:13, 4.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<07:43, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<06:58, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:06<05:11, 2.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<06:28, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<07:22, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<05:52, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:09<04:16, 3.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<42:36, 310kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<31:10, 424kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<22:07, 596kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<18:31, 710kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<14:18, 919kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<10:16, 1.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<10:18, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<08:32, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:14<06:18, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<07:29, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:35, 1.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:16<04:56, 2.63MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<06:30, 1.99MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<05:53, 2.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:18<04:26, 2.91MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<06:09, 2.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<05:23, 2.39MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<04:07, 3.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:20<03:01, 4.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<50:35, 253kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<36:42, 349kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:22<25:58, 492kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<21:09, 603kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<16:05, 792kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<11:33, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:04, 1.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:02, 1.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<06:38, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:37, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:37, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<04:53, 2.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:25, 1.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:03, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:34, 2.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<04:01, 3.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<20:58, 594kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<15:59, 780kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<11:29, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:54, 1.14MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:10, 1.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:38, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<05:31, 2.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:43, 1.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:22, 1.67MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:28, 2.25MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:41, 1.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:45, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:22, 2.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<03:11, 3.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<51:20, 238kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<36:55, 330kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<26:06, 466kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<18:22, 660kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<58:40, 207kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<43:40, 278kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<31:10, 389kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<21:52, 551kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<1:18:12, 154kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<55:58, 215kB/s]  .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<39:24, 305kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<30:18, 396kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<23:40, 506kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<17:10, 697kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<13:54, 858kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:58, 1.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<07:56, 1.50MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<08:18, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:15, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:16, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:35, 2.58MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<07:27, 1.58MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:25, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:47, 2.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:05, 1.93MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:27, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:06, 2.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:37, 2.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:07, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:52, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:27, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<04:48, 2.41MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<03:38, 3.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<02:41, 4.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<45:19, 254kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<34:04, 338kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<24:24, 471kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<18:52, 606kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<14:12, 804kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<10:15, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<07:17, 1.56MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<43:13, 263kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<31:24, 362kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<22:13, 510kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<18:11, 621kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<13:51, 815kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<09:58, 1.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<09:36, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:53, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:47, 1.93MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:41, 1.67MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:50, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<04:21, 2.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:40, 1.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:07, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:52, 2.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:18, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:50, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:39, 3.00MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:09, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<05:51, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:34, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:26, 3.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:19, 2.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:51, 2.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:38, 2.99MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:04, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:39, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:32, 3.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:00, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:36, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:27, 3.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:57, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:39, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:25, 2.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:15, 3.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:13, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:29, 1.93MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:06, 2.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:20, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:48, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:37, 2.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:00, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:34, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:26, 3.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:52, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:32, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:20, 2.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:09, 3.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<10:14, 1.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:04, 1.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:01, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<04:18, 2.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<27:50, 369kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<21:35, 476kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<15:37, 657kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<10:59, 929kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<9:57:46, 17.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<6:59:13, 24.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<4:52:58, 34.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<3:26:44, 49.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<2:26:46, 69.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<1:43:02, 98.2kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<1:12:09, 140kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<53:21, 189kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<38:23, 262kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<27:04, 371kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<21:13, 471kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<16:55, 591kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<12:15, 815kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:49, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<08:36, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<06:52, 1.44MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:00, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:37, 2.72MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<39:44, 248kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<29:49, 331kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<21:21, 461kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<16:27, 595kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<12:31, 782kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<08:57, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<08:32, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<08:00, 1.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:05, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:48, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:54, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:37, 2.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<02:40, 3.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<25:58, 369kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<19:09, 500kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<13:37, 702kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<11:45, 810kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<09:12, 1.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<06:40, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:53, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:47, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<04:16, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:12, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:36, 2.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:27, 2.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:37, 2.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:11, 2.22MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:07, 2.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:21, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:58, 1.86MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:56, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:14, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:55, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:56, 3.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:10, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:47, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:48, 2.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:07, 2.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:49, 2.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<02:54, 3.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:06, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:43, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:45, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<02:43, 3.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<18:35, 479kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<17:01, 523kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<12:54, 690kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<09:15, 959kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<08:17, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<06:46, 1.30MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<04:58, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:23, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:43, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:30, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<03:14, 2.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<11:45, 740kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<09:10, 947kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<06:38, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<06:31, 1.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<06:33, 1.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:59, 1.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<03:36, 2.38MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:48, 1.47MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:01, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:43, 2.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:27, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:00, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:58, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<02:53, 2.92MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<12:12, 690kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<09:19, 903kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<06:42, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<04:47, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<12:39, 659kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<10:43, 778kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<07:58, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<05:39, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<11:40, 709kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<09:05, 910kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<06:34, 1.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:22, 1.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:23, 1.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:59, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:33, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:06, 1.98MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:03, 2.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:54, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:37, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:38, 2.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<02:38, 3.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:54, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:01, 1.59MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:41, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:19, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:49, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:45, 2.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:43, 2.89MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:05, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:25, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:16, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:59, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:32, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:36, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:37, 2.95MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<10:59, 703kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<08:33, 903kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<06:10, 1.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:58, 1.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:53, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:33, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<03:15, 2.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<10:22, 731kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<08:06, 935kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:51, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<05:43, 1.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<05:41, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:23, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<03:09, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<10:09, 733kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<07:54, 940kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:42, 1.30MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:35, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:33, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:17, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<03:04, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<09:56, 735kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<07:46, 940kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<05:35, 1.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:28, 1.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:27, 1.33MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:13, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<03:02, 2.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<10:47, 664kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<08:21, 857kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:59, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<05:43, 1.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:47, 1.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:30, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<04:00, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:27, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:30, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<02:32, 2.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<09:58, 697kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<07:37, 911kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:28, 1.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<03:55, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<10:25, 660kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<08:50, 779kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<06:34, 1.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:12<04:39, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<10:31, 648kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<08:07, 839kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:51, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<05:33, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<05:23, 1.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:05, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:56, 2.27MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<04:31, 1.48MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:52, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:53, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:29, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:55, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:05, 2.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:14, 2.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<11:50, 553kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<08:52, 737kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<06:22, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<05:52, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<05:32, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:12, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<03:00, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<22:19, 287kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<16:18, 392kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<11:32, 552kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<09:26, 671kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<08:01, 789kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:54, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<04:12, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<05:05, 1.23MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:15, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:08, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<03:32, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:51, 1.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:59, 2.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:15, 2.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:56, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:43, 2.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:04, 2.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:46, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:34, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<01:57, 3.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:44, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:15, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:36, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:53, 3.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<08:28, 698kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<06:35, 897kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:44, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<04:34, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<04:27, 1.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:22, 1.73MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:26, 2.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<03:45, 1.54MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:15, 1.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:25, 2.36MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:56, 1.94MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:39, 2.15MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:00, 2.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:41, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:08, 1.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:27, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<01:48, 3.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:17, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<02:55, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:11, 2.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:43, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<03:08, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:26, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<01:47, 3.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:12, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:48, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:04, 2.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:41, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<03:01, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:21, 2.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<01:45, 3.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:40, 1.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:26, 2.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<01:49, 2.90MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:26, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:54, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:16, 2.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<01:42, 3.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:25, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:16, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:42, 3.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:18, 2.21MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:44, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:09, 2.35MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:34, 3.19MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:03, 1.64MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:34, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:56, 2.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:24, 3.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<12:02, 411kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<09:32, 519kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<06:56, 712kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<04:52, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<09:18, 524kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<07:01, 694kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:59, 971kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:32, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<13:04, 368kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<09:34, 502kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<06:46, 706kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<04:47, 992kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<07:58, 595kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<06:35, 720kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<04:49, 981kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:27, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:45, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:08, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:19, 2.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:37, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:51, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:15, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:37, 2.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<06:26, 703kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:54, 922kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<03:32, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:29, 1.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:54, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:08, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:29, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:14, 1.96MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:39, 2.63MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:06, 2.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:57, 2.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:28, 2.90MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:57, 2.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:16, 1.87MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:47, 2.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:18, 3.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:55, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:29, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:49, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:12, 1.87MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:27, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:56, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<01:23, 2.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<05:17, 765kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:09, 973kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<03:00, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:57, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:57, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:17, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<01:37, 2.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<05:19, 734kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:05, 955kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:57, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<02:06, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<05:12, 737kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:27, 860kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:17, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<02:20, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<03:25, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:48, 1.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:02, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:14, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:23, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:50, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:19, 2.74MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:28, 1.47MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:07, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:33, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:51, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:06, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:37, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:11, 2.97MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:13, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:56, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:26, 2.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:45, 1.96MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:59, 1.71MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<01:35, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:08, 2.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<04:51, 691kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:46, 889kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:42, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:35, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:33, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:57, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:23, 2.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<04:24, 729kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<03:26, 934kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:28, 1.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:24, 1.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:23, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:49, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:19, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:46, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:35, 1.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:10, 2.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:28, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:21, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:00, 2.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:20, 2.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:36, 1.84MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:15, 2.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:54, 3.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:45, 1.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:32, 1.86MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:08, 2.48MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:24, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:36, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:16, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:54, 3.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:36, 755kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:46, 982kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:00, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:57, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:56, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:28, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:03, 2.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:41, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:27, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:04, 2.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:17, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:26, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:08, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<00:48, 3.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<06:28, 380kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<04:47, 512kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<03:22, 717kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:51, 836kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:31, 942kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:53, 1.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<01:19, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<03:24, 680kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:35, 890kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:52, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<01:19, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<03:06, 723kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:40, 839kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:59, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<01:23, 1.57MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<03:18, 658kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:32, 855kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:48, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:43, 1.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:25, 1.47MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:02, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:10, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:02, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:45, 2.62MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:58, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:07, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:53, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:38, 3.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:44, 693kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<02:07, 891kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:30, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<01:03, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<09:53, 185kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<07:05, 257kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<04:56, 365kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<03:47, 465kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:49, 621kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:59, 866kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:45, 964kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:24, 1.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:01, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:03, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:05, 1.49MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:50, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:35, 2.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<05:20, 291kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<03:51, 401kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:43, 564kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<01:52, 796kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<03:24, 437kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<02:41, 551kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:56, 755kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<01:20, 1.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<05:01, 283kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<03:39, 386kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<02:32, 545kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<02:01, 664kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:33, 858kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:06, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:02, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:52, 1.47MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:37, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:41, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:50, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:39, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:27, 2.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:39, 689kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:16, 892kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:54, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:51, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:50, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:37, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:26, 2.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:43, 1.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:36, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:26, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:30, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:27, 2.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:19, 2.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:25, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:28, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:22, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:15, 3.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:30, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:26, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:19, 2.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:22, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:25, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:19, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:13, 2.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:46, 854kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:36, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:25, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:24, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:21, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:14, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:16, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:18, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:14, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:09, 2.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:35, 765kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:27, 981kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:18, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:17, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:17, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:08, 2.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:24, 759kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:18, 985kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:12, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:08, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:19, 744kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:15, 950kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:09, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:08, 734kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 942kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:03, 1.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.54MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.07MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<66:50:29,  1.66it/s]  0%|          | 739/400000 [00:00<46:42:26,  2.37it/s]  0%|          | 1579/400000 [00:00<32:37:48,  3.39it/s]  1%|          | 2418/400000 [00:00<22:47:49,  4.84it/s]  1%|          | 3253/400000 [00:01<15:55:42,  6.92it/s]  1%|          | 4058/400000 [00:01<11:07:52,  9.88it/s]  1%|          | 4880/400000 [00:01<7:46:47, 14.11it/s]   1%|▏         | 5687/400000 [00:01<5:26:19, 20.14it/s]  2%|▏         | 6517/400000 [00:01<3:48:11, 28.74it/s]  2%|▏         | 7323/400000 [00:01<2:39:38, 40.99it/s]  2%|▏         | 8145/400000 [00:01<1:51:45, 58.44it/s]  2%|▏         | 8986/400000 [00:01<1:18:17, 83.24it/s]  2%|▏         | 9821/400000 [00:01<54:55, 118.40it/s]   3%|▎         | 10663/400000 [00:01<38:35, 168.13it/s]  3%|▎         | 11517/400000 [00:02<27:11, 238.18it/s]  3%|▎         | 12358/400000 [00:02<19:13, 336.17it/s]  3%|▎         | 13204/400000 [00:02<13:39, 472.20it/s]  4%|▎         | 14043/400000 [00:02<09:46, 657.97it/s]  4%|▎         | 14886/400000 [00:02<07:03, 909.53it/s]  4%|▍         | 15728/400000 [00:02<05:09, 1241.74it/s]  4%|▍         | 16563/400000 [00:02<03:50, 1667.12it/s]  4%|▍         | 17414/400000 [00:02<02:54, 2197.06it/s]  5%|▍         | 18253/400000 [00:02<02:15, 2812.64it/s]  5%|▍         | 19100/400000 [00:02<01:48, 3517.20it/s]  5%|▍         | 19936/400000 [00:03<01:29, 4236.50it/s]  5%|▌         | 20764/400000 [00:03<01:16, 4929.57it/s]  5%|▌         | 21583/400000 [00:03<01:07, 5597.97it/s]  6%|▌         | 22402/400000 [00:03<01:02, 6021.44it/s]  6%|▌         | 23199/400000 [00:03<00:57, 6496.72it/s]  6%|▌         | 24017/400000 [00:03<00:54, 6922.82it/s]  6%|▌         | 24853/400000 [00:03<00:51, 7296.50it/s]  6%|▋         | 25664/400000 [00:03<00:50, 7466.31it/s]  7%|▋         | 26508/400000 [00:03<00:48, 7732.98it/s]  7%|▋         | 27325/400000 [00:03<00:47, 7816.35it/s]  7%|▋         | 28168/400000 [00:04<00:46, 7989.90it/s]  7%|▋         | 28990/400000 [00:04<00:46, 7959.84it/s]  7%|▋         | 29807/400000 [00:04<00:46, 7969.88it/s]  8%|▊         | 30628/400000 [00:04<00:45, 8039.03it/s]  8%|▊         | 31465/400000 [00:04<00:45, 8133.55it/s]  8%|▊         | 32306/400000 [00:04<00:44, 8213.28it/s]  8%|▊         | 33153/400000 [00:04<00:44, 8288.37it/s]  8%|▊         | 33994/400000 [00:04<00:43, 8322.79it/s]  9%|▊         | 34842/400000 [00:04<00:43, 8369.22it/s]  9%|▉         | 35681/400000 [00:04<00:43, 8337.49it/s]  9%|▉         | 36516/400000 [00:05<00:43, 8314.57it/s]  9%|▉         | 37362/400000 [00:05<00:43, 8355.34it/s] 10%|▉         | 38203/400000 [00:05<00:43, 8370.54it/s] 10%|▉         | 39051/400000 [00:05<00:42, 8400.83it/s] 10%|▉         | 39892/400000 [00:05<00:42, 8385.77it/s] 10%|█         | 40736/400000 [00:05<00:42, 8400.34it/s] 10%|█         | 41577/400000 [00:05<00:42, 8397.21it/s] 11%|█         | 42421/400000 [00:05<00:42, 8407.47it/s] 11%|█         | 43270/400000 [00:05<00:42, 8431.43it/s] 11%|█         | 44118/400000 [00:05<00:42, 8444.78it/s] 11%|█         | 44963/400000 [00:06<00:42, 8436.73it/s] 11%|█▏        | 45807/400000 [00:06<00:42, 8423.75it/s] 12%|█▏        | 46650/400000 [00:06<00:42, 8358.16it/s] 12%|█▏        | 47497/400000 [00:06<00:42, 8389.18it/s] 12%|█▏        | 48337/400000 [00:06<00:42, 8298.65it/s] 12%|█▏        | 49176/400000 [00:06<00:42, 8325.68it/s] 13%|█▎        | 50009/400000 [00:06<00:43, 7993.29it/s] 13%|█▎        | 50851/400000 [00:06<00:43, 8115.16it/s] 13%|█▎        | 51691/400000 [00:06<00:42, 8196.57it/s] 13%|█▎        | 52513/400000 [00:06<00:42, 8188.17it/s] 13%|█▎        | 53334/400000 [00:07<00:42, 8140.22it/s] 14%|█▎        | 54158/400000 [00:07<00:42, 8167.78it/s] 14%|█▍        | 55000/400000 [00:07<00:41, 8241.22it/s] 14%|█▍        | 55848/400000 [00:07<00:41, 8309.45it/s] 14%|█▍        | 56694/400000 [00:07<00:41, 8351.26it/s] 14%|█▍        | 57530/400000 [00:07<00:41, 8350.50it/s] 15%|█▍        | 58366/400000 [00:07<00:41, 8247.01it/s] 15%|█▍        | 59193/400000 [00:07<00:41, 8251.50it/s] 15%|█▌        | 60039/400000 [00:07<00:40, 8312.80it/s] 15%|█▌        | 60887/400000 [00:07<00:40, 8361.55it/s] 15%|█▌        | 61732/400000 [00:08<00:40, 8386.27it/s] 16%|█▌        | 62574/400000 [00:08<00:40, 8394.19it/s] 16%|█▌        | 63414/400000 [00:08<00:40, 8318.40it/s] 16%|█▌        | 64247/400000 [00:08<00:41, 8154.64it/s] 16%|█▋        | 65090/400000 [00:08<00:40, 8235.19it/s] 16%|█▋        | 65935/400000 [00:08<00:40, 8296.03it/s] 17%|█▋        | 66774/400000 [00:08<00:40, 8321.92it/s] 17%|█▋        | 67621/400000 [00:08<00:39, 8362.97it/s] 17%|█▋        | 68458/400000 [00:08<00:40, 8283.29it/s] 17%|█▋        | 69305/400000 [00:08<00:39, 8337.23it/s] 18%|█▊        | 70149/400000 [00:09<00:39, 8364.93it/s] 18%|█▊        | 70986/400000 [00:09<00:39, 8299.30it/s] 18%|█▊        | 71817/400000 [00:09<00:39, 8301.16it/s] 18%|█▊        | 72655/400000 [00:09<00:39, 8321.98it/s] 18%|█▊        | 73503/400000 [00:09<00:39, 8365.93it/s] 19%|█▊        | 74351/400000 [00:09<00:38, 8397.09it/s] 19%|█▉        | 75191/400000 [00:09<00:38, 8396.88it/s] 19%|█▉        | 76031/400000 [00:09<00:39, 8215.51it/s] 19%|█▉        | 76854/400000 [00:09<00:39, 8081.67it/s] 19%|█▉        | 77664/400000 [00:10<00:39, 8059.10it/s] 20%|█▉        | 78512/400000 [00:10<00:39, 8179.74it/s] 20%|█▉        | 79347/400000 [00:10<00:38, 8227.42it/s] 20%|██        | 80187/400000 [00:10<00:38, 8275.88it/s] 20%|██        | 81016/400000 [00:10<00:38, 8213.72it/s] 20%|██        | 81857/400000 [00:10<00:38, 8270.27it/s] 21%|██        | 82701/400000 [00:10<00:38, 8318.01it/s] 21%|██        | 83534/400000 [00:10<00:38, 8272.64it/s] 21%|██        | 84380/400000 [00:10<00:37, 8326.70it/s] 21%|██▏       | 85223/400000 [00:10<00:37, 8354.77it/s] 22%|██▏       | 86059/400000 [00:11<00:38, 8232.38it/s] 22%|██▏       | 86897/400000 [00:11<00:37, 8275.23it/s] 22%|██▏       | 87725/400000 [00:11<00:38, 8120.73it/s] 22%|██▏       | 88542/400000 [00:11<00:38, 8132.82it/s] 22%|██▏       | 89371/400000 [00:11<00:37, 8177.59it/s] 23%|██▎       | 90211/400000 [00:11<00:37, 8240.29it/s] 23%|██▎       | 91054/400000 [00:11<00:37, 8296.04it/s] 23%|██▎       | 91893/400000 [00:11<00:37, 8322.89it/s] 23%|██▎       | 92735/400000 [00:11<00:36, 8350.83it/s] 23%|██▎       | 93578/400000 [00:11<00:36, 8372.17it/s] 24%|██▎       | 94416/400000 [00:12<00:36, 8360.98it/s] 24%|██▍       | 95261/400000 [00:12<00:36, 8384.69it/s] 24%|██▍       | 96100/400000 [00:12<00:36, 8361.08it/s] 24%|██▍       | 96937/400000 [00:12<00:36, 8342.24it/s] 24%|██▍       | 97785/400000 [00:12<00:36, 8381.64it/s] 25%|██▍       | 98632/400000 [00:12<00:35, 8407.17it/s] 25%|██▍       | 99473/400000 [00:12<00:35, 8401.51it/s] 25%|██▌       | 100321/400000 [00:12<00:35, 8423.54it/s] 25%|██▌       | 101164/400000 [00:12<00:35, 8414.27it/s] 26%|██▌       | 102006/400000 [00:12<00:35, 8407.84it/s] 26%|██▌       | 102854/400000 [00:13<00:35, 8429.13it/s] 26%|██▌       | 103697/400000 [00:13<00:35, 8417.56it/s] 26%|██▌       | 104539/400000 [00:13<00:35, 8390.25it/s] 26%|██▋       | 105379/400000 [00:13<00:36, 8142.83it/s] 27%|██▋       | 106220/400000 [00:13<00:35, 8220.86it/s] 27%|██▋       | 107044/400000 [00:13<00:35, 8152.72it/s] 27%|██▋       | 107872/400000 [00:13<00:35, 8188.36it/s] 27%|██▋       | 108715/400000 [00:13<00:35, 8256.60it/s] 27%|██▋       | 109556/400000 [00:13<00:34, 8301.18it/s] 28%|██▊       | 110391/400000 [00:13<00:34, 8313.58it/s] 28%|██▊       | 111227/400000 [00:14<00:34, 8326.99it/s] 28%|██▊       | 112073/400000 [00:14<00:34, 8365.08it/s] 28%|██▊       | 112914/400000 [00:14<00:34, 8375.69it/s] 28%|██▊       | 113752/400000 [00:14<00:34, 8373.56it/s] 29%|██▊       | 114590/400000 [00:14<00:34, 8364.66it/s] 29%|██▉       | 115439/400000 [00:14<00:33, 8401.01it/s] 29%|██▉       | 116290/400000 [00:14<00:33, 8431.90it/s] 29%|██▉       | 117143/400000 [00:14<00:33, 8458.80it/s] 29%|██▉       | 117989/400000 [00:14<00:33, 8439.60it/s] 30%|██▉       | 118834/400000 [00:14<00:33, 8366.36it/s] 30%|██▉       | 119676/400000 [00:15<00:33, 8382.12it/s] 30%|███       | 120521/400000 [00:15<00:33, 8401.56it/s] 30%|███       | 121365/400000 [00:15<00:33, 8412.96it/s] 31%|███       | 122207/400000 [00:15<00:33, 8276.97it/s] 31%|███       | 123046/400000 [00:15<00:33, 8308.64it/s] 31%|███       | 123893/400000 [00:15<00:33, 8354.64it/s] 31%|███       | 124742/400000 [00:15<00:32, 8394.05it/s] 31%|███▏      | 125582/400000 [00:15<00:33, 8287.04it/s] 32%|███▏      | 126421/400000 [00:15<00:32, 8316.69it/s] 32%|███▏      | 127254/400000 [00:15<00:33, 8209.83it/s] 32%|███▏      | 128094/400000 [00:16<00:32, 8264.14it/s] 32%|███▏      | 128921/400000 [00:16<00:33, 8151.98it/s] 32%|███▏      | 129737/400000 [00:16<00:33, 8076.23it/s] 33%|███▎      | 130575/400000 [00:16<00:32, 8164.78it/s] 33%|███▎      | 131393/400000 [00:16<00:32, 8168.68it/s] 33%|███▎      | 132234/400000 [00:16<00:32, 8239.11it/s] 33%|███▎      | 133078/400000 [00:16<00:32, 8295.95it/s] 33%|███▎      | 133928/400000 [00:16<00:31, 8353.32it/s] 34%|███▎      | 134777/400000 [00:16<00:31, 8392.99it/s] 34%|███▍      | 135617/400000 [00:16<00:31, 8368.24it/s] 34%|███▍      | 136464/400000 [00:17<00:31, 8398.43it/s] 34%|███▍      | 137314/400000 [00:17<00:31, 8426.32it/s] 35%|███▍      | 138161/400000 [00:17<00:31, 8439.08it/s] 35%|███▍      | 139006/400000 [00:17<00:31, 8265.40it/s] 35%|███▍      | 139837/400000 [00:17<00:31, 8275.84it/s] 35%|███▌      | 140666/400000 [00:17<00:31, 8228.36it/s] 35%|███▌      | 141503/400000 [00:17<00:31, 8267.78it/s] 36%|███▌      | 142346/400000 [00:17<00:30, 8313.32it/s] 36%|███▌      | 143178/400000 [00:17<00:30, 8310.33it/s] 36%|███▌      | 144010/400000 [00:17<00:30, 8288.31it/s] 36%|███▌      | 144844/400000 [00:18<00:30, 8301.36it/s] 36%|███▋      | 145682/400000 [00:18<00:30, 8324.16it/s] 37%|███▋      | 146515/400000 [00:18<00:30, 8313.66it/s] 37%|███▋      | 147357/400000 [00:18<00:30, 8344.91it/s] 37%|███▋      | 148193/400000 [00:18<00:30, 8349.33it/s] 37%|███▋      | 149039/400000 [00:18<00:29, 8381.97it/s] 37%|███▋      | 149878/400000 [00:18<00:29, 8341.85it/s] 38%|███▊      | 150713/400000 [00:18<00:30, 8302.14it/s] 38%|███▊      | 151548/400000 [00:18<00:29, 8308.82it/s] 38%|███▊      | 152379/400000 [00:18<00:30, 8247.83it/s] 38%|███▊      | 153204/400000 [00:19<00:30, 8213.22it/s] 39%|███▊      | 154036/400000 [00:19<00:29, 8243.14it/s] 39%|███▊      | 154877/400000 [00:19<00:29, 8290.90it/s] 39%|███▉      | 155726/400000 [00:19<00:29, 8346.72it/s] 39%|███▉      | 156561/400000 [00:19<00:29, 8341.60it/s] 39%|███▉      | 157412/400000 [00:19<00:28, 8389.05it/s] 40%|███▉      | 158262/400000 [00:19<00:28, 8421.06it/s] 40%|███▉      | 159105/400000 [00:19<00:28, 8390.12it/s] 40%|███▉      | 159952/400000 [00:19<00:28, 8413.41it/s] 40%|████      | 160796/400000 [00:19<00:28, 8420.20it/s] 40%|████      | 161640/400000 [00:20<00:28, 8423.97it/s] 41%|████      | 162484/400000 [00:20<00:28, 8426.15it/s] 41%|████      | 163331/400000 [00:20<00:28, 8438.75it/s] 41%|████      | 164175/400000 [00:20<00:28, 8275.60it/s] 41%|████▏     | 165004/400000 [00:20<00:28, 8248.54it/s] 41%|████▏     | 165844/400000 [00:20<00:28, 8292.98it/s] 42%|████▏     | 166684/400000 [00:20<00:28, 8323.47it/s] 42%|████▏     | 167525/400000 [00:20<00:27, 8347.07it/s] 42%|████▏     | 168373/400000 [00:20<00:27, 8385.12it/s] 42%|████▏     | 169212/400000 [00:21<00:28, 8126.31it/s] 43%|████▎     | 170057/400000 [00:21<00:27, 8220.00it/s] 43%|████▎     | 170904/400000 [00:21<00:27, 8293.25it/s] 43%|████▎     | 171756/400000 [00:21<00:27, 8358.70it/s] 43%|████▎     | 172593/400000 [00:21<00:27, 8283.67it/s] 43%|████▎     | 173432/400000 [00:21<00:27, 8314.74it/s] 44%|████▎     | 174268/400000 [00:21<00:27, 8326.80it/s] 44%|████▍     | 175115/400000 [00:21<00:26, 8366.81it/s] 44%|████▍     | 175962/400000 [00:21<00:26, 8394.72it/s] 44%|████▍     | 176806/400000 [00:21<00:26, 8407.34it/s] 44%|████▍     | 177647/400000 [00:22<00:26, 8329.04it/s] 45%|████▍     | 178481/400000 [00:22<00:26, 8217.34it/s] 45%|████▍     | 179315/400000 [00:22<00:26, 8252.96it/s] 45%|████▌     | 180141/400000 [00:22<00:26, 8153.81it/s] 45%|████▌     | 180975/400000 [00:22<00:26, 8207.41it/s] 45%|████▌     | 181797/400000 [00:22<00:26, 8098.40it/s] 46%|████▌     | 182624/400000 [00:22<00:26, 8147.79it/s] 46%|████▌     | 183472/400000 [00:22<00:26, 8243.97it/s] 46%|████▌     | 184314/400000 [00:22<00:26, 8293.53it/s] 46%|████▋     | 185148/400000 [00:22<00:25, 8305.07it/s] 46%|████▋     | 185979/400000 [00:23<00:25, 8269.65it/s] 47%|████▋     | 186819/400000 [00:23<00:25, 8306.24it/s] 47%|████▋     | 187671/400000 [00:23<00:25, 8368.38it/s] 47%|████▋     | 188521/400000 [00:23<00:25, 8405.45it/s] 47%|████▋     | 189368/400000 [00:23<00:25, 8422.11it/s] 48%|████▊     | 190216/400000 [00:23<00:24, 8438.20it/s] 48%|████▊     | 191060/400000 [00:23<00:24, 8422.09it/s] 48%|████▊     | 191903/400000 [00:23<00:24, 8419.18it/s] 48%|████▊     | 192745/400000 [00:23<00:24, 8407.68it/s] 48%|████▊     | 193586/400000 [00:23<00:24, 8289.45it/s] 49%|████▊     | 194416/400000 [00:24<00:25, 8122.80it/s] 49%|████▉     | 195255/400000 [00:24<00:24, 8199.28it/s] 49%|████▉     | 196105/400000 [00:24<00:24, 8285.60it/s] 49%|████▉     | 196954/400000 [00:24<00:24, 8344.90it/s] 49%|████▉     | 197802/400000 [00:24<00:24, 8384.79it/s] 50%|████▉     | 198642/400000 [00:24<00:24, 8385.07it/s] 50%|████▉     | 199487/400000 [00:24<00:23, 8402.81it/s] 50%|█████     | 200328/400000 [00:24<00:23, 8365.04it/s] 50%|█████     | 201174/400000 [00:24<00:23, 8392.99it/s] 51%|█████     | 202014/400000 [00:24<00:23, 8327.31it/s] 51%|█████     | 202847/400000 [00:25<00:23, 8297.21it/s] 51%|█████     | 203677/400000 [00:25<00:23, 8293.76it/s] 51%|█████     | 204519/400000 [00:25<00:23, 8329.33it/s] 51%|█████▏    | 205356/400000 [00:25<00:23, 8340.65it/s] 52%|█████▏    | 206191/400000 [00:25<00:23, 8304.25it/s] 52%|█████▏    | 207022/400000 [00:25<00:23, 8192.18it/s] 52%|█████▏    | 207861/400000 [00:25<00:23, 8249.18it/s] 52%|█████▏    | 208687/400000 [00:25<00:23, 8208.99it/s] 52%|█████▏    | 209532/400000 [00:25<00:23, 8278.87it/s] 53%|█████▎    | 210379/400000 [00:25<00:22, 8332.93it/s] 53%|█████▎    | 211222/400000 [00:26<00:22, 8361.34it/s] 53%|█████▎    | 212063/400000 [00:26<00:22, 8374.35it/s] 53%|█████▎    | 212901/400000 [00:26<00:23, 8088.19it/s] 53%|█████▎    | 213730/400000 [00:26<00:22, 8146.47it/s] 54%|█████▎    | 214565/400000 [00:26<00:22, 8203.85it/s] 54%|█████▍    | 215409/400000 [00:26<00:22, 8271.12it/s] 54%|█████▍    | 216247/400000 [00:26<00:22, 8300.66it/s] 54%|█████▍    | 217094/400000 [00:26<00:21, 8350.12it/s] 54%|█████▍    | 217930/400000 [00:26<00:21, 8297.93it/s] 55%|█████▍    | 218761/400000 [00:26<00:22, 8205.93it/s] 55%|█████▍    | 219603/400000 [00:27<00:21, 8266.83it/s] 55%|█████▌    | 220436/400000 [00:27<00:21, 8285.45it/s] 55%|█████▌    | 221279/400000 [00:27<00:21, 8327.92it/s] 56%|█████▌    | 222127/400000 [00:27<00:21, 8370.99it/s] 56%|█████▌    | 222975/400000 [00:27<00:21, 8401.70it/s] 56%|█████▌    | 223816/400000 [00:27<00:21, 8227.47it/s] 56%|█████▌    | 224649/400000 [00:27<00:21, 8255.88it/s] 56%|█████▋    | 225479/400000 [00:27<00:21, 8268.31it/s] 57%|█████▋    | 226319/400000 [00:27<00:20, 8305.07it/s] 57%|█████▋    | 227150/400000 [00:27<00:20, 8290.85it/s] 57%|█████▋    | 227980/400000 [00:28<00:21, 8183.12it/s] 57%|█████▋    | 228820/400000 [00:28<00:20, 8246.86it/s] 57%|█████▋    | 229648/400000 [00:28<00:20, 8254.31it/s] 58%|█████▊    | 230492/400000 [00:28<00:20, 8306.71it/s] 58%|█████▊    | 231323/400000 [00:28<00:20, 8180.81it/s] 58%|█████▊    | 232142/400000 [00:28<00:20, 8040.41it/s] 58%|█████▊    | 232988/400000 [00:28<00:20, 8160.21it/s] 58%|█████▊    | 233834/400000 [00:28<00:20, 8246.76it/s] 59%|█████▊    | 234676/400000 [00:28<00:19, 8296.29it/s] 59%|█████▉    | 235507/400000 [00:29<00:19, 8273.49it/s] 59%|█████▉    | 236348/400000 [00:29<00:19, 8312.29it/s] 59%|█████▉    | 237185/400000 [00:29<00:19, 8328.92it/s] 60%|█████▉    | 238022/400000 [00:29<00:19, 8340.79it/s] 60%|█████▉    | 238857/400000 [00:29<00:19, 8294.23it/s] 60%|█████▉    | 239699/400000 [00:29<00:19, 8330.03it/s] 60%|██████    | 240533/400000 [00:29<00:19, 8310.54it/s] 60%|██████    | 241369/400000 [00:29<00:19, 8323.41it/s] 61%|██████    | 242202/400000 [00:29<00:18, 8322.30it/s] 61%|██████    | 243046/400000 [00:29<00:18, 8355.56it/s] 61%|██████    | 243891/400000 [00:30<00:18, 8382.64it/s] 61%|██████    | 244738/400000 [00:30<00:18, 8407.53it/s] 61%|██████▏   | 245583/400000 [00:30<00:18, 8419.32it/s] 62%|██████▏   | 246425/400000 [00:30<00:18, 8382.99it/s] 62%|██████▏   | 247270/400000 [00:30<00:18, 8402.22it/s] 62%|██████▏   | 248116/400000 [00:30<00:18, 8416.59it/s] 62%|██████▏   | 248958/400000 [00:30<00:18, 8390.34it/s] 62%|██████▏   | 249798/400000 [00:30<00:18, 8331.37it/s] 63%|██████▎   | 250636/400000 [00:30<00:17, 8344.19it/s] 63%|██████▎   | 251483/400000 [00:30<00:17, 8381.23it/s] 63%|██████▎   | 252322/400000 [00:31<00:18, 8162.46it/s] 63%|██████▎   | 253156/400000 [00:31<00:17, 8214.66it/s] 63%|██████▎   | 253979/400000 [00:31<00:18, 8089.41it/s] 64%|██████▎   | 254813/400000 [00:31<00:17, 8161.83it/s] 64%|██████▍   | 255661/400000 [00:31<00:17, 8249.81it/s] 64%|██████▍   | 256487/400000 [00:31<00:17, 8084.56it/s] 64%|██████▍   | 257304/400000 [00:31<00:17, 8109.80it/s] 65%|██████▍   | 258145/400000 [00:31<00:17, 8197.34it/s] 65%|██████▍   | 258966/400000 [00:31<00:17, 7849.65it/s] 65%|██████▍   | 259755/400000 [00:31<00:17, 7801.98it/s] 65%|██████▌   | 260572/400000 [00:32<00:17, 7907.71it/s] 65%|██████▌   | 261366/400000 [00:32<00:17, 7906.93it/s] 66%|██████▌   | 262189/400000 [00:32<00:17, 7999.98it/s] 66%|██████▌   | 263025/400000 [00:32<00:16, 8103.12it/s] 66%|██████▌   | 263837/400000 [00:32<00:16, 8087.38it/s] 66%|██████▌   | 264680/400000 [00:32<00:16, 8186.84it/s] 66%|██████▋   | 265502/400000 [00:32<00:16, 8194.57it/s] 67%|██████▋   | 266352/400000 [00:32<00:16, 8281.64it/s] 67%|██████▋   | 267186/400000 [00:32<00:16, 8296.59it/s] 67%|██████▋   | 268034/400000 [00:32<00:15, 8348.90it/s] 67%|██████▋   | 268882/400000 [00:33<00:15, 8386.21it/s] 67%|██████▋   | 269721/400000 [00:33<00:15, 8346.55it/s] 68%|██████▊   | 270568/400000 [00:33<00:15, 8380.52it/s] 68%|██████▊   | 271415/400000 [00:33<00:15, 8406.66it/s] 68%|██████▊   | 272264/400000 [00:33<00:15, 8429.57it/s] 68%|██████▊   | 273112/400000 [00:33<00:15, 8442.88it/s] 68%|██████▊   | 273957/400000 [00:33<00:14, 8422.20it/s] 69%|██████▊   | 274800/400000 [00:33<00:14, 8422.15it/s] 69%|██████▉   | 275648/400000 [00:33<00:14, 8437.10it/s] 69%|██████▉   | 276497/400000 [00:33<00:14, 8451.70it/s] 69%|██████▉   | 277343/400000 [00:34<00:14, 8452.92it/s] 70%|██████▉   | 278189/400000 [00:34<00:14, 8409.91it/s] 70%|██████▉   | 279031/400000 [00:34<00:14, 8405.94it/s] 70%|██████▉   | 279878/400000 [00:34<00:14, 8423.07it/s] 70%|███████   | 280721/400000 [00:34<00:14, 8410.32it/s] 70%|███████   | 281563/400000 [00:34<00:14, 8406.25it/s] 71%|███████   | 282408/400000 [00:34<00:13, 8419.21it/s] 71%|███████   | 283250/400000 [00:34<00:13, 8400.68it/s] 71%|███████   | 284099/400000 [00:34<00:13, 8424.55it/s] 71%|███████   | 284942/400000 [00:34<00:14, 8149.07it/s] 71%|███████▏  | 285759/400000 [00:35<00:14, 7864.57it/s] 72%|███████▏  | 286549/400000 [00:35<00:15, 7512.47it/s] 72%|███████▏  | 287307/400000 [00:35<00:15, 7490.72it/s] 72%|███████▏  | 288069/400000 [00:35<00:14, 7528.68it/s] 72%|███████▏  | 288890/400000 [00:35<00:14, 7720.45it/s] 72%|███████▏  | 289729/400000 [00:35<00:13, 7907.89it/s] 73%|███████▎  | 290577/400000 [00:35<00:13, 8070.14it/s] 73%|███████▎  | 291396/400000 [00:35<00:13, 8103.10it/s] 73%|███████▎  | 292228/400000 [00:35<00:13, 8165.94it/s] 73%|███████▎  | 293047/400000 [00:36<00:13, 8009.33it/s] 73%|███████▎  | 293850/400000 [00:36<00:13, 7701.71it/s] 74%|███████▎  | 294625/400000 [00:36<00:14, 7509.07it/s] 74%|███████▍  | 295471/400000 [00:36<00:13, 7768.61it/s] 74%|███████▍  | 296308/400000 [00:36<00:13, 7938.85it/s] 74%|███████▍  | 297143/400000 [00:36<00:12, 8056.95it/s] 74%|███████▍  | 297982/400000 [00:36<00:12, 8151.61it/s] 75%|███████▍  | 298819/400000 [00:36<00:12, 8213.71it/s] 75%|███████▍  | 299660/400000 [00:36<00:12, 8269.23it/s] 75%|███████▌  | 300496/400000 [00:36<00:11, 8296.06it/s] 75%|███████▌  | 301334/400000 [00:37<00:11, 8320.03it/s] 76%|███████▌  | 302178/400000 [00:37<00:11, 8355.29it/s] 76%|███████▌  | 303022/400000 [00:37<00:11, 8379.14it/s] 76%|███████▌  | 303867/400000 [00:37<00:11, 8398.40it/s] 76%|███████▌  | 304708/400000 [00:37<00:11, 8396.94it/s] 76%|███████▋  | 305548/400000 [00:37<00:11, 8285.47it/s] 77%|███████▋  | 306378/400000 [00:37<00:11, 8285.60it/s] 77%|███████▋  | 307218/400000 [00:37<00:11, 8318.72it/s] 77%|███████▋  | 308055/400000 [00:37<00:11, 8332.92it/s] 77%|███████▋  | 308890/400000 [00:37<00:10, 8336.01it/s] 77%|███████▋  | 309736/400000 [00:38<00:10, 8372.59it/s] 78%|███████▊  | 310574/400000 [00:38<00:10, 8358.04it/s] 78%|███████▊  | 311410/400000 [00:38<00:10, 8298.99it/s] 78%|███████▊  | 312244/400000 [00:38<00:10, 8311.23it/s] 78%|███████▊  | 313091/400000 [00:38<00:10, 8356.72it/s] 78%|███████▊  | 313937/400000 [00:38<00:10, 8384.80it/s] 79%|███████▊  | 314776/400000 [00:38<00:10, 8347.18it/s] 79%|███████▉  | 315611/400000 [00:38<00:10, 8340.45it/s] 79%|███████▉  | 316446/400000 [00:38<00:10, 8316.96it/s] 79%|███████▉  | 317285/400000 [00:38<00:09, 8338.39it/s] 80%|███████▉  | 318119/400000 [00:39<00:09, 8333.38it/s] 80%|███████▉  | 318959/400000 [00:39<00:09, 8350.52it/s] 80%|███████▉  | 319802/400000 [00:39<00:09, 8373.70it/s] 80%|████████  | 320643/400000 [00:39<00:09, 8382.92it/s] 80%|████████  | 321483/400000 [00:39<00:09, 8386.22it/s] 81%|████████  | 322322/400000 [00:39<00:09, 8351.33it/s] 81%|████████  | 323158/400000 [00:39<00:09, 8318.83it/s] 81%|████████  | 323990/400000 [00:39<00:09, 8012.91it/s] 81%|████████  | 324836/400000 [00:39<00:09, 8140.26it/s] 81%|████████▏ | 325653/400000 [00:39<00:09, 8116.08it/s] 82%|████████▏ | 326492/400000 [00:40<00:08, 8196.22it/s] 82%|████████▏ | 327328/400000 [00:40<00:08, 8244.67it/s] 82%|████████▏ | 328169/400000 [00:40<00:08, 8290.98it/s] 82%|████████▏ | 329007/400000 [00:40<00:08, 8315.37it/s] 82%|████████▏ | 329840/400000 [00:40<00:08, 8127.35it/s] 83%|████████▎ | 330655/400000 [00:40<00:08, 8093.51it/s] 83%|████████▎ | 331493/400000 [00:40<00:08, 8175.06it/s] 83%|████████▎ | 332324/400000 [00:40<00:08, 8213.17it/s] 83%|████████▎ | 333156/400000 [00:40<00:08, 8243.64it/s] 84%|████████▎ | 334005/400000 [00:40<00:07, 8315.34it/s] 84%|████████▎ | 334854/400000 [00:41<00:07, 8366.79it/s] 84%|████████▍ | 335692/400000 [00:41<00:07, 8186.65it/s] 84%|████████▍ | 336515/400000 [00:41<00:07, 8197.30it/s] 84%|████████▍ | 337349/400000 [00:41<00:07, 8238.37it/s] 85%|████████▍ | 338195/400000 [00:41<00:07, 8303.13it/s] 85%|████████▍ | 339037/400000 [00:41<00:07, 8335.10it/s] 85%|████████▍ | 339876/400000 [00:41<00:07, 8349.39it/s] 85%|████████▌ | 340722/400000 [00:41<00:07, 8381.81it/s] 85%|████████▌ | 341567/400000 [00:41<00:06, 8401.65it/s] 86%|████████▌ | 342408/400000 [00:41<00:07, 8210.28it/s] 86%|████████▌ | 343255/400000 [00:42<00:06, 8284.02it/s] 86%|████████▌ | 344101/400000 [00:42<00:06, 8334.03it/s] 86%|████████▌ | 344936/400000 [00:42<00:06, 8145.13it/s] 86%|████████▋ | 345776/400000 [00:42<00:06, 8219.83it/s] 87%|████████▋ | 346616/400000 [00:42<00:06, 8273.02it/s] 87%|████████▋ | 347445/400000 [00:42<00:06, 7993.18it/s] 87%|████████▋ | 348247/400000 [00:42<00:06, 7900.59it/s] 87%|████████▋ | 349040/400000 [00:42<00:06, 7754.38it/s] 87%|████████▋ | 349818/400000 [00:42<00:06, 7697.03it/s] 88%|████████▊ | 350624/400000 [00:43<00:06, 7799.83it/s] 88%|████████▊ | 351406/400000 [00:43<00:06, 7777.04it/s] 88%|████████▊ | 352199/400000 [00:43<00:06, 7821.57it/s] 88%|████████▊ | 352982/400000 [00:43<00:06, 7822.20it/s] 88%|████████▊ | 353778/400000 [00:43<00:05, 7862.50it/s] 89%|████████▊ | 354600/400000 [00:43<00:05, 7964.92it/s] 89%|████████▉ | 355407/400000 [00:43<00:05, 7994.41it/s] 89%|████████▉ | 356256/400000 [00:43<00:05, 8136.49it/s] 89%|████████▉ | 357096/400000 [00:43<00:05, 8212.59it/s] 89%|████████▉ | 357939/400000 [00:43<00:05, 8275.22it/s] 90%|████████▉ | 358768/400000 [00:44<00:05, 8228.62it/s] 90%|████████▉ | 359605/400000 [00:44<00:04, 8269.61it/s] 90%|█████████ | 360440/400000 [00:44<00:04, 8292.44it/s] 90%|█████████ | 361276/400000 [00:44<00:04, 8311.08it/s] 91%|█████████ | 362108/400000 [00:44<00:04, 8298.18it/s] 91%|█████████ | 362952/400000 [00:44<00:04, 8339.66it/s] 91%|█████████ | 363799/400000 [00:44<00:04, 8375.61it/s] 91%|█████████ | 364643/400000 [00:44<00:04, 8393.78it/s] 91%|█████████▏| 365483/400000 [00:44<00:04, 8365.15it/s] 92%|█████████▏| 366320/400000 [00:44<00:04, 8286.31it/s] 92%|█████████▏| 367158/400000 [00:45<00:03, 8312.68it/s] 92%|█████████▏| 367996/400000 [00:45<00:03, 8330.03it/s] 92%|█████████▏| 368830/400000 [00:45<00:03, 8315.10it/s] 92%|█████████▏| 369662/400000 [00:45<00:03, 8314.95it/s] 93%|█████████▎| 370497/400000 [00:45<00:03, 8322.77it/s] 93%|█████████▎| 371333/400000 [00:45<00:03, 8331.94it/s] 93%|█████████▎| 372167/400000 [00:45<00:03, 8300.14it/s] 93%|█████████▎| 373011/400000 [00:45<00:03, 8339.52it/s] 93%|█████████▎| 373846/400000 [00:45<00:03, 8314.32it/s] 94%|█████████▎| 374678/400000 [00:45<00:03, 8287.07it/s] 94%|█████████▍| 375525/400000 [00:46<00:02, 8339.68it/s] 94%|█████████▍| 376360/400000 [00:46<00:02, 8338.84it/s] 94%|█████████▍| 377204/400000 [00:46<00:02, 8366.60it/s] 95%|█████████▍| 378049/400000 [00:46<00:02, 8390.94it/s] 95%|█████████▍| 378889/400000 [00:46<00:02, 8368.42it/s] 95%|█████████▍| 379735/400000 [00:46<00:02, 8395.20it/s] 95%|█████████▌| 380578/400000 [00:46<00:02, 8404.84it/s] 95%|█████████▌| 381426/400000 [00:46<00:02, 8426.23it/s] 96%|█████████▌| 382272/400000 [00:46<00:02, 8434.30it/s] 96%|█████████▌| 383116/400000 [00:46<00:02, 8422.37it/s] 96%|█████████▌| 383959/400000 [00:47<00:01, 8376.21it/s] 96%|█████████▌| 384797/400000 [00:47<00:01, 8359.26it/s] 96%|█████████▋| 385633/400000 [00:47<00:01, 8275.54it/s] 97%|█████████▋| 386479/400000 [00:47<00:01, 8327.78it/s] 97%|█████████▋| 387319/400000 [00:47<00:01, 8348.13it/s] 97%|█████████▋| 388157/400000 [00:47<00:01, 8356.94it/s] 97%|█████████▋| 388998/400000 [00:47<00:01, 8371.16it/s] 97%|█████████▋| 389844/400000 [00:47<00:01, 8395.73it/s] 98%|█████████▊| 390691/400000 [00:47<00:01, 8417.39it/s] 98%|█████████▊| 391533/400000 [00:47<00:01, 8393.60it/s] 98%|█████████▊| 392378/400000 [00:48<00:00, 8409.13it/s] 98%|█████████▊| 393222/400000 [00:48<00:00, 8417.77it/s] 99%|█████████▊| 394064/400000 [00:48<00:00, 8400.74it/s] 99%|█████████▊| 394905/400000 [00:48<00:00, 8401.06it/s] 99%|█████████▉| 395746/400000 [00:48<00:00, 8400.74it/s] 99%|█████████▉| 396591/400000 [00:48<00:00, 8413.30it/s] 99%|█████████▉| 397433/400000 [00:48<00:00, 8227.62it/s]100%|█████████▉| 398257/400000 [00:48<00:00, 8221.34it/s]100%|█████████▉| 399093/400000 [00:48<00:00, 8253.91it/s]100%|█████████▉| 399919/400000 [00:48<00:00, 8048.49it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8167.49it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f21db668e10>, <torchtext.data.dataset.TabularDataset object at 0x7f21db668f60>, <torchtext.vocab.Vocab object at 0x7f21db668e80>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.25 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.27 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.27 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.30 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.40 file/s]2020-07-24 12:18:28.007335: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-24 12:18:28.011782: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095139999 Hz
2020-07-24 12:18:28.011952: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555a8540dda0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 12:18:28.011968: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3629056/9912422 [00:00<00:00, 35682320.07it/s]9920512it [00:00, 34533432.12it/s]                             
0it [00:00, ?it/s]32768it [00:00, 680067.66it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 464568.77it/s]1654784it [00:00, 11687458.16it/s]                         
0it [00:00, ?it/s]8192it [00:00, 227153.80it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
