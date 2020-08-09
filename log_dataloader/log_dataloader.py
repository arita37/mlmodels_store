
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f10f46af598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f10f46af598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f115f35f488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f115f35f488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f117e57dea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f117e57dea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f110c69a158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f110c69a158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f110c69a158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 132231.92it/s] 86%|████████▌ | 8511488/9912422 [00:00<00:07, 188776.64it/s]9920512it [00:00, 41557073.47it/s]                           
0it [00:00, ?it/s]32768it [00:00, 586979.72it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158897.78it/s]1654784it [00:00, 11191140.74it/s]                         
0it [00:00, ?it/s]8192it [00:00, 265689.31it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f10f38ea6d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f110aa82978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f110c690d90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f110c690d90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f110c690d90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:44,  3.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:44,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:44,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:43,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:43,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:31,  4.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:31,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:31,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:31,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:31,  4.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:23,  6.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:23,  6.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:22,  6.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:22,  6.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:17,  8.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:17,  8.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:17,  8.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:17,  8.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:13, 10.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:13, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:13, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:13, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:10, 13.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:10, 13.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:10, 13.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:10, 13.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:08, 15.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:08, 15.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:08, 15.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:08, 15.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:01<00:07, 18.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:07, 18.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:07, 18.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:07, 18.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:06, 20.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:06, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:06, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:06, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:05, 22.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:05, 22.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:05, 22.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:05, 22.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:05, 23.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:05, 23.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:05, 23.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:05, 23.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:05, 24.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:05, 24.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:05, 24.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:05, 24.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:04, 25.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:04, 25.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:04, 25.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 25.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:04, 26.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:04, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:04, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:04, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 26.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 26.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:04, 26.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 26.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 26.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 26.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:03, 30.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:03, 30.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:03, 30.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:03, 30.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 30.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:03, 30.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:02<00:03, 33.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:03, 33.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:03, 33.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:03, 33.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 33.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 33.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 33.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 38.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 38.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:02, 38.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:02, 38.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:02, 38.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 38.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:02, 38.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 41.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 41.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 41.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 41.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 41.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 41.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 41.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:02<00:01, 45.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 47.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 49.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 51.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 51.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 51.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 51.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 51.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 51.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 51.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 52.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 52.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 52.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 52.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 52.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 52.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 52.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 53.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 53.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 53.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 53.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 53.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 53.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 53.08 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:00, 53.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:00, 53.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:00, 53.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:00, 53.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:00, 53.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 53.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 53.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 54.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 54.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 54.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 54.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 54.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 54.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 54.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 54.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 54.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 54.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 54.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 54.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 54.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 54.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 54.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 54.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 54.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 54.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 54.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 54.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 54.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 54.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 54.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 54.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 54.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 54.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 54.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 54.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 54.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 54.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 54.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 54.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 54.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 54.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 54.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 55.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 55.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 54.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.99s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.99s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.99s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 54.93 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.41s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  3.99s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 54.93 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.41s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.42s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.25 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.42s/ url]
0 examples [00:00, ? examples/s]2020-08-09 12:08:39.379013: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-09 12:08:39.394464: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-09 12:08:39.394812: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a708814590 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-09 12:08:39.394836: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
31 examples [00:00, 308.39 examples/s]117 examples [00:00, 381.48 examples/s]200 examples [00:00, 455.28 examples/s]287 examples [00:00, 531.13 examples/s]362 examples [00:00, 581.93 examples/s]444 examples [00:00, 636.76 examples/s]533 examples [00:00, 695.14 examples/s]618 examples [00:00, 734.88 examples/s]706 examples [00:00, 772.32 examples/s]789 examples [00:01, 786.83 examples/s]871 examples [00:01, 792.58 examples/s]953 examples [00:01, 798.45 examples/s]1037 examples [00:01, 810.23 examples/s]1119 examples [00:01, 810.28 examples/s]1204 examples [00:01, 821.27 examples/s]1289 examples [00:01, 829.40 examples/s]1378 examples [00:01, 842.87 examples/s]1466 examples [00:01, 852.28 examples/s]1552 examples [00:01, 833.98 examples/s]1636 examples [00:02, 819.14 examples/s]1719 examples [00:02, 811.77 examples/s]1803 examples [00:02, 819.83 examples/s]1886 examples [00:02, 817.26 examples/s]1969 examples [00:02, 820.11 examples/s]2052 examples [00:02, 814.28 examples/s]2142 examples [00:02, 836.86 examples/s]2232 examples [00:02, 853.60 examples/s]2318 examples [00:02, 852.87 examples/s]2407 examples [00:02, 862.94 examples/s]2494 examples [00:03, 825.88 examples/s]2578 examples [00:03, 797.33 examples/s]2666 examples [00:03, 819.98 examples/s]2749 examples [00:03, 817.42 examples/s]2832 examples [00:03, 799.16 examples/s]2913 examples [00:03, 797.38 examples/s]3004 examples [00:03, 827.30 examples/s]3088 examples [00:03, 825.01 examples/s]3175 examples [00:03, 837.52 examples/s]3260 examples [00:03, 831.30 examples/s]3350 examples [00:04, 848.79 examples/s]3436 examples [00:04, 837.76 examples/s]3523 examples [00:04, 845.02 examples/s]3608 examples [00:04, 818.72 examples/s]3696 examples [00:04, 834.69 examples/s]3780 examples [00:04, 795.25 examples/s]3861 examples [00:04, 794.41 examples/s]3948 examples [00:04, 810.23 examples/s]4037 examples [00:04, 831.77 examples/s]4124 examples [00:05, 840.25 examples/s]4210 examples [00:05, 844.03 examples/s]4298 examples [00:05, 852.39 examples/s]4386 examples [00:05, 859.25 examples/s]4473 examples [00:05, 845.43 examples/s]4560 examples [00:05, 850.29 examples/s]4646 examples [00:05, 841.80 examples/s]4731 examples [00:05, 815.90 examples/s]4818 examples [00:05, 828.96 examples/s]4902 examples [00:05, 824.38 examples/s]4985 examples [00:06, 795.14 examples/s]5065 examples [00:06, 794.57 examples/s]5149 examples [00:06, 807.36 examples/s]5233 examples [00:06, 814.44 examples/s]5315 examples [00:06, 813.40 examples/s]5397 examples [00:06, 797.35 examples/s]5482 examples [00:06, 809.91 examples/s]5564 examples [00:06, 777.83 examples/s]5643 examples [00:06, 774.93 examples/s]5721 examples [00:07, 751.49 examples/s]5804 examples [00:07, 772.50 examples/s]5882 examples [00:07, 758.80 examples/s]5959 examples [00:07, 750.71 examples/s]6042 examples [00:07, 772.42 examples/s]6127 examples [00:07, 791.91 examples/s]6211 examples [00:07, 805.47 examples/s]6298 examples [00:07, 822.86 examples/s]6386 examples [00:07, 838.19 examples/s]6476 examples [00:07, 853.78 examples/s]6565 examples [00:08, 861.55 examples/s]6652 examples [00:08, 853.85 examples/s]6741 examples [00:08, 862.94 examples/s]6831 examples [00:08, 871.21 examples/s]6919 examples [00:08, 872.55 examples/s]7007 examples [00:08, 864.98 examples/s]7094 examples [00:08, 838.72 examples/s]7179 examples [00:08, 822.64 examples/s]7262 examples [00:08, 813.88 examples/s]7350 examples [00:08, 831.08 examples/s]7438 examples [00:09, 843.01 examples/s]7524 examples [00:09, 846.02 examples/s]7613 examples [00:09, 856.89 examples/s]7700 examples [00:09, 860.65 examples/s]7787 examples [00:09, 849.97 examples/s]7876 examples [00:09, 860.99 examples/s]7963 examples [00:09, 860.34 examples/s]8051 examples [00:09, 864.48 examples/s]8142 examples [00:09, 875.29 examples/s]8230 examples [00:09, 869.55 examples/s]8318 examples [00:10, 864.10 examples/s]8405 examples [00:10, 857.28 examples/s]8491 examples [00:10, 849.49 examples/s]8578 examples [00:10, 855.15 examples/s]8665 examples [00:10, 859.42 examples/s]8752 examples [00:10, 861.78 examples/s]8839 examples [00:10, 858.46 examples/s]8925 examples [00:10, 835.61 examples/s]9009 examples [00:10, 816.55 examples/s]9091 examples [00:11, 816.92 examples/s]9177 examples [00:11, 826.77 examples/s]9260 examples [00:11, 825.26 examples/s]9348 examples [00:11, 840.04 examples/s]9433 examples [00:11, 820.67 examples/s]9519 examples [00:11, 830.09 examples/s]9604 examples [00:11, 834.68 examples/s]9688 examples [00:11, 836.18 examples/s]9777 examples [00:11, 850.02 examples/s]9864 examples [00:11, 855.87 examples/s]9950 examples [00:12, 845.77 examples/s]10035 examples [00:12, 788.27 examples/s]10123 examples [00:12, 806.92 examples/s]10205 examples [00:12, 795.04 examples/s]10288 examples [00:12, 803.27 examples/s]10375 examples [00:12, 821.96 examples/s]10458 examples [00:12, 802.40 examples/s]10539 examples [00:12, 777.03 examples/s]10619 examples [00:12, 783.08 examples/s]10701 examples [00:12, 793.74 examples/s]10788 examples [00:13, 815.09 examples/s]10874 examples [00:13, 826.34 examples/s]10963 examples [00:13, 844.42 examples/s]11051 examples [00:13, 851.83 examples/s]11137 examples [00:13, 833.97 examples/s]11223 examples [00:13, 841.54 examples/s]11308 examples [00:13, 832.36 examples/s]11392 examples [00:13, 800.24 examples/s]11473 examples [00:13, 766.88 examples/s]11554 examples [00:14, 777.83 examples/s]11638 examples [00:14, 795.15 examples/s]11722 examples [00:14, 807.95 examples/s]11809 examples [00:14, 824.53 examples/s]11894 examples [00:14, 831.91 examples/s]11978 examples [00:14, 811.64 examples/s]12060 examples [00:14, 810.02 examples/s]12142 examples [00:14, 804.01 examples/s]12233 examples [00:14, 831.46 examples/s]12317 examples [00:14, 830.65 examples/s]12405 examples [00:15, 844.37 examples/s]12492 examples [00:15, 849.02 examples/s]12578 examples [00:15, 838.75 examples/s]12665 examples [00:15, 845.74 examples/s]12752 examples [00:15, 852.85 examples/s]12838 examples [00:15, 843.95 examples/s]12923 examples [00:15, 817.18 examples/s]13005 examples [00:15, 788.64 examples/s]13093 examples [00:15, 813.57 examples/s]13181 examples [00:15, 829.99 examples/s]13265 examples [00:16, 831.32 examples/s]13349 examples [00:16, 827.01 examples/s]13432 examples [00:16, 827.76 examples/s]13520 examples [00:16, 841.94 examples/s]13605 examples [00:16, 833.09 examples/s]13690 examples [00:16, 837.58 examples/s]13779 examples [00:16, 850.37 examples/s]13865 examples [00:16, 816.40 examples/s]13948 examples [00:16, 815.21 examples/s]14035 examples [00:17, 828.70 examples/s]14122 examples [00:17, 839.56 examples/s]14213 examples [00:17, 857.44 examples/s]14299 examples [00:17, 841.15 examples/s]14386 examples [00:17, 847.79 examples/s]14471 examples [00:17, 833.27 examples/s]14555 examples [00:17, 816.86 examples/s]14644 examples [00:17, 836.46 examples/s]14729 examples [00:17, 840.35 examples/s]14814 examples [00:17, 837.62 examples/s]14898 examples [00:18, 819.27 examples/s]14986 examples [00:18, 835.63 examples/s]15070 examples [00:18, 806.87 examples/s]15157 examples [00:18, 818.68 examples/s]15243 examples [00:18, 828.42 examples/s]15331 examples [00:18, 841.32 examples/s]15419 examples [00:18, 850.59 examples/s]15505 examples [00:18, 841.46 examples/s]15590 examples [00:18, 820.98 examples/s]15678 examples [00:18, 837.13 examples/s]15765 examples [00:19, 845.62 examples/s]15852 examples [00:19, 852.62 examples/s]15941 examples [00:19, 860.49 examples/s]16028 examples [00:19, 840.66 examples/s]16113 examples [00:19, 823.65 examples/s]16198 examples [00:19, 829.01 examples/s]16283 examples [00:19, 833.50 examples/s]16368 examples [00:19, 835.65 examples/s]16452 examples [00:19, 820.66 examples/s]16535 examples [00:20, 821.30 examples/s]16618 examples [00:20, 791.80 examples/s]16705 examples [00:20, 812.76 examples/s]16790 examples [00:20, 821.88 examples/s]16874 examples [00:20, 824.76 examples/s]16959 examples [00:20, 831.27 examples/s]17047 examples [00:20, 845.10 examples/s]17136 examples [00:20, 856.70 examples/s]17222 examples [00:20, 827.01 examples/s]17306 examples [00:20, 826.41 examples/s]17392 examples [00:21, 834.29 examples/s]17478 examples [00:21, 839.94 examples/s]17566 examples [00:21, 849.23 examples/s]17652 examples [00:21, 852.08 examples/s]17738 examples [00:21, 846.40 examples/s]17823 examples [00:21, 842.94 examples/s]17908 examples [00:21, 828.17 examples/s]17991 examples [00:21, 811.32 examples/s]18073 examples [00:21, 813.02 examples/s]18155 examples [00:21, 810.43 examples/s]18241 examples [00:22, 823.07 examples/s]18329 examples [00:22, 838.59 examples/s]18418 examples [00:22, 852.10 examples/s]18504 examples [00:22, 853.82 examples/s]18590 examples [00:22, 845.18 examples/s]18675 examples [00:22, 834.23 examples/s]18760 examples [00:22, 836.87 examples/s]18844 examples [00:22, 834.89 examples/s]18928 examples [00:22, 828.23 examples/s]19014 examples [00:22, 836.77 examples/s]19099 examples [00:23, 840.61 examples/s]19187 examples [00:23, 851.36 examples/s]19273 examples [00:23, 840.57 examples/s]19358 examples [00:23, 817.58 examples/s]19440 examples [00:23, 817.71 examples/s]19526 examples [00:23, 828.86 examples/s]19611 examples [00:23, 835.06 examples/s]19697 examples [00:23, 840.99 examples/s]19782 examples [00:23, 837.78 examples/s]19866 examples [00:24, 797.82 examples/s]19947 examples [00:24, 798.40 examples/s]20028 examples [00:24, 766.47 examples/s]20115 examples [00:24, 794.42 examples/s]20199 examples [00:24, 807.25 examples/s]20281 examples [00:24, 801.58 examples/s]20362 examples [00:24, 801.71 examples/s]20443 examples [00:24, 772.33 examples/s]20523 examples [00:24, 779.39 examples/s]20602 examples [00:24, 781.00 examples/s]20681 examples [00:25, 755.32 examples/s]20766 examples [00:25, 779.44 examples/s]20853 examples [00:25, 802.55 examples/s]20941 examples [00:25, 822.64 examples/s]21024 examples [00:25, 814.04 examples/s]21111 examples [00:25, 827.79 examples/s]21200 examples [00:25, 843.73 examples/s]21285 examples [00:25, 840.88 examples/s]21370 examples [00:25, 832.69 examples/s]21454 examples [00:25, 804.76 examples/s]21535 examples [00:26, 796.48 examples/s]21615 examples [00:26, 789.64 examples/s]21702 examples [00:26, 811.44 examples/s]21786 examples [00:26, 817.75 examples/s]21868 examples [00:26, 818.11 examples/s]21950 examples [00:26, 817.08 examples/s]22036 examples [00:26, 827.49 examples/s]22121 examples [00:26, 833.21 examples/s]22205 examples [00:26, 809.65 examples/s]22289 examples [00:27, 817.95 examples/s]22378 examples [00:27, 836.42 examples/s]22465 examples [00:27, 844.38 examples/s]22550 examples [00:27, 843.61 examples/s]22635 examples [00:27, 843.83 examples/s]22721 examples [00:27, 846.74 examples/s]22808 examples [00:27, 853.01 examples/s]22894 examples [00:27, 814.43 examples/s]22976 examples [00:27, 810.76 examples/s]23058 examples [00:27, 808.84 examples/s]23143 examples [00:28, 818.51 examples/s]23226 examples [00:28, 796.69 examples/s]23311 examples [00:28, 810.53 examples/s]23396 examples [00:28, 820.07 examples/s]23479 examples [00:28, 792.34 examples/s]23565 examples [00:28, 809.49 examples/s]23647 examples [00:28, 792.21 examples/s]23727 examples [00:28, 759.77 examples/s]23804 examples [00:28, 749.72 examples/s]23880 examples [00:29, 742.16 examples/s]23957 examples [00:29, 749.97 examples/s]24033 examples [00:29, 748.67 examples/s]24109 examples [00:29, 750.21 examples/s]24185 examples [00:29, 751.80 examples/s]24261 examples [00:29, 735.85 examples/s]24342 examples [00:29, 754.11 examples/s]24419 examples [00:29, 756.55 examples/s]24509 examples [00:29, 793.78 examples/s]24593 examples [00:29, 804.62 examples/s]24678 examples [00:30, 817.22 examples/s]24766 examples [00:30, 834.44 examples/s]24852 examples [00:30, 839.93 examples/s]24939 examples [00:30, 845.77 examples/s]25027 examples [00:30, 855.02 examples/s]25113 examples [00:30, 845.58 examples/s]25204 examples [00:30, 862.73 examples/s]25291 examples [00:30, 858.93 examples/s]25378 examples [00:30, 846.47 examples/s]25463 examples [00:30, 835.79 examples/s]25547 examples [00:31, 833.31 examples/s]25636 examples [00:31, 847.07 examples/s]25725 examples [00:31, 857.54 examples/s]25813 examples [00:31, 863.20 examples/s]25901 examples [00:31, 867.18 examples/s]25988 examples [00:31, 862.84 examples/s]26075 examples [00:31, 854.41 examples/s]26162 examples [00:31, 856.72 examples/s]26249 examples [00:31, 859.09 examples/s]26336 examples [00:31, 860.84 examples/s]26423 examples [00:32, 862.70 examples/s]26510 examples [00:32, 845.36 examples/s]26596 examples [00:32, 847.88 examples/s]26681 examples [00:32, 831.49 examples/s]26765 examples [00:32, 833.14 examples/s]26851 examples [00:32, 839.04 examples/s]26935 examples [00:32, 824.56 examples/s]27019 examples [00:32, 827.59 examples/s]27102 examples [00:32, 805.28 examples/s]27183 examples [00:32, 803.18 examples/s]27269 examples [00:33, 816.54 examples/s]27351 examples [00:33, 803.36 examples/s]27436 examples [00:33, 816.55 examples/s]27522 examples [00:33, 828.37 examples/s]27610 examples [00:33, 842.52 examples/s]27697 examples [00:33, 847.74 examples/s]27782 examples [00:33, 847.32 examples/s]27867 examples [00:33, 824.16 examples/s]27950 examples [00:33, 814.18 examples/s]28037 examples [00:34, 829.47 examples/s]28126 examples [00:34, 845.06 examples/s]28211 examples [00:34, 844.97 examples/s]28299 examples [00:34, 852.82 examples/s]28385 examples [00:34, 819.34 examples/s]28468 examples [00:34, 808.47 examples/s]28550 examples [00:34, 810.06 examples/s]28632 examples [00:34, 806.62 examples/s]28714 examples [00:34, 808.10 examples/s]28795 examples [00:34, 800.50 examples/s]28880 examples [00:35, 813.59 examples/s]28966 examples [00:35, 825.24 examples/s]29053 examples [00:35, 837.27 examples/s]29141 examples [00:35, 849.10 examples/s]29229 examples [00:35, 855.93 examples/s]29318 examples [00:35, 862.89 examples/s]29405 examples [00:35, 858.80 examples/s]29491 examples [00:35, 856.72 examples/s]29579 examples [00:35, 863.00 examples/s]29667 examples [00:35, 865.10 examples/s]29754 examples [00:36, 850.63 examples/s]29840 examples [00:36, 853.24 examples/s]29928 examples [00:36, 860.77 examples/s]30015 examples [00:36, 781.43 examples/s]30095 examples [00:36, 776.89 examples/s]30174 examples [00:36, 774.89 examples/s]30259 examples [00:36, 794.82 examples/s]30347 examples [00:36, 814.97 examples/s]30432 examples [00:36, 822.68 examples/s]30518 examples [00:37, 833.50 examples/s]30602 examples [00:37, 819.93 examples/s]30687 examples [00:37, 826.08 examples/s]30775 examples [00:37, 838.99 examples/s]30862 examples [00:37, 847.21 examples/s]30950 examples [00:37, 856.75 examples/s]31037 examples [00:37, 858.13 examples/s]31123 examples [00:37, 850.60 examples/s]31210 examples [00:37, 855.88 examples/s]31296 examples [00:37, 855.43 examples/s]31382 examples [00:38, 835.52 examples/s]31466 examples [00:38, 808.41 examples/s]31548 examples [00:38, 802.15 examples/s]31636 examples [00:38, 821.43 examples/s]31723 examples [00:38, 834.16 examples/s]31810 examples [00:38, 843.86 examples/s]31895 examples [00:38, 827.59 examples/s]31978 examples [00:38, 813.46 examples/s]32060 examples [00:38, 797.42 examples/s]32144 examples [00:38, 808.38 examples/s]32229 examples [00:39, 819.43 examples/s]32312 examples [00:39, 818.36 examples/s]32401 examples [00:39, 837.91 examples/s]32485 examples [00:39, 837.98 examples/s]32574 examples [00:39, 851.58 examples/s]32660 examples [00:39, 847.89 examples/s]32745 examples [00:39, 839.17 examples/s]32830 examples [00:39, 811.38 examples/s]32923 examples [00:39, 843.19 examples/s]33008 examples [00:39, 841.08 examples/s]33095 examples [00:40, 847.27 examples/s]33180 examples [00:40, 836.45 examples/s]33264 examples [00:40, 836.24 examples/s]33350 examples [00:40, 843.07 examples/s]33438 examples [00:40, 852.48 examples/s]33524 examples [00:40, 854.49 examples/s]33610 examples [00:40, 829.59 examples/s]33694 examples [00:40, 819.80 examples/s]33780 examples [00:40, 830.58 examples/s]33864 examples [00:41, 821.90 examples/s]33947 examples [00:41, 811.71 examples/s]34034 examples [00:41, 827.63 examples/s]34123 examples [00:41, 843.99 examples/s]34208 examples [00:41, 843.52 examples/s]34293 examples [00:41, 834.73 examples/s]34377 examples [00:41, 814.15 examples/s]34459 examples [00:41, 805.98 examples/s]34546 examples [00:41, 823.38 examples/s]34631 examples [00:41, 831.03 examples/s]34716 examples [00:42, 836.57 examples/s]34800 examples [00:42, 811.26 examples/s]34890 examples [00:42, 835.65 examples/s]34976 examples [00:42, 842.24 examples/s]35064 examples [00:42, 852.26 examples/s]35150 examples [00:42, 827.33 examples/s]35234 examples [00:42, 823.40 examples/s]35317 examples [00:42, 806.25 examples/s]35399 examples [00:42, 807.35 examples/s]35484 examples [00:42, 817.78 examples/s]35568 examples [00:43, 820.97 examples/s]35651 examples [00:43, 772.29 examples/s]35730 examples [00:43, 775.69 examples/s]35812 examples [00:43, 788.11 examples/s]35892 examples [00:43, 786.75 examples/s]35973 examples [00:43, 791.52 examples/s]36057 examples [00:43, 803.02 examples/s]36145 examples [00:43, 821.67 examples/s]36233 examples [00:43, 836.01 examples/s]36319 examples [00:44, 842.93 examples/s]36407 examples [00:44, 851.28 examples/s]36493 examples [00:44, 838.26 examples/s]36579 examples [00:44, 842.58 examples/s]36668 examples [00:44, 854.62 examples/s]36757 examples [00:44, 864.45 examples/s]36846 examples [00:44, 870.05 examples/s]36934 examples [00:44, 837.61 examples/s]37019 examples [00:44, 840.90 examples/s]37109 examples [00:44, 857.60 examples/s]37197 examples [00:45, 862.89 examples/s]37284 examples [00:45, 863.96 examples/s]37371 examples [00:45, 807.81 examples/s]37456 examples [00:45, 817.58 examples/s]37539 examples [00:45, 819.78 examples/s]37622 examples [00:45, 819.63 examples/s]37707 examples [00:45, 828.09 examples/s]37791 examples [00:45, 825.45 examples/s]37877 examples [00:45, 834.62 examples/s]37961 examples [00:45, 832.00 examples/s]38045 examples [00:46, 827.08 examples/s]38131 examples [00:46, 836.44 examples/s]38216 examples [00:46, 839.82 examples/s]38303 examples [00:46, 845.99 examples/s]38388 examples [00:46, 844.61 examples/s]38476 examples [00:46, 852.24 examples/s]38563 examples [00:46, 855.15 examples/s]38649 examples [00:46, 809.57 examples/s]38735 examples [00:46, 822.49 examples/s]38818 examples [00:46, 823.14 examples/s]38909 examples [00:47, 845.88 examples/s]38994 examples [00:47, 841.43 examples/s]39079 examples [00:47, 828.70 examples/s]39163 examples [00:47, 822.67 examples/s]39246 examples [00:47, 818.97 examples/s]39330 examples [00:47, 822.75 examples/s]39413 examples [00:47, 817.69 examples/s]39495 examples [00:47, 790.16 examples/s]39580 examples [00:47, 805.19 examples/s]39666 examples [00:48, 818.69 examples/s]39752 examples [00:48, 828.82 examples/s]39837 examples [00:48, 832.53 examples/s]39921 examples [00:48, 816.49 examples/s]40003 examples [00:48, 766.76 examples/s]40089 examples [00:48, 790.98 examples/s]40174 examples [00:48, 805.94 examples/s]40256 examples [00:48, 760.08 examples/s]40336 examples [00:48, 770.91 examples/s]40419 examples [00:48, 787.27 examples/s]40503 examples [00:49, 800.09 examples/s]40584 examples [00:49, 781.02 examples/s]40663 examples [00:49, 765.49 examples/s]40749 examples [00:49, 789.15 examples/s]40838 examples [00:49, 814.89 examples/s]40926 examples [00:49, 832.43 examples/s]41010 examples [00:49, 829.50 examples/s]41094 examples [00:49, 801.50 examples/s]41181 examples [00:49, 820.05 examples/s]41269 examples [00:50, 835.61 examples/s]41353 examples [00:50, 828.73 examples/s]41437 examples [00:50, 831.67 examples/s]41522 examples [00:50, 835.99 examples/s]41612 examples [00:50, 853.86 examples/s]41698 examples [00:50, 854.74 examples/s]41787 examples [00:50, 863.11 examples/s]41875 examples [00:50, 867.20 examples/s]41962 examples [00:50, 861.82 examples/s]42050 examples [00:50, 866.15 examples/s]42141 examples [00:51, 876.55 examples/s]42232 examples [00:51, 885.74 examples/s]42321 examples [00:51, 864.10 examples/s]42408 examples [00:51, 862.94 examples/s]42498 examples [00:51, 870.89 examples/s]42586 examples [00:51, 868.77 examples/s]42673 examples [00:51, 849.20 examples/s]42761 examples [00:51, 855.61 examples/s]42847 examples [00:51, 840.57 examples/s]42935 examples [00:51, 849.36 examples/s]43022 examples [00:52, 851.63 examples/s]43108 examples [00:52, 848.69 examples/s]43196 examples [00:52, 854.94 examples/s]43282 examples [00:52, 853.79 examples/s]43372 examples [00:52, 866.91 examples/s]43459 examples [00:52, 846.23 examples/s]43547 examples [00:52, 854.34 examples/s]43636 examples [00:52, 864.32 examples/s]43724 examples [00:52, 868.64 examples/s]43811 examples [00:52, 866.49 examples/s]43898 examples [00:53, 864.21 examples/s]43985 examples [00:53, 851.55 examples/s]44073 examples [00:53, 858.31 examples/s]44159 examples [00:53, 858.32 examples/s]44249 examples [00:53, 868.72 examples/s]44336 examples [00:53, 847.79 examples/s]44422 examples [00:53, 850.15 examples/s]44508 examples [00:53, 836.82 examples/s]44597 examples [00:53, 851.64 examples/s]44683 examples [00:54, 838.27 examples/s]44768 examples [00:54, 839.56 examples/s]44858 examples [00:54, 856.10 examples/s]44947 examples [00:54, 864.45 examples/s]45034 examples [00:54, 862.74 examples/s]45121 examples [00:54, 860.87 examples/s]45208 examples [00:54, 858.01 examples/s]45295 examples [00:54, 860.95 examples/s]45382 examples [00:54, 851.91 examples/s]45468 examples [00:54, 836.70 examples/s]45552 examples [00:55, 835.40 examples/s]45641 examples [00:55, 850.57 examples/s]45727 examples [00:55, 841.61 examples/s]45816 examples [00:55, 852.33 examples/s]45902 examples [00:55, 847.30 examples/s]45987 examples [00:55, 846.44 examples/s]46073 examples [00:55, 850.05 examples/s]46160 examples [00:55, 854.52 examples/s]46250 examples [00:55, 865.48 examples/s]46337 examples [00:55, 864.00 examples/s]46424 examples [00:56, 856.79 examples/s]46510 examples [00:56, 823.63 examples/s]46593 examples [00:56, 806.67 examples/s]46674 examples [00:56, 804.85 examples/s]46758 examples [00:56, 814.48 examples/s]46842 examples [00:56, 819.61 examples/s]46925 examples [00:56, 815.40 examples/s]47007 examples [00:56, 813.56 examples/s]47091 examples [00:56, 820.98 examples/s]47174 examples [00:56, 816.79 examples/s]47263 examples [00:57, 833.82 examples/s]47354 examples [00:57, 853.02 examples/s]47441 examples [00:57, 857.71 examples/s]47530 examples [00:57, 865.28 examples/s]47617 examples [00:57, 864.05 examples/s]47706 examples [00:57, 870.45 examples/s]47794 examples [00:57, 865.05 examples/s]47881 examples [00:57, 862.97 examples/s]47968 examples [00:57, 858.73 examples/s]48054 examples [00:57, 826.27 examples/s]48139 examples [00:58, 829.84 examples/s]48225 examples [00:58, 838.59 examples/s]48312 examples [00:58, 845.08 examples/s]48400 examples [00:58, 855.08 examples/s]48486 examples [00:58, 848.67 examples/s]48571 examples [00:58, 847.12 examples/s]48656 examples [00:58, 841.05 examples/s]48741 examples [00:58, 818.36 examples/s]48824 examples [00:58, 814.68 examples/s]48912 examples [00:59, 830.85 examples/s]49001 examples [00:59, 846.62 examples/s]49090 examples [00:59, 857.84 examples/s]49179 examples [00:59, 864.67 examples/s]49266 examples [00:59, 854.17 examples/s]49352 examples [00:59, 820.38 examples/s]49442 examples [00:59, 840.30 examples/s]49527 examples [00:59, 842.44 examples/s]49616 examples [00:59, 854.88 examples/s]49704 examples [00:59, 860.96 examples/s]49791 examples [01:00, 862.84 examples/s]49878 examples [01:00, 847.25 examples/s]49963 examples [01:00, 828.94 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6048/50000 [00:00<00:00, 60478.18 examples/s] 33%|███▎      | 16749/50000 [00:00<00:00, 69550.50 examples/s] 56%|█████▌    | 27967/50000 [00:00<00:00, 78499.51 examples/s] 79%|███████▉  | 39497/50000 [00:00<00:00, 86811.66 examples/s]                                                               0 examples [00:00, ? examples/s]64 examples [00:00, 638.04 examples/s]145 examples [00:00, 672.72 examples/s]236 examples [00:00, 728.81 examples/s]324 examples [00:00, 766.45 examples/s]413 examples [00:00, 797.81 examples/s]506 examples [00:00, 832.34 examples/s]594 examples [00:00, 841.98 examples/s]681 examples [00:00, 848.56 examples/s]774 examples [00:00, 870.22 examples/s]870 examples [00:01, 893.62 examples/s]963 examples [00:01, 904.02 examples/s]1053 examples [00:01, 895.72 examples/s]1143 examples [00:01, 882.57 examples/s]1231 examples [00:01, 879.21 examples/s]1321 examples [00:01, 883.05 examples/s]1412 examples [00:01, 890.91 examples/s]1503 examples [00:01, 894.00 examples/s]1593 examples [00:01, 887.84 examples/s]1682 examples [00:01, 875.18 examples/s]1775 examples [00:02, 888.74 examples/s]1866 examples [00:02, 892.92 examples/s]1956 examples [00:02, 875.79 examples/s]2044 examples [00:02, 874.72 examples/s]2132 examples [00:02, 872.79 examples/s]2221 examples [00:02, 875.92 examples/s]2309 examples [00:02, 873.05 examples/s]2399 examples [00:02, 879.97 examples/s]2491 examples [00:02, 890.79 examples/s]2585 examples [00:02, 903.16 examples/s]2676 examples [00:03, 893.88 examples/s]2766 examples [00:03, 879.67 examples/s]2855 examples [00:03, 871.62 examples/s]2943 examples [00:03, 845.32 examples/s]3028 examples [00:03, 808.69 examples/s]3114 examples [00:03, 823.32 examples/s]3197 examples [00:03, 817.26 examples/s]3281 examples [00:03, 821.73 examples/s]3377 examples [00:03, 857.71 examples/s]3464 examples [00:04, 836.98 examples/s]3551 examples [00:04, 846.05 examples/s]3641 examples [00:04, 860.27 examples/s]3728 examples [00:04, 856.96 examples/s]3819 examples [00:04, 870.03 examples/s]3907 examples [00:04, 863.95 examples/s]3995 examples [00:04, 865.87 examples/s]4082 examples [00:04, 857.21 examples/s]4170 examples [00:04, 862.38 examples/s]4257 examples [00:04, 854.55 examples/s]4348 examples [00:05, 870.41 examples/s]4436 examples [00:05, 866.01 examples/s]4528 examples [00:05, 880.86 examples/s]4617 examples [00:05, 861.69 examples/s]4706 examples [00:05, 867.63 examples/s]4796 examples [00:05, 876.03 examples/s]4888 examples [00:05, 886.91 examples/s]4978 examples [00:05, 887.70 examples/s]5068 examples [00:05, 889.19 examples/s]5157 examples [00:05, 882.59 examples/s]5246 examples [00:06, 866.39 examples/s]5334 examples [00:06, 868.87 examples/s]5427 examples [00:06, 885.34 examples/s]5516 examples [00:06, 870.34 examples/s]5604 examples [00:06, 848.87 examples/s]5690 examples [00:06, 836.78 examples/s]5778 examples [00:06, 847.48 examples/s]5870 examples [00:06, 866.73 examples/s]5957 examples [00:06, 850.63 examples/s]6043 examples [00:06, 835.86 examples/s]6135 examples [00:07, 858.43 examples/s]6222 examples [00:07, 855.74 examples/s]6308 examples [00:07, 845.51 examples/s]6396 examples [00:07, 853.35 examples/s]6482 examples [00:07, 795.68 examples/s]6563 examples [00:07, 791.82 examples/s]6643 examples [00:07, 790.90 examples/s]6723 examples [00:07, 790.40 examples/s]6806 examples [00:07, 801.81 examples/s]6890 examples [00:08, 812.53 examples/s]6982 examples [00:08, 840.88 examples/s]7073 examples [00:08, 857.93 examples/s]7162 examples [00:08, 866.72 examples/s]7255 examples [00:08, 883.41 examples/s]7348 examples [00:08, 896.09 examples/s]7441 examples [00:08, 904.48 examples/s]7532 examples [00:08, 898.15 examples/s]7623 examples [00:08, 900.64 examples/s]7714 examples [00:08, 893.41 examples/s]7805 examples [00:09, 897.25 examples/s]7897 examples [00:09, 902.33 examples/s]7988 examples [00:09, 901.13 examples/s]8079 examples [00:09, 876.13 examples/s]8168 examples [00:09, 879.53 examples/s]8257 examples [00:09, 863.13 examples/s]8347 examples [00:09, 872.50 examples/s]8435 examples [00:09, 866.80 examples/s]8522 examples [00:09, 837.68 examples/s]8609 examples [00:09, 845.35 examples/s]8700 examples [00:10, 862.90 examples/s]8793 examples [00:10, 880.35 examples/s]8882 examples [00:10, 868.50 examples/s]8970 examples [00:10, 864.75 examples/s]9058 examples [00:10, 868.82 examples/s]9147 examples [00:10, 873.28 examples/s]9235 examples [00:10, 871.23 examples/s]9323 examples [00:10, 873.39 examples/s]9411 examples [00:10, 867.69 examples/s]9502 examples [00:10, 879.11 examples/s]9592 examples [00:11, 884.31 examples/s]9682 examples [00:11, 888.53 examples/s]9771 examples [00:11, 886.78 examples/s]9860 examples [00:11, 883.03 examples/s]9949 examples [00:11, 865.06 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 97%|█████████▋| 9674/10000 [00:00<00:00, 96734.33 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYS9797/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYS9797/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f110c69a158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f110c69a158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f110c69a158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f11580780f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1096944278>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f110c69a158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f110c69a158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f110c69a158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f110aa82358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f110aa821d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.02 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.12 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.12 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.73 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.73 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.57 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.57 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.65 file/s]2020-08-09 12:10:00.857019: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-09 12:10:00.861469: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-09 12:10:00.861693: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b57cfe5a60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-09 12:10:00.861727: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 479810.90it/s] 84%|████████▍ | 8355840/9912422 [00:00<00:02, 683704.23it/s]9920512it [00:00, 45716191.50it/s]                           
0it [00:00, ?it/s]32768it [00:00, 666373.91it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 993847.03it/s]1654784it [00:00, 12650722.61it/s]                          
0it [00:00, ?it/s]8192it [00:00, 232370.78it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
