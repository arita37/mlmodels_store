
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f3db18a8b70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f3db18a8b70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f3e1c518510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f3e1c518510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3e3b745ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3e3b745ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3dc98c0a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3dc98c0a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3dc98c0a60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 28%|██▊       | 2752512/9912422 [00:00<00:00, 25469794.61it/s]9920512it [00:00, 30567073.79it/s]                             
0it [00:00, ?it/s]32768it [00:00, 596802.99it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 148973.37it/s]1654784it [00:00, 10989823.75it/s]                         
0it [00:00, ?it/s]8192it [00:00, 170423.38it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3db17646d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3db1772978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f3dc98c06a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f3dc98c06a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f3dc98c06a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:14,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:13,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:13,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:12,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:12,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:51,  3.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:51,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:50,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:50,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:49,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:49,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:49,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:34,  4.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:34,  4.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:33,  4.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:23,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 11.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 20.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 33.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 41.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 49.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 49.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 57.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 57.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 57.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 57.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 57.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 57.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 64.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 70.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 75.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 75.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 75.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 75.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 75.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 75.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 75.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 75.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 75.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 75.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 75.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 80.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 80.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 80.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 80.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 80.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 80.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 80.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.84 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.58s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 76.84 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.58s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.58s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.35 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.58s/ url]
0 examples [00:00, ? examples/s]2020-07-31 00:09:10.240813: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-31 00:09:10.258530: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-31 00:09:10.258724: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5591c267fbf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-31 00:09:10.258744: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.40 examples/s]89 examples [00:00,  4.84 examples/s]180 examples [00:00,  6.90 examples/s]275 examples [00:00,  9.83 examples/s]368 examples [00:00, 13.98 examples/s]458 examples [00:00, 19.84 examples/s]545 examples [00:00, 28.07 examples/s]635 examples [00:00, 39.56 examples/s]733 examples [00:01, 55.55 examples/s]835 examples [00:01, 77.53 examples/s]936 examples [00:01, 107.15 examples/s]1032 examples [00:01, 146.05 examples/s]1129 examples [00:01, 195.95 examples/s]1228 examples [00:01, 257.94 examples/s]1324 examples [00:01, 329.99 examples/s]1424 examples [00:01, 412.83 examples/s]1521 examples [00:01, 495.86 examples/s]1620 examples [00:02, 582.24 examples/s]1717 examples [00:02, 660.57 examples/s]1816 examples [00:02, 732.22 examples/s]1913 examples [00:02, 784.62 examples/s]2010 examples [00:02, 818.00 examples/s]2105 examples [00:02, 834.89 examples/s]2198 examples [00:02, 855.85 examples/s]2295 examples [00:02, 887.17 examples/s]2390 examples [00:02, 904.73 examples/s]2484 examples [00:02, 904.29 examples/s]2577 examples [00:03, 898.06 examples/s]2671 examples [00:03, 907.79 examples/s]2763 examples [00:03, 907.28 examples/s]2855 examples [00:03, 910.25 examples/s]2947 examples [00:03, 904.28 examples/s]3038 examples [00:03, 891.49 examples/s]3135 examples [00:03, 912.99 examples/s]3227 examples [00:03, 911.70 examples/s]3323 examples [00:03, 925.30 examples/s]3419 examples [00:03, 932.87 examples/s]3516 examples [00:04, 942.59 examples/s]3611 examples [00:04, 926.89 examples/s]3704 examples [00:04, 925.69 examples/s]3798 examples [00:04, 927.57 examples/s]3891 examples [00:04, 917.81 examples/s]3986 examples [00:04, 926.17 examples/s]4081 examples [00:04, 930.74 examples/s]4176 examples [00:04, 934.06 examples/s]4273 examples [00:04, 944.04 examples/s]4368 examples [00:04, 939.17 examples/s]4463 examples [00:05, 940.72 examples/s]4558 examples [00:05, 939.97 examples/s]4653 examples [00:05, 935.28 examples/s]4747 examples [00:05, 927.23 examples/s]4840 examples [00:05, 900.10 examples/s]4931 examples [00:05, 894.56 examples/s]5021 examples [00:05, 866.04 examples/s]5108 examples [00:05, 837.19 examples/s]5195 examples [00:05, 845.86 examples/s]5288 examples [00:06, 868.31 examples/s]5386 examples [00:06, 897.60 examples/s]5483 examples [00:06, 916.51 examples/s]5576 examples [00:06, 918.21 examples/s]5670 examples [00:06, 922.65 examples/s]5763 examples [00:06, 911.60 examples/s]5858 examples [00:06, 920.04 examples/s]5951 examples [00:06, 902.40 examples/s]6042 examples [00:06, 887.62 examples/s]6131 examples [00:06, 872.52 examples/s]6221 examples [00:07, 878.11 examples/s]6310 examples [00:07, 880.42 examples/s]6401 examples [00:07, 888.39 examples/s]6490 examples [00:07, 873.06 examples/s]6580 examples [00:07, 879.52 examples/s]6670 examples [00:07, 883.55 examples/s]6764 examples [00:07, 897.29 examples/s]6855 examples [00:07, 899.49 examples/s]6951 examples [00:07, 915.73 examples/s]7043 examples [00:07, 910.94 examples/s]7136 examples [00:08, 915.65 examples/s]7229 examples [00:08, 919.44 examples/s]7321 examples [00:08, 902.63 examples/s]7412 examples [00:08, 871.19 examples/s]7500 examples [00:08, 845.08 examples/s]7587 examples [00:08, 851.51 examples/s]7682 examples [00:08, 876.42 examples/s]7771 examples [00:08, 872.14 examples/s]7859 examples [00:08, 864.04 examples/s]7946 examples [00:09, 856.16 examples/s]8036 examples [00:09, 867.67 examples/s]8123 examples [00:09, 866.10 examples/s]8217 examples [00:09, 885.22 examples/s]8306 examples [00:09, 875.76 examples/s]8394 examples [00:09, 870.69 examples/s]8482 examples [00:09, 832.99 examples/s]8571 examples [00:09, 848.52 examples/s]8664 examples [00:09, 869.93 examples/s]8757 examples [00:09, 885.85 examples/s]8854 examples [00:10, 908.07 examples/s]8946 examples [00:10, 866.10 examples/s]9034 examples [00:10, 862.74 examples/s]9121 examples [00:10, 850.16 examples/s]9210 examples [00:10, 861.07 examples/s]9297 examples [00:10, 848.68 examples/s]9388 examples [00:10, 864.65 examples/s]9486 examples [00:10, 895.51 examples/s]9577 examples [00:10, 895.92 examples/s]9669 examples [00:10, 900.77 examples/s]9760 examples [00:11, 901.10 examples/s]9857 examples [00:11, 919.29 examples/s]9955 examples [00:11, 935.65 examples/s]10049 examples [00:11, 352.32 examples/s]10137 examples [00:12, 428.89 examples/s]10222 examples [00:12, 503.66 examples/s]10316 examples [00:12, 584.61 examples/s]10415 examples [00:12, 665.42 examples/s]10512 examples [00:12, 733.13 examples/s]10607 examples [00:12, 785.52 examples/s]10699 examples [00:12, 808.00 examples/s]10794 examples [00:12, 844.18 examples/s]10892 examples [00:12, 880.25 examples/s]10990 examples [00:12, 907.35 examples/s]11092 examples [00:13, 936.75 examples/s]11189 examples [00:13, 911.83 examples/s]11283 examples [00:13, 906.91 examples/s]11377 examples [00:13, 913.90 examples/s]11470 examples [00:13, 914.90 examples/s]11563 examples [00:13, 901.18 examples/s]11654 examples [00:13, 890.18 examples/s]11744 examples [00:13, 874.79 examples/s]11836 examples [00:13, 885.20 examples/s]11930 examples [00:13, 898.80 examples/s]12021 examples [00:14, 899.61 examples/s]12112 examples [00:14, 891.96 examples/s]12202 examples [00:14, 892.26 examples/s]12293 examples [00:14, 895.21 examples/s]12385 examples [00:14, 901.69 examples/s]12476 examples [00:14, 896.25 examples/s]12566 examples [00:14, 883.12 examples/s]12657 examples [00:14, 882.20 examples/s]12752 examples [00:14, 900.25 examples/s]12847 examples [00:14, 913.20 examples/s]12944 examples [00:15, 929.31 examples/s]13038 examples [00:15, 932.30 examples/s]13132 examples [00:15, 913.47 examples/s]13229 examples [00:15, 929.14 examples/s]13323 examples [00:15, 903.02 examples/s]13414 examples [00:15, 899.61 examples/s]13505 examples [00:15, 894.84 examples/s]13600 examples [00:15, 909.56 examples/s]13692 examples [00:15, 893.99 examples/s]13786 examples [00:16, 906.59 examples/s]13877 examples [00:16, 907.50 examples/s]13968 examples [00:16, 886.74 examples/s]14061 examples [00:16, 898.17 examples/s]14156 examples [00:16, 910.42 examples/s]14248 examples [00:16, 906.68 examples/s]14344 examples [00:16, 919.53 examples/s]14441 examples [00:16, 931.40 examples/s]14537 examples [00:16, 938.29 examples/s]14631 examples [00:16, 906.18 examples/s]14727 examples [00:17, 919.42 examples/s]14820 examples [00:17, 917.49 examples/s]14912 examples [00:17, 916.81 examples/s]15011 examples [00:17, 936.86 examples/s]15114 examples [00:17, 961.88 examples/s]15211 examples [00:17, 951.57 examples/s]15307 examples [00:17, 949.35 examples/s]15403 examples [00:17, 937.05 examples/s]15497 examples [00:17, 928.35 examples/s]15590 examples [00:17, 925.09 examples/s]15686 examples [00:18, 932.69 examples/s]15788 examples [00:18, 955.07 examples/s]15884 examples [00:18, 948.31 examples/s]15979 examples [00:18, 934.86 examples/s]16073 examples [00:18, 927.68 examples/s]16166 examples [00:18, 928.17 examples/s]16259 examples [00:18, 914.67 examples/s]16351 examples [00:18, 899.59 examples/s]16448 examples [00:18, 917.65 examples/s]16542 examples [00:19, 923.91 examples/s]16636 examples [00:19, 927.43 examples/s]16732 examples [00:19, 934.36 examples/s]16826 examples [00:19, 932.09 examples/s]16924 examples [00:19, 945.02 examples/s]17019 examples [00:19, 942.39 examples/s]17114 examples [00:19, 937.07 examples/s]17208 examples [00:19, 928.94 examples/s]17301 examples [00:19, 925.41 examples/s]17399 examples [00:19, 940.24 examples/s]17496 examples [00:20, 946.06 examples/s]17591 examples [00:20, 938.02 examples/s]17692 examples [00:20, 957.82 examples/s]17788 examples [00:20, 952.82 examples/s]17884 examples [00:20, 953.22 examples/s]17980 examples [00:20, 939.48 examples/s]18077 examples [00:20, 946.12 examples/s]18172 examples [00:20, 940.36 examples/s]18267 examples [00:20, 905.38 examples/s]18358 examples [00:20, 898.54 examples/s]18449 examples [00:21, 890.36 examples/s]18547 examples [00:21, 913.96 examples/s]18639 examples [00:21, 902.02 examples/s]18730 examples [00:21, 898.51 examples/s]18822 examples [00:21, 899.50 examples/s]18913 examples [00:21, 900.56 examples/s]19004 examples [00:21, 871.77 examples/s]19092 examples [00:21, 852.32 examples/s]19178 examples [00:21, 853.40 examples/s]19268 examples [00:21, 866.36 examples/s]19356 examples [00:22, 867.76 examples/s]19447 examples [00:22, 878.04 examples/s]19535 examples [00:22, 876.74 examples/s]19624 examples [00:22, 879.71 examples/s]19721 examples [00:22, 903.31 examples/s]19812 examples [00:22, 903.82 examples/s]19903 examples [00:22, 892.11 examples/s]19993 examples [00:22, 888.23 examples/s]20082 examples [00:22, 838.84 examples/s]20170 examples [00:23, 850.10 examples/s]20263 examples [00:23, 872.22 examples/s]20354 examples [00:23, 882.77 examples/s]20443 examples [00:23, 872.55 examples/s]20536 examples [00:23, 888.56 examples/s]20630 examples [00:23, 902.39 examples/s]20721 examples [00:23, 898.70 examples/s]20812 examples [00:23, 889.12 examples/s]20902 examples [00:23, 872.21 examples/s]20990 examples [00:23, 872.28 examples/s]21078 examples [00:24, 866.92 examples/s]21165 examples [00:24, 854.26 examples/s]21256 examples [00:24, 869.49 examples/s]21347 examples [00:24, 879.38 examples/s]21439 examples [00:24, 889.32 examples/s]21532 examples [00:24, 900.27 examples/s]21625 examples [00:24, 908.39 examples/s]21716 examples [00:24, 891.00 examples/s]21806 examples [00:24, 881.57 examples/s]21897 examples [00:24, 889.46 examples/s]21996 examples [00:25, 915.50 examples/s]22090 examples [00:25, 921.50 examples/s]22183 examples [00:25, 901.21 examples/s]22275 examples [00:25, 905.01 examples/s]22366 examples [00:25, 898.26 examples/s]22456 examples [00:25, 896.38 examples/s]22546 examples [00:25, 889.17 examples/s]22639 examples [00:25, 900.15 examples/s]22730 examples [00:25, 881.66 examples/s]22819 examples [00:25, 876.30 examples/s]22907 examples [00:26, 856.88 examples/s]22996 examples [00:26, 865.98 examples/s]23083 examples [00:26, 849.56 examples/s]23169 examples [00:26, 848.97 examples/s]23255 examples [00:26, 842.81 examples/s]23349 examples [00:26, 868.68 examples/s]23441 examples [00:26, 882.73 examples/s]23537 examples [00:26, 902.29 examples/s]23630 examples [00:26, 909.24 examples/s]23722 examples [00:27, 905.02 examples/s]23821 examples [00:27, 928.49 examples/s]23921 examples [00:27, 947.45 examples/s]24017 examples [00:27, 950.58 examples/s]24118 examples [00:27, 965.66 examples/s]24215 examples [00:27, 942.21 examples/s]24310 examples [00:27, 942.94 examples/s]24405 examples [00:27, 936.45 examples/s]24503 examples [00:27, 947.19 examples/s]24599 examples [00:27, 950.55 examples/s]24695 examples [00:28, 926.76 examples/s]24788 examples [00:28, 878.73 examples/s]24883 examples [00:28, 897.54 examples/s]24977 examples [00:28, 907.97 examples/s]25069 examples [00:28, 908.84 examples/s]25166 examples [00:28, 924.03 examples/s]25259 examples [00:28, 919.26 examples/s]25352 examples [00:28, 894.23 examples/s]25442 examples [00:28, 894.56 examples/s]25532 examples [00:28, 886.10 examples/s]25621 examples [00:29, 858.42 examples/s]25715 examples [00:29, 879.15 examples/s]25807 examples [00:29, 886.74 examples/s]25901 examples [00:29, 900.53 examples/s]25992 examples [00:29, 900.29 examples/s]26086 examples [00:29, 909.32 examples/s]26178 examples [00:29, 852.50 examples/s]26270 examples [00:29, 871.17 examples/s]26358 examples [00:29, 843.51 examples/s]26448 examples [00:30, 858.60 examples/s]26544 examples [00:30, 885.74 examples/s]26637 examples [00:30, 898.30 examples/s]26732 examples [00:30, 913.19 examples/s]26827 examples [00:30, 921.19 examples/s]26920 examples [00:30, 915.40 examples/s]27012 examples [00:30, 911.17 examples/s]27104 examples [00:30, 900.14 examples/s]27195 examples [00:30, 896.37 examples/s]27289 examples [00:30, 905.93 examples/s]27380 examples [00:31, 892.46 examples/s]27470 examples [00:31, 884.85 examples/s]27565 examples [00:31, 900.58 examples/s]27661 examples [00:31, 917.16 examples/s]27756 examples [00:31, 924.24 examples/s]27850 examples [00:31, 928.59 examples/s]27944 examples [00:31, 930.86 examples/s]28038 examples [00:31, 929.01 examples/s]28131 examples [00:31, 920.50 examples/s]28224 examples [00:31, 916.58 examples/s]28316 examples [00:32, 909.93 examples/s]28411 examples [00:32, 919.71 examples/s]28504 examples [00:32, 904.48 examples/s]28596 examples [00:32, 907.24 examples/s]28687 examples [00:32, 905.96 examples/s]28781 examples [00:32, 914.31 examples/s]28873 examples [00:32, 912.12 examples/s]28970 examples [00:32, 927.29 examples/s]29063 examples [00:32, 892.33 examples/s]29158 examples [00:32, 908.39 examples/s]29250 examples [00:33, 909.39 examples/s]29342 examples [00:33, 911.86 examples/s]29434 examples [00:33, 890.55 examples/s]29524 examples [00:33, 889.92 examples/s]29618 examples [00:33, 901.23 examples/s]29709 examples [00:33, 875.44 examples/s]29797 examples [00:33, 875.29 examples/s]29885 examples [00:33, 867.88 examples/s]29981 examples [00:33, 892.35 examples/s]30071 examples [00:34, 843.77 examples/s]30171 examples [00:34, 882.46 examples/s]30263 examples [00:34, 892.09 examples/s]30364 examples [00:34, 922.42 examples/s]30458 examples [00:34, 874.71 examples/s]30548 examples [00:34, 880.83 examples/s]30637 examples [00:34, 876.77 examples/s]30727 examples [00:34, 880.46 examples/s]30816 examples [00:34, 872.36 examples/s]30904 examples [00:34, 868.74 examples/s]30992 examples [00:35, 869.61 examples/s]31084 examples [00:35, 883.37 examples/s]31178 examples [00:35, 897.61 examples/s]31270 examples [00:35, 901.42 examples/s]31361 examples [00:35, 903.76 examples/s]31454 examples [00:35, 910.12 examples/s]31546 examples [00:35, 900.25 examples/s]31637 examples [00:35, 891.83 examples/s]31727 examples [00:35, 873.23 examples/s]31818 examples [00:35, 883.08 examples/s]31907 examples [00:36, 847.54 examples/s]32003 examples [00:36, 876.30 examples/s]32100 examples [00:36, 902.36 examples/s]32191 examples [00:36, 885.28 examples/s]32280 examples [00:36, 883.26 examples/s]32369 examples [00:36, 881.80 examples/s]32462 examples [00:36, 894.36 examples/s]32553 examples [00:36, 898.81 examples/s]32644 examples [00:36, 893.90 examples/s]32736 examples [00:37, 898.96 examples/s]32828 examples [00:37, 902.55 examples/s]32926 examples [00:37, 923.11 examples/s]33023 examples [00:37, 935.87 examples/s]33118 examples [00:37, 937.45 examples/s]33221 examples [00:37, 963.36 examples/s]33318 examples [00:37, 922.44 examples/s]33411 examples [00:37, 899.10 examples/s]33502 examples [00:37, 894.04 examples/s]33593 examples [00:37, 897.25 examples/s]33686 examples [00:38, 905.40 examples/s]33777 examples [00:38, 902.18 examples/s]33873 examples [00:38, 916.36 examples/s]33965 examples [00:38, 903.51 examples/s]34061 examples [00:38, 917.28 examples/s]34155 examples [00:38, 921.77 examples/s]34248 examples [00:38, 915.15 examples/s]34341 examples [00:38, 917.85 examples/s]34433 examples [00:38, 899.94 examples/s]34527 examples [00:38, 910.05 examples/s]34621 examples [00:39, 915.33 examples/s]34717 examples [00:39, 925.84 examples/s]34813 examples [00:39, 934.30 examples/s]34907 examples [00:39, 934.50 examples/s]35003 examples [00:39, 938.64 examples/s]35099 examples [00:39, 944.41 examples/s]35195 examples [00:39, 947.20 examples/s]35290 examples [00:39, 937.27 examples/s]35384 examples [00:39, 930.76 examples/s]35480 examples [00:39, 939.35 examples/s]35574 examples [00:40, 923.32 examples/s]35669 examples [00:40, 929.98 examples/s]35763 examples [00:40, 928.54 examples/s]35856 examples [00:40, 924.77 examples/s]35949 examples [00:40, 910.60 examples/s]36041 examples [00:40, 909.53 examples/s]36133 examples [00:40, 905.41 examples/s]36228 examples [00:40, 916.40 examples/s]36320 examples [00:40, 891.45 examples/s]36418 examples [00:41, 915.69 examples/s]36510 examples [00:41, 903.08 examples/s]36606 examples [00:41, 919.21 examples/s]36703 examples [00:41, 933.83 examples/s]36797 examples [00:41, 923.24 examples/s]36899 examples [00:41, 949.08 examples/s]36995 examples [00:41, 941.21 examples/s]37090 examples [00:41, 930.41 examples/s]37184 examples [00:41, 928.48 examples/s]37281 examples [00:41, 938.33 examples/s]37377 examples [00:42, 942.61 examples/s]37473 examples [00:42, 946.99 examples/s]37569 examples [00:42, 947.82 examples/s]37664 examples [00:42, 945.53 examples/s]37759 examples [00:42, 929.33 examples/s]37853 examples [00:42, 923.70 examples/s]37946 examples [00:42, 920.81 examples/s]38039 examples [00:42, 917.45 examples/s]38131 examples [00:42, 909.73 examples/s]38226 examples [00:42, 920.44 examples/s]38325 examples [00:43, 939.93 examples/s]38420 examples [00:43, 937.25 examples/s]38514 examples [00:43, 934.30 examples/s]38608 examples [00:43, 930.34 examples/s]38702 examples [00:43, 924.58 examples/s]38798 examples [00:43, 931.26 examples/s]38892 examples [00:43, 929.73 examples/s]38986 examples [00:43, 901.72 examples/s]39077 examples [00:43, 899.14 examples/s]39170 examples [00:43, 908.00 examples/s]39261 examples [00:44, 866.34 examples/s]39354 examples [00:44, 883.27 examples/s]39446 examples [00:44, 891.90 examples/s]39540 examples [00:44, 903.43 examples/s]39631 examples [00:44, 898.02 examples/s]39721 examples [00:44, 890.78 examples/s]39813 examples [00:44, 897.70 examples/s]39903 examples [00:44, 887.49 examples/s]39992 examples [00:44, 858.78 examples/s]40079 examples [00:45, 816.99 examples/s]40168 examples [00:45, 836.18 examples/s]40263 examples [00:45, 865.52 examples/s]40358 examples [00:45, 886.72 examples/s]40455 examples [00:45, 908.46 examples/s]40549 examples [00:45, 916.88 examples/s]40642 examples [00:45, 897.62 examples/s]40733 examples [00:45, 877.83 examples/s]40822 examples [00:45, 869.68 examples/s]40913 examples [00:45, 878.69 examples/s]41009 examples [00:46, 899.08 examples/s]41108 examples [00:46, 923.55 examples/s]41202 examples [00:46, 928.03 examples/s]41301 examples [00:46, 944.15 examples/s]41396 examples [00:46, 942.01 examples/s]41491 examples [00:46, 920.99 examples/s]41584 examples [00:46, 922.83 examples/s]41677 examples [00:46, 905.02 examples/s]41769 examples [00:46, 907.64 examples/s]41865 examples [00:46, 919.93 examples/s]41958 examples [00:47, 910.93 examples/s]42050 examples [00:47, 886.65 examples/s]42139 examples [00:47, 882.50 examples/s]42234 examples [00:47, 899.91 examples/s]42333 examples [00:47, 923.45 examples/s]42426 examples [00:47, 922.23 examples/s]42520 examples [00:47, 926.76 examples/s]42613 examples [00:47, 900.05 examples/s]42704 examples [00:47, 876.69 examples/s]42797 examples [00:48, 889.42 examples/s]42887 examples [00:48, 889.42 examples/s]42985 examples [00:48, 914.72 examples/s]43082 examples [00:48, 928.49 examples/s]43180 examples [00:48, 942.17 examples/s]43277 examples [00:48, 948.45 examples/s]43373 examples [00:48, 934.67 examples/s]43467 examples [00:48, 921.74 examples/s]43560 examples [00:48, 895.41 examples/s]43650 examples [00:48, 890.82 examples/s]43740 examples [00:49, 885.01 examples/s]43831 examples [00:49, 890.80 examples/s]43928 examples [00:49, 912.39 examples/s]44025 examples [00:49, 926.93 examples/s]44118 examples [00:49, 897.92 examples/s]44217 examples [00:49, 921.21 examples/s]44310 examples [00:49, 893.90 examples/s]44400 examples [00:49, 889.48 examples/s]44495 examples [00:49, 905.46 examples/s]44591 examples [00:49, 919.26 examples/s]44684 examples [00:50, 893.79 examples/s]44774 examples [00:50, 879.61 examples/s]44863 examples [00:50, 878.94 examples/s]44952 examples [00:50, 847.03 examples/s]45045 examples [00:50, 869.70 examples/s]45137 examples [00:50, 883.71 examples/s]45228 examples [00:50, 890.09 examples/s]45320 examples [00:50, 897.44 examples/s]45410 examples [00:50, 889.43 examples/s]45500 examples [00:51, 865.01 examples/s]45589 examples [00:51, 871.86 examples/s]45682 examples [00:51, 888.40 examples/s]45780 examples [00:51, 912.23 examples/s]45872 examples [00:51, 908.79 examples/s]45964 examples [00:51, 901.81 examples/s]46055 examples [00:51, 893.25 examples/s]46148 examples [00:51, 901.27 examples/s]46239 examples [00:51, 896.16 examples/s]46329 examples [00:51, 891.16 examples/s]46419 examples [00:52, 888.97 examples/s]46510 examples [00:52, 893.33 examples/s]46601 examples [00:52, 898.21 examples/s]46694 examples [00:52, 907.30 examples/s]46786 examples [00:52, 910.65 examples/s]46880 examples [00:52, 917.07 examples/s]46979 examples [00:52, 937.08 examples/s]47073 examples [00:52, 924.95 examples/s]47166 examples [00:52, 922.54 examples/s]47259 examples [00:52, 912.70 examples/s]47355 examples [00:53, 924.34 examples/s]47450 examples [00:53, 931.82 examples/s]47549 examples [00:53, 946.27 examples/s]47648 examples [00:53, 957.32 examples/s]47744 examples [00:53, 952.13 examples/s]47840 examples [00:53, 945.65 examples/s]47935 examples [00:53, 925.27 examples/s]48028 examples [00:53, 924.03 examples/s]48121 examples [00:53, 919.15 examples/s]48221 examples [00:53, 939.93 examples/s]48316 examples [00:54, 924.40 examples/s]48410 examples [00:54, 926.83 examples/s]48506 examples [00:54, 935.71 examples/s]48600 examples [00:54, 935.22 examples/s]48694 examples [00:54, 926.78 examples/s]48788 examples [00:54, 929.22 examples/s]48881 examples [00:54, 927.16 examples/s]48974 examples [00:54, 891.25 examples/s]49064 examples [00:54, 867.24 examples/s]49157 examples [00:55, 883.70 examples/s]49249 examples [00:55, 892.45 examples/s]49339 examples [00:55, 867.24 examples/s]49440 examples [00:55, 904.25 examples/s]49532 examples [00:55, 899.53 examples/s]49624 examples [00:55, 903.28 examples/s]49716 examples [00:55, 907.57 examples/s]49807 examples [00:55, 894.01 examples/s]49897 examples [00:55, 884.36 examples/s]49986 examples [00:55, 880.14 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6462/50000 [00:00<00:00, 64619.14 examples/s] 36%|███▌      | 17995/50000 [00:00<00:00, 74438.06 examples/s] 59%|█████▉    | 29674/50000 [00:00<00:00, 83523.78 examples/s] 84%|████████▍ | 42122/50000 [00:00<00:00, 92670.64 examples/s]                                                               0 examples [00:00, ? examples/s]75 examples [00:00, 744.99 examples/s]176 examples [00:00, 807.28 examples/s]268 examples [00:00, 836.11 examples/s]357 examples [00:00, 850.29 examples/s]450 examples [00:00, 870.25 examples/s]541 examples [00:00, 881.79 examples/s]640 examples [00:00, 910.68 examples/s]735 examples [00:00, 920.39 examples/s]833 examples [00:00, 934.94 examples/s]927 examples [00:01, 935.94 examples/s]1026 examples [00:01, 950.62 examples/s]1121 examples [00:01, 948.72 examples/s]1216 examples [00:01, 923.01 examples/s]1309 examples [00:01, 924.25 examples/s]1411 examples [00:01, 949.90 examples/s]1514 examples [00:01, 971.00 examples/s]1613 examples [00:01, 974.41 examples/s]1711 examples [00:01, 956.18 examples/s]1807 examples [00:01, 928.98 examples/s]1905 examples [00:02, 942.74 examples/s]2004 examples [00:02, 955.37 examples/s]2102 examples [00:02, 962.08 examples/s]2204 examples [00:02, 978.12 examples/s]2302 examples [00:02, 953.37 examples/s]2398 examples [00:02, 941.28 examples/s]2494 examples [00:02, 946.15 examples/s]2590 examples [00:02, 947.69 examples/s]2685 examples [00:02, 943.97 examples/s]2782 examples [00:02, 951.00 examples/s]2880 examples [00:03, 957.19 examples/s]2979 examples [00:03, 965.80 examples/s]3076 examples [00:03, 966.01 examples/s]3173 examples [00:03, 951.79 examples/s]3269 examples [00:03, 921.77 examples/s]3362 examples [00:03, 889.39 examples/s]3454 examples [00:03, 896.25 examples/s]3548 examples [00:03, 908.82 examples/s]3640 examples [00:03, 907.19 examples/s]3736 examples [00:03, 921.16 examples/s]3829 examples [00:04, 923.34 examples/s]3924 examples [00:04, 928.67 examples/s]4019 examples [00:04, 932.77 examples/s]4113 examples [00:04, 913.28 examples/s]4205 examples [00:04, 899.15 examples/s]4301 examples [00:04, 913.76 examples/s]4398 examples [00:04, 928.72 examples/s]4495 examples [00:04, 940.02 examples/s]4592 examples [00:04, 945.04 examples/s]4687 examples [00:05, 941.26 examples/s]4782 examples [00:05, 896.14 examples/s]4877 examples [00:05, 909.11 examples/s]4969 examples [00:05, 904.53 examples/s]5060 examples [00:05, 895.84 examples/s]5153 examples [00:05, 905.01 examples/s]5248 examples [00:05, 916.16 examples/s]5340 examples [00:05, 892.98 examples/s]5430 examples [00:05, 888.92 examples/s]5521 examples [00:05, 893.99 examples/s]5616 examples [00:06, 908.78 examples/s]5708 examples [00:06, 911.01 examples/s]5800 examples [00:06, 908.96 examples/s]5891 examples [00:06, 899.30 examples/s]5982 examples [00:06, 896.09 examples/s]6078 examples [00:06, 912.99 examples/s]6170 examples [00:06, 900.67 examples/s]6277 examples [00:06, 943.59 examples/s]6372 examples [00:06, 939.16 examples/s]6475 examples [00:06, 963.10 examples/s]6576 examples [00:07, 973.98 examples/s]6674 examples [00:07, 970.02 examples/s]6772 examples [00:07, 968.50 examples/s]6870 examples [00:07, 952.13 examples/s]6966 examples [00:07, 946.94 examples/s]7064 examples [00:07, 955.09 examples/s]7166 examples [00:07, 972.07 examples/s]7264 examples [00:07, 973.31 examples/s]7362 examples [00:07, 963.80 examples/s]7459 examples [00:07, 952.69 examples/s]7555 examples [00:08, 948.36 examples/s]7650 examples [00:08, 928.79 examples/s]7744 examples [00:08, 914.25 examples/s]7836 examples [00:08, 903.37 examples/s]7930 examples [00:08, 910.90 examples/s]8024 examples [00:08, 917.69 examples/s]8116 examples [00:08, 905.06 examples/s]8207 examples [00:08, 878.61 examples/s]8298 examples [00:08, 885.64 examples/s]8396 examples [00:09, 911.55 examples/s]8488 examples [00:09, 911.88 examples/s]8580 examples [00:09, 886.29 examples/s]8677 examples [00:09, 907.87 examples/s]8769 examples [00:09, 905.05 examples/s]8860 examples [00:09, 895.69 examples/s]8951 examples [00:09, 898.11 examples/s]9044 examples [00:09, 906.46 examples/s]9135 examples [00:09, 900.47 examples/s]9226 examples [00:09, 882.36 examples/s]9319 examples [00:10, 894.45 examples/s]9414 examples [00:10, 910.04 examples/s]9512 examples [00:10, 929.63 examples/s]9610 examples [00:10, 943.17 examples/s]9705 examples [00:10, 926.17 examples/s]9798 examples [00:10, 923.02 examples/s]9896 examples [00:10, 937.63 examples/s]9991 examples [00:10, 939.59 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWLKH2U/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWLKH2U/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3dc98c0a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3dc98c0a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3dc98c0a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3dc984bb00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3d549072e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3dc98c0a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3dc98c0a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3dc98c0a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3db1772358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3db17721d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.48 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.20 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.20 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.20 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.15 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.15 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.41 file/s]2020-07-31 00:10:25.284625: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-31 00:10:25.289715: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-31 00:10:25.290597: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f552558690 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-31 00:10:25.290615: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 146033.31it/s] 36%|███▌      | 3530752/9912422 [00:00<00:30, 208227.88it/s]9920512it [00:00, 32641414.52it/s]                           
0it [00:00, ?it/s]32768it [00:00, 556883.93it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161098.53it/s]1654784it [00:00, 11135569.99it/s]                         
0it [00:00, ?it/s]8192it [00:00, 181471.10it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
