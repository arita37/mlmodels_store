
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f90f15ac9d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f90f15ac9d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f915c217510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f915c217510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f917b466ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f917b466ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f910954a950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f910954a950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f910954a950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:20, 123383.59it/s] 49%|████▉     | 4833280/9912422 [00:00<00:28, 176055.64it/s]9920512it [00:00, 31086980.16it/s]                           
0it [00:00, ?it/s]32768it [00:00, 546746.52it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 979279.06it/s]1654784it [00:00, 12369971.22it/s]                          
0it [00:00, ?it/s]8192it [00:00, 231047.31it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9107a965f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9107aba898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f910954a598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f910954a598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f910954a598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:33,  1.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:32,  1.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:32,  1.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:31,  1.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:31,  1.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:30,  1.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:03,  2.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:03,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:02,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:02,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:02,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:01,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:01,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:00,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:00,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:59,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:41,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:41,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:41,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:40,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:40,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:40,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:40,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:39,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:39,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:27,  4.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:27,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:27,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:26,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:26,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:26,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:26,  4.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:17,  6.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:11,  9.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 12.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 22.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 29.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 29.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 37.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 45.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 45.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 54.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 54.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 62.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 79.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.30 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.30 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.49 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.57s/ url]
0 examples [00:00, ? examples/s]2020-07-21 18:09:32.482668: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-21 18:09:32.540857: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-07-21 18:09:32.541207: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bb0cf3a3f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-21 18:09:32.541230: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  6.69 examples/s]107 examples [00:00,  9.54 examples/s]214 examples [00:00, 13.57 examples/s]326 examples [00:00, 19.29 examples/s]434 examples [00:00, 27.34 examples/s]539 examples [00:00, 38.63 examples/s]644 examples [00:00, 54.32 examples/s]756 examples [00:00, 76.01 examples/s]867 examples [00:00, 105.46 examples/s]975 examples [00:01, 144.56 examples/s]1086 examples [00:01, 195.52 examples/s]1195 examples [00:01, 259.23 examples/s]1304 examples [00:01, 335.90 examples/s]1411 examples [00:01, 421.15 examples/s]1518 examples [00:01, 514.19 examples/s]1627 examples [00:01, 610.44 examples/s]1734 examples [00:01, 698.42 examples/s]1842 examples [00:01, 780.62 examples/s]1949 examples [00:01, 840.06 examples/s]2060 examples [00:02, 905.16 examples/s]2167 examples [00:02, 937.98 examples/s]2280 examples [00:02, 987.16 examples/s]2393 examples [00:02, 1023.67 examples/s]2502 examples [00:02, 1036.65 examples/s]2614 examples [00:02, 1059.16 examples/s]2724 examples [00:02, 1061.56 examples/s]2833 examples [00:02, 1041.16 examples/s]2945 examples [00:02, 1055.92 examples/s]3053 examples [00:02, 1061.81 examples/s]3162 examples [00:03, 1067.45 examples/s]3272 examples [00:03, 1075.45 examples/s]3382 examples [00:03, 1081.53 examples/s]3493 examples [00:03, 1089.04 examples/s]3603 examples [00:03, 1028.64 examples/s]3707 examples [00:03, 1030.04 examples/s]3814 examples [00:03, 1040.29 examples/s]3926 examples [00:03, 1061.26 examples/s]4035 examples [00:03, 1067.87 examples/s]4143 examples [00:04, 1046.98 examples/s]4255 examples [00:04, 1067.46 examples/s]4363 examples [00:04, 1033.03 examples/s]4475 examples [00:04, 1055.29 examples/s]4587 examples [00:04, 1072.92 examples/s]4695 examples [00:04, 1066.76 examples/s]4802 examples [00:04, 1020.75 examples/s]4913 examples [00:04, 1044.19 examples/s]5027 examples [00:04, 1069.04 examples/s]5135 examples [00:04, 1025.42 examples/s]5243 examples [00:05, 1040.62 examples/s]5348 examples [00:05, 1030.50 examples/s]5458 examples [00:05, 1047.67 examples/s]5567 examples [00:05, 1057.36 examples/s]5677 examples [00:05, 1069.39 examples/s]5787 examples [00:05, 1076.89 examples/s]5895 examples [00:05, 1075.13 examples/s]6006 examples [00:05, 1084.99 examples/s]6119 examples [00:05, 1097.64 examples/s]6229 examples [00:05, 1089.34 examples/s]6341 examples [00:06, 1097.87 examples/s]6451 examples [00:06, 1073.72 examples/s]6561 examples [00:06, 1079.12 examples/s]6672 examples [00:06, 1086.88 examples/s]6783 examples [00:06, 1092.78 examples/s]6893 examples [00:06, 1094.01 examples/s]7004 examples [00:06, 1096.72 examples/s]7114 examples [00:06, 1073.94 examples/s]7222 examples [00:06, 1054.05 examples/s]7328 examples [00:07, 1039.40 examples/s]7433 examples [00:07, 1024.68 examples/s]7541 examples [00:07, 1039.26 examples/s]7655 examples [00:07, 1064.49 examples/s]7765 examples [00:07, 1074.03 examples/s]7879 examples [00:07, 1091.36 examples/s]7989 examples [00:07, 1069.49 examples/s]8097 examples [00:07, 1070.92 examples/s]8207 examples [00:07, 1079.21 examples/s]8316 examples [00:07, 1063.55 examples/s]8424 examples [00:08, 1066.84 examples/s]8531 examples [00:08, 1044.77 examples/s]8636 examples [00:08, 1044.11 examples/s]8747 examples [00:08, 1060.67 examples/s]8858 examples [00:08, 1074.05 examples/s]8966 examples [00:08, 1060.46 examples/s]9073 examples [00:08, 1056.97 examples/s]9187 examples [00:08, 1077.97 examples/s]9300 examples [00:08, 1091.12 examples/s]9413 examples [00:08, 1101.56 examples/s]9524 examples [00:09, 1072.31 examples/s]9632 examples [00:09, 1037.12 examples/s]9737 examples [00:09, 978.56 examples/s] 9841 examples [00:09, 994.77 examples/s]9944 examples [00:09, 1003.91 examples/s]10045 examples [00:09, 973.78 examples/s]10152 examples [00:09, 998.78 examples/s]10255 examples [00:09, 1006.10 examples/s]10357 examples [00:09, 995.24 examples/s] 10460 examples [00:10, 1002.88 examples/s]10568 examples [00:10, 1024.24 examples/s]10671 examples [00:10, 1015.66 examples/s]10781 examples [00:10, 1038.11 examples/s]10893 examples [00:10, 1060.56 examples/s]11007 examples [00:10, 1081.66 examples/s]11117 examples [00:10, 1085.17 examples/s]11226 examples [00:10, 1080.07 examples/s]11340 examples [00:10, 1095.33 examples/s]11451 examples [00:10, 1098.54 examples/s]11561 examples [00:11, 1092.10 examples/s]11671 examples [00:11, 1055.61 examples/s]11777 examples [00:11, 1024.59 examples/s]11887 examples [00:11, 1043.51 examples/s]11992 examples [00:11, 1042.37 examples/s]12097 examples [00:11, 1034.86 examples/s]12202 examples [00:11, 1039.10 examples/s]12309 examples [00:11, 1047.63 examples/s]12414 examples [00:11, 1047.16 examples/s]12519 examples [00:11, 1036.13 examples/s]12623 examples [00:12, 1019.67 examples/s]12726 examples [00:12, 1000.04 examples/s]12830 examples [00:12, 1010.59 examples/s]12938 examples [00:12, 1028.05 examples/s]13046 examples [00:12, 1041.56 examples/s]13157 examples [00:12, 1058.79 examples/s]13264 examples [00:12, 1058.47 examples/s]13373 examples [00:12, 1067.66 examples/s]13482 examples [00:12, 1073.90 examples/s]13590 examples [00:12, 1070.51 examples/s]13700 examples [00:13, 1077.25 examples/s]13809 examples [00:13, 1077.50 examples/s]13917 examples [00:13, 1077.90 examples/s]14025 examples [00:13, 1073.02 examples/s]14133 examples [00:13, 1075.07 examples/s]14241 examples [00:13, 1074.40 examples/s]14349 examples [00:13, 1014.82 examples/s]14455 examples [00:13, 1025.69 examples/s]14569 examples [00:13, 1055.03 examples/s]14680 examples [00:14, 1069.59 examples/s]14793 examples [00:14, 1085.75 examples/s]14902 examples [00:14, 1081.86 examples/s]15011 examples [00:14, 1082.32 examples/s]15122 examples [00:14, 1090.43 examples/s]15232 examples [00:14, 1079.16 examples/s]15345 examples [00:14, 1091.97 examples/s]15455 examples [00:14, 1086.12 examples/s]15564 examples [00:14, 1041.95 examples/s]15672 examples [00:14, 1051.55 examples/s]15783 examples [00:15, 1065.86 examples/s]15894 examples [00:15, 1077.75 examples/s]16003 examples [00:15, 1065.56 examples/s]16110 examples [00:15, 1060.75 examples/s]16217 examples [00:15, 1043.30 examples/s]16328 examples [00:15, 1060.39 examples/s]16437 examples [00:15, 1068.24 examples/s]16544 examples [00:15, 1017.90 examples/s]16648 examples [00:15, 1021.83 examples/s]16753 examples [00:15, 1027.44 examples/s]16858 examples [00:16, 1025.98 examples/s]16961 examples [00:16, 1023.91 examples/s]17068 examples [00:16, 1034.50 examples/s]17172 examples [00:16, 1018.44 examples/s]17274 examples [00:16, 1016.00 examples/s]17377 examples [00:16, 1019.43 examples/s]17482 examples [00:16, 1027.51 examples/s]17586 examples [00:16, 1030.81 examples/s]17690 examples [00:16, 1029.02 examples/s]17794 examples [00:16, 1031.85 examples/s]17900 examples [00:17, 1037.49 examples/s]18004 examples [00:17, 1026.88 examples/s]18107 examples [00:17, 1010.43 examples/s]18211 examples [00:17, 1017.68 examples/s]18318 examples [00:17, 1030.66 examples/s]18430 examples [00:17, 1054.78 examples/s]18541 examples [00:17, 1068.64 examples/s]18649 examples [00:17, 1069.22 examples/s]18757 examples [00:17, 1060.08 examples/s]18866 examples [00:17, 1068.69 examples/s]18973 examples [00:18, 1020.33 examples/s]19076 examples [00:18, 986.60 examples/s] 19176 examples [00:18, 977.19 examples/s]19275 examples [00:18, 966.92 examples/s]19380 examples [00:18, 988.36 examples/s]19480 examples [00:18, 967.12 examples/s]19588 examples [00:18, 998.40 examples/s]19695 examples [00:18, 1017.03 examples/s]19803 examples [00:18, 1034.12 examples/s]19909 examples [00:19, 1039.01 examples/s]20014 examples [00:19, 1001.59 examples/s]20124 examples [00:19, 1027.55 examples/s]20230 examples [00:19, 1036.66 examples/s]20335 examples [00:19, 1003.31 examples/s]20436 examples [00:19, 1000.69 examples/s]20540 examples [00:19, 1010.04 examples/s]20647 examples [00:19, 1026.06 examples/s]20751 examples [00:19, 1028.69 examples/s]20861 examples [00:19, 1049.04 examples/s]20972 examples [00:20, 1064.85 examples/s]21079 examples [00:20, 1051.94 examples/s]21188 examples [00:20, 1060.74 examples/s]21295 examples [00:20, 1026.64 examples/s]21404 examples [00:20, 1043.70 examples/s]21514 examples [00:20, 1059.64 examples/s]21622 examples [00:20, 1063.11 examples/s]21734 examples [00:20, 1077.49 examples/s]21842 examples [00:20, 1021.19 examples/s]21950 examples [00:21, 1037.96 examples/s]22058 examples [00:21, 1048.55 examples/s]22164 examples [00:21, 1045.28 examples/s]22269 examples [00:21, 1038.99 examples/s]22374 examples [00:21, 1031.26 examples/s]22483 examples [00:21, 1046.02 examples/s]22595 examples [00:21, 1065.01 examples/s]22708 examples [00:21, 1081.65 examples/s]22817 examples [00:21, 1080.20 examples/s]22926 examples [00:21, 1070.53 examples/s]23034 examples [00:22, 1051.12 examples/s]23140 examples [00:22, 1029.88 examples/s]23250 examples [00:22, 1048.93 examples/s]23357 examples [00:22, 1052.99 examples/s]23465 examples [00:22, 1059.36 examples/s]23572 examples [00:22, 1033.84 examples/s]23679 examples [00:22, 1043.81 examples/s]23784 examples [00:22, 1030.66 examples/s]23888 examples [00:22, 1014.97 examples/s]23994 examples [00:22, 1027.06 examples/s]24104 examples [00:23, 1045.86 examples/s]24209 examples [00:23, 1028.51 examples/s]24316 examples [00:23, 1038.54 examples/s]24421 examples [00:23, 1039.93 examples/s]24526 examples [00:23, 1033.22 examples/s]24638 examples [00:23, 1057.35 examples/s]24749 examples [00:23, 1069.49 examples/s]24859 examples [00:23, 1077.92 examples/s]24970 examples [00:23, 1084.98 examples/s]25079 examples [00:23, 1084.68 examples/s]25191 examples [00:24, 1092.74 examples/s]25301 examples [00:24, 1087.15 examples/s]25412 examples [00:24, 1092.84 examples/s]25522 examples [00:24, 1088.50 examples/s]25632 examples [00:24, 1091.47 examples/s]25743 examples [00:24, 1094.60 examples/s]25853 examples [00:24, 1051.75 examples/s]25959 examples [00:24, 1052.24 examples/s]26068 examples [00:24, 1062.09 examples/s]26178 examples [00:25, 1071.76 examples/s]26290 examples [00:25, 1084.71 examples/s]26399 examples [00:25, 1075.57 examples/s]26507 examples [00:25, 1063.58 examples/s]26614 examples [00:25, 1030.28 examples/s]26721 examples [00:25, 1040.08 examples/s]26831 examples [00:25, 1057.13 examples/s]26940 examples [00:25, 1065.74 examples/s]27047 examples [00:25, 1009.86 examples/s]27149 examples [00:25, 1011.14 examples/s]27255 examples [00:26, 1023.93 examples/s]27358 examples [00:26, 1022.07 examples/s]27466 examples [00:26, 1038.51 examples/s]27573 examples [00:26, 1046.77 examples/s]27681 examples [00:26, 1056.48 examples/s]27790 examples [00:26, 1065.48 examples/s]27899 examples [00:26, 1070.40 examples/s]28009 examples [00:26, 1076.65 examples/s]28117 examples [00:26, 1068.82 examples/s]28224 examples [00:26, 1054.42 examples/s]28330 examples [00:27, 1046.18 examples/s]28436 examples [00:27, 1049.13 examples/s]28545 examples [00:27, 1059.94 examples/s]28654 examples [00:27, 1068.72 examples/s]28761 examples [00:27, 1047.66 examples/s]28869 examples [00:27, 1057.03 examples/s]28975 examples [00:27, 1040.10 examples/s]29080 examples [00:27, 1014.04 examples/s]29182 examples [00:27, 1004.97 examples/s]29285 examples [00:27, 1010.40 examples/s]29391 examples [00:28, 1021.61 examples/s]29494 examples [00:28, 1004.48 examples/s]29598 examples [00:28, 1014.12 examples/s]29702 examples [00:28, 1020.29 examples/s]29805 examples [00:28, 1006.75 examples/s]29910 examples [00:28, 1017.35 examples/s]30012 examples [00:28, 977.79 examples/s] 30111 examples [00:28, 974.04 examples/s]30212 examples [00:28, 979.96 examples/s]30318 examples [00:29, 1000.03 examples/s]30419 examples [00:29, 991.21 examples/s] 30519 examples [00:29, 983.00 examples/s]30618 examples [00:29, 970.71 examples/s]30716 examples [00:29, 966.14 examples/s]30820 examples [00:29, 986.86 examples/s]30919 examples [00:29, 977.45 examples/s]31029 examples [00:29, 1010.79 examples/s]31134 examples [00:29, 1021.29 examples/s]31239 examples [00:29, 1029.17 examples/s]31344 examples [00:30, 1035.04 examples/s]31452 examples [00:30, 1047.94 examples/s]31558 examples [00:30, 1048.96 examples/s]31664 examples [00:30, 1033.63 examples/s]31768 examples [00:30, 1014.34 examples/s]31870 examples [00:30, 1015.33 examples/s]31972 examples [00:30, 1012.11 examples/s]32086 examples [00:30, 1045.53 examples/s]32197 examples [00:30, 1062.24 examples/s]32304 examples [00:30, 1061.54 examples/s]32411 examples [00:31, 1062.79 examples/s]32523 examples [00:31, 1077.06 examples/s]32631 examples [00:31, 1054.95 examples/s]32738 examples [00:31, 1055.80 examples/s]32844 examples [00:31, 1048.14 examples/s]32949 examples [00:31, 1030.89 examples/s]33057 examples [00:31, 1042.73 examples/s]33169 examples [00:31, 1064.51 examples/s]33280 examples [00:31, 1075.40 examples/s]33390 examples [00:31, 1080.30 examples/s]33499 examples [00:32, 1059.75 examples/s]33606 examples [00:32, 1054.48 examples/s]33717 examples [00:32, 1068.03 examples/s]33825 examples [00:32, 1070.83 examples/s]33934 examples [00:32, 1076.19 examples/s]34044 examples [00:32, 1081.67 examples/s]34153 examples [00:32, 1065.03 examples/s]34262 examples [00:32, 1071.83 examples/s]34370 examples [00:32, 1066.03 examples/s]34477 examples [00:33, 1044.11 examples/s]34582 examples [00:33, 1015.72 examples/s]34684 examples [00:33, 995.14 examples/s] 34784 examples [00:33, 966.96 examples/s]34888 examples [00:33, 986.73 examples/s]34988 examples [00:33, 982.76 examples/s]35087 examples [00:33, 943.28 examples/s]35183 examples [00:33, 947.11 examples/s]35279 examples [00:33, 941.44 examples/s]35391 examples [00:33, 985.96 examples/s]35501 examples [00:34, 1015.27 examples/s]35604 examples [00:34, 1018.45 examples/s]35713 examples [00:34, 1037.45 examples/s]35824 examples [00:34, 1057.80 examples/s]35936 examples [00:34, 1075.24 examples/s]36047 examples [00:34, 1082.87 examples/s]36156 examples [00:34, 1055.96 examples/s]36267 examples [00:34, 1070.73 examples/s]36375 examples [00:34, 1071.73 examples/s]36483 examples [00:34, 1061.95 examples/s]36590 examples [00:35, 1050.10 examples/s]36696 examples [00:35, 1047.42 examples/s]36803 examples [00:35, 1051.46 examples/s]36910 examples [00:35, 1054.76 examples/s]37016 examples [00:35, 1038.87 examples/s]37122 examples [00:35, 1041.35 examples/s]37227 examples [00:35, 1042.09 examples/s]37337 examples [00:35, 1056.96 examples/s]37443 examples [00:35, 1021.52 examples/s]37549 examples [00:35, 1030.56 examples/s]37655 examples [00:36, 1038.22 examples/s]37760 examples [00:36, 1037.29 examples/s]37864 examples [00:36, 1013.99 examples/s]37974 examples [00:36, 1037.75 examples/s]38079 examples [00:36, 1038.12 examples/s]38186 examples [00:36, 1045.02 examples/s]38291 examples [00:36, 1038.76 examples/s]38402 examples [00:36, 1057.08 examples/s]38508 examples [00:36, 1033.79 examples/s]38618 examples [00:37, 1052.42 examples/s]38724 examples [00:37, 1032.72 examples/s]38832 examples [00:37, 1044.14 examples/s]38944 examples [00:37, 1064.93 examples/s]39051 examples [00:37, 1018.75 examples/s]39160 examples [00:37, 1036.57 examples/s]39268 examples [00:37, 1046.44 examples/s]39378 examples [00:37, 1059.22 examples/s]39485 examples [00:37, 1062.09 examples/s]39594 examples [00:37, 1068.60 examples/s]39702 examples [00:38, 1064.37 examples/s]39809 examples [00:38, 1053.57 examples/s]39915 examples [00:38, 1045.33 examples/s]40020 examples [00:38, 974.14 examples/s] 40120 examples [00:38, 979.99 examples/s]40232 examples [00:38, 1015.79 examples/s]40335 examples [00:38, 1012.40 examples/s]40447 examples [00:38, 1042.01 examples/s]40559 examples [00:38, 1063.50 examples/s]40672 examples [00:38, 1080.41 examples/s]40783 examples [00:39, 1088.19 examples/s]40893 examples [00:39, 1067.29 examples/s]41006 examples [00:39, 1082.81 examples/s]41115 examples [00:39, 1082.33 examples/s]41229 examples [00:39, 1098.34 examples/s]41340 examples [00:39, 1079.75 examples/s]41449 examples [00:39, 1069.05 examples/s]41559 examples [00:39, 1077.82 examples/s]41667 examples [00:39, 1073.22 examples/s]41779 examples [00:40, 1085.96 examples/s]41890 examples [00:40, 1092.58 examples/s]42000 examples [00:40, 1013.78 examples/s]42103 examples [00:40, 1001.58 examples/s]42213 examples [00:40, 1027.94 examples/s]42324 examples [00:40, 1048.35 examples/s]42430 examples [00:40, 1031.28 examples/s]42540 examples [00:40, 1042.06 examples/s]42645 examples [00:40, 1019.01 examples/s]42748 examples [00:40, 1014.43 examples/s]42854 examples [00:41, 1024.90 examples/s]42960 examples [00:41, 1034.32 examples/s]43065 examples [00:41, 1036.51 examples/s]43175 examples [00:41, 1053.83 examples/s]43281 examples [00:41, 1033.65 examples/s]43394 examples [00:41, 1060.56 examples/s]43501 examples [00:41, 1058.54 examples/s]43613 examples [00:41, 1074.82 examples/s]43725 examples [00:41, 1087.96 examples/s]43834 examples [00:41, 1041.84 examples/s]43943 examples [00:42, 1055.50 examples/s]44049 examples [00:42, 1053.50 examples/s]44155 examples [00:42, 1045.58 examples/s]44264 examples [00:42, 1055.57 examples/s]44370 examples [00:42, 1035.57 examples/s]44477 examples [00:42, 1044.30 examples/s]44582 examples [00:42, 1044.30 examples/s]44690 examples [00:42, 1054.52 examples/s]44796 examples [00:42, 1054.96 examples/s]44902 examples [00:43, 1051.22 examples/s]45009 examples [00:43, 1054.26 examples/s]45119 examples [00:43, 1065.91 examples/s]45226 examples [00:43, 1048.34 examples/s]45334 examples [00:43, 1054.90 examples/s]45443 examples [00:43, 1065.16 examples/s]45553 examples [00:43, 1074.80 examples/s]45661 examples [00:43, 1076.23 examples/s]45773 examples [00:43, 1086.88 examples/s]45882 examples [00:43, 1081.31 examples/s]45995 examples [00:44, 1092.94 examples/s]46106 examples [00:44, 1095.67 examples/s]46216 examples [00:44, 1073.93 examples/s]46324 examples [00:44, 1073.52 examples/s]46432 examples [00:44, 1057.72 examples/s]46540 examples [00:44, 1061.99 examples/s]46647 examples [00:44, 1035.03 examples/s]46755 examples [00:44, 1046.64 examples/s]46867 examples [00:44, 1065.04 examples/s]46974 examples [00:44, 1057.03 examples/s]47084 examples [00:45, 1067.51 examples/s]47196 examples [00:45, 1081.60 examples/s]47305 examples [00:45, 1082.38 examples/s]47414 examples [00:45, 1053.64 examples/s]47521 examples [00:45, 1056.85 examples/s]47627 examples [00:45, 1050.20 examples/s]47733 examples [00:45, 1038.31 examples/s]47837 examples [00:45, 1024.55 examples/s]47948 examples [00:45, 1047.03 examples/s]48055 examples [00:45, 1051.29 examples/s]48161 examples [00:46, 1049.21 examples/s]48269 examples [00:46, 1058.15 examples/s]48375 examples [00:46, 1037.89 examples/s]48479 examples [00:46, 1032.87 examples/s]48583 examples [00:46, 1023.72 examples/s]48690 examples [00:46, 1037.01 examples/s]48794 examples [00:46, 977.95 examples/s] 48897 examples [00:46, 991.33 examples/s]48997 examples [00:46, 993.05 examples/s]49102 examples [00:47, 1008.34 examples/s]49204 examples [00:47, 1005.01 examples/s]49305 examples [00:47, 1006.24 examples/s]49406 examples [00:47, 977.55 examples/s] 49505 examples [00:47, 970.34 examples/s]49612 examples [00:47, 997.26 examples/s]49715 examples [00:47, 1005.08 examples/s]49816 examples [00:47, 996.98 examples/s] 49916 examples [00:47, 991.50 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6488/50000 [00:00<00:00, 64876.35 examples/s] 40%|███▉      | 19981/50000 [00:00<00:00, 76844.62 examples/s] 67%|██████▋   | 33396/50000 [00:00<00:00, 88139.43 examples/s] 94%|█████████▎| 46808/50000 [00:00<00:00, 98241.81 examples/s]                                                               0 examples [00:00, ? examples/s]92 examples [00:00, 912.79 examples/s]201 examples [00:00, 958.28 examples/s]291 examples [00:00, 937.47 examples/s]389 examples [00:00, 949.80 examples/s]498 examples [00:00, 985.63 examples/s]608 examples [00:00, 1014.81 examples/s]716 examples [00:00, 1031.40 examples/s]824 examples [00:00, 1044.88 examples/s]935 examples [00:00, 1060.61 examples/s]1038 examples [00:01, 1043.55 examples/s]1145 examples [00:01, 1048.74 examples/s]1256 examples [00:01, 1063.97 examples/s]1362 examples [00:01, 1038.83 examples/s]1469 examples [00:01, 1047.96 examples/s]1574 examples [00:01, 1021.12 examples/s]1684 examples [00:01, 1041.18 examples/s]1793 examples [00:01, 1054.53 examples/s]1902 examples [00:01, 1062.51 examples/s]2009 examples [00:01, 1064.37 examples/s]2116 examples [00:02, 1059.72 examples/s]2222 examples [00:02, 1057.81 examples/s]2334 examples [00:02, 1073.54 examples/s]2442 examples [00:02, 1051.35 examples/s]2553 examples [00:02, 1066.14 examples/s]2660 examples [00:02, 1015.71 examples/s]2763 examples [00:02, 1005.96 examples/s]2870 examples [00:02, 1022.85 examples/s]2978 examples [00:02, 1037.32 examples/s]3092 examples [00:02, 1065.49 examples/s]3200 examples [00:03, 1069.34 examples/s]3314 examples [00:03, 1087.31 examples/s]3424 examples [00:03, 1083.78 examples/s]3533 examples [00:03, 1081.20 examples/s]3647 examples [00:03, 1096.09 examples/s]3757 examples [00:03, 1072.64 examples/s]3868 examples [00:03, 1082.23 examples/s]3977 examples [00:03, 1083.20 examples/s]4090 examples [00:03, 1094.20 examples/s]4200 examples [00:03, 1051.39 examples/s]4306 examples [00:04, 1044.16 examples/s]4415 examples [00:04, 1055.96 examples/s]4521 examples [00:04, 1054.17 examples/s]4627 examples [00:04, 1035.64 examples/s]4735 examples [00:04, 1047.16 examples/s]4840 examples [00:04, 1045.06 examples/s]4945 examples [00:04, 1033.83 examples/s]5051 examples [00:04, 1040.69 examples/s]5157 examples [00:04, 1046.36 examples/s]5272 examples [00:05, 1074.19 examples/s]5380 examples [00:05, 1066.20 examples/s]5492 examples [00:05, 1079.94 examples/s]5601 examples [00:05, 1078.18 examples/s]5712 examples [00:05, 1086.95 examples/s]5821 examples [00:05, 1066.57 examples/s]5930 examples [00:05, 1070.60 examples/s]6041 examples [00:05, 1079.73 examples/s]6150 examples [00:05, 1041.15 examples/s]6255 examples [00:05, 1011.88 examples/s]6358 examples [00:06, 1016.58 examples/s]6460 examples [00:06, 1007.72 examples/s]6571 examples [00:06, 1034.44 examples/s]6675 examples [00:06, 1003.27 examples/s]6776 examples [00:06, 990.56 examples/s] 6887 examples [00:06, 1021.89 examples/s]6999 examples [00:06, 1047.87 examples/s]7113 examples [00:06, 1073.15 examples/s]7221 examples [00:06, 1072.69 examples/s]7329 examples [00:06, 1028.38 examples/s]7435 examples [00:07, 1035.90 examples/s]7545 examples [00:07, 1053.81 examples/s]7655 examples [00:07, 1065.01 examples/s]7762 examples [00:07, 1064.26 examples/s]7869 examples [00:07, 1055.21 examples/s]7976 examples [00:07, 1057.88 examples/s]8083 examples [00:07, 1059.16 examples/s]8189 examples [00:07, 1007.94 examples/s]8297 examples [00:07, 1027.47 examples/s]8409 examples [00:08, 1053.33 examples/s]8518 examples [00:08, 1063.81 examples/s]8628 examples [00:08, 1072.50 examples/s]8736 examples [00:08, 1054.04 examples/s]8849 examples [00:08, 1074.30 examples/s]8961 examples [00:08, 1086.54 examples/s]9070 examples [00:08, 1056.99 examples/s]9183 examples [00:08, 1075.88 examples/s]9293 examples [00:08, 1080.59 examples/s]9406 examples [00:08, 1093.41 examples/s]9516 examples [00:09, 1089.47 examples/s]9626 examples [00:09, 1065.08 examples/s]9739 examples [00:09, 1080.34 examples/s]9848 examples [00:09, 1005.88 examples/s]9952 examples [00:09, 1014.92 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete06JOYU/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete06JOYU/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f910954a950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f910954a950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f910954a950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f90c8695dd8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f90c8699438>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f910954a950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f910954a950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f910954a950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f90c8699e48>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9107aba4a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f908784ce18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f908784ce18> 

  function with postional parmater data_info <function split_train_valid at 0x7f908784ce18> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f908784cf28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f908784cf28> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f908784cf28> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=baf309c524dff4bba9d3bfbfc5f3b7c32337f32c4bb03455727e13dad4244eee
  Stored in directory: /tmp/pip-ephem-wheel-cache-0d8i5sw2/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:03:40, 9.95kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:05:04, 14.0kB/s].vector_cache/glove.6B.zip:   0%|          | 180k/862M [00:01<12:01:52, 19.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 762k/862M [00:01<8:26:00, 28.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.07M/862M [00:01<5:53:30, 40.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.27M/862M [00:01<4:05:45, 57.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.5M/862M [00:01<2:51:04, 82.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.3M/862M [00:01<1:59:19, 118kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.7M/862M [00:01<1:23:04, 168kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.9M/862M [00:01<58:04, 240kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.6M/862M [00:02<40:30, 342kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:02<28:22, 486kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:02<19:53, 690kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<13:56, 979kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.4M/862M [00:02<09:47, 1.39MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:03<07:24, 1.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<07:04, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<06:50, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<05:15, 2.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<06:10, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<07:03, 1.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:07<05:31, 2.41MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:07<04:02, 3.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<09:15, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<07:37, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:09<05:38, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:09<04:06, 3.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<56:09, 236kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<40:38, 325kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<28:40, 460kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<23:09, 568kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<17:32, 749kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:13<12:32, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<11:52, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<09:39, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:15<07:01, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<08:01, 1.62MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<06:56, 1.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:17<05:11, 2.50MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<06:41, 1.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<07:20, 1.76MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<05:42, 2.27MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:19<04:08, 3.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<10:59, 1.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<09:00, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:20<06:35, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<07:36, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<07:56, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<06:12, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<06:23, 1.99MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<05:47, 2.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:24<04:22, 2.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<03:46, 3.37MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<9:46:01, 21.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<6:50:30, 30.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<4:46:20, 44.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<3:47:13, 55.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<2:41:37, 78.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<1:53:33, 111kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<1:19:23, 158kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:04:53, 193kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<46:43, 268kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<32:57, 379kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<25:54, 481kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<19:25, 642kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<13:53, 896kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<12:37, 982kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:07, 1.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<07:20, 1.69MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:04, 1.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:12, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:22, 1.93MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:25, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:47, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:22, 2.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:52, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:45, 1.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:15, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:51, 3.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<07:34, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:33, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:53, 2.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:14, 1.93MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:23, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<04:05, 2.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<03:00, 3.99MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<50:22, 238kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<36:28, 329kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<25:47, 464kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<20:48, 573kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<15:47, 755kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<11:20, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<10:42, 1.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:43, 1.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:20, 1.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<07:14, 1.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<07:36, 1.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:56, 1.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<04:16, 2.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<45:09, 259kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<32:50, 357kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<23:14, 503kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<18:55, 616kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<14:13, 819kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<10:21, 1.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<07:21, 1.57MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<48:55, 237kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<36:40, 316kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<26:14, 441kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<18:24, 625kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<11:16:44, 17.0kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<7:54:39, 24.2kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<5:31:49, 34.6kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<3:54:13, 48.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<2:45:05, 69.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<1:55:36, 98.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<1:23:18, 137kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<1:00:45, 187kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<43:05, 264kB/s]  .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<30:10, 375kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:41:22, 112kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:12:07, 157kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<50:37, 223kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<37:57, 296kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<27:31, 408kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<19:28, 576kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<13:44, 813kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<54:37, 204kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<39:22, 284kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<27:46, 401kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<22:00, 504kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<17:42, 626kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<12:51, 862kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<09:12, 1.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<09:29, 1.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:47, 1.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<05:41, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<04:38, 2.36MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<8:22:39, 21.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<5:51:56, 31.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<4:05:46, 44.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<2:55:56, 61.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<2:05:23, 86.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<1:28:12, 123kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<1:03:12, 171kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<45:24, 238kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<31:59, 337kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<24:47, 434kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<19:33, 550kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<14:13, 755kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<10:01, 1.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<10:24:40, 17.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<7:18:06, 24.4kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<5:06:12, 34.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<3:36:06, 49.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<2:33:24, 69.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<1:47:48, 98.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<1:15:13, 140kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<2:19:28, 75.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<1:38:40, 107kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<1:09:10, 152kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<50:41, 207kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<36:31, 287kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<25:46, 406kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<20:26, 509kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<15:23, 677kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<11:00, 943kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<10:07, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:09, 1.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:57, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<06:35, 1.56MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:40, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:11, 2.44MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:21, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:47, 2.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:36, 2.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:54, 2.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:29, 2.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:23, 2.98MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:45, 2.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:21, 2.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:18, 3.04MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:40, 2.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<04:17, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:12, 3.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:36, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<05:10, 1.92MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:07, 2.40MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<02:58, 3.32MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<1:14:01, 133kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<52:49, 187kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<37:05, 265kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<28:09, 348kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<20:41, 473kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<14:42, 664kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<12:33, 775kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<09:46, 994kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<07:02, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<07:12, 1.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<06:57, 1.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:17, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:49, 2.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<08:06, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:41, 1.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:55, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:39, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:54, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:33, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:17, 2.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<11:09, 848kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<08:46, 1.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<06:20, 1.49MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:37, 1.42MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:33, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:03, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:02, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:28, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:22, 2.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<02:53, 3.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<7:02:47, 21.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<4:56:02, 31.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<3:26:11, 44.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<2:38:23, 58.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<1:52:42, 81.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<1:19:16, 116kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<56:39, 161kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<40:35, 224kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<28:32, 318kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<21:59, 411kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<17:19, 522kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<12:30, 722kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<08:50, 1.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<11:19, 793kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:51, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:23, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:30, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:23, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:51, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<03:31, 2.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<06:06, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:11, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:49, 2.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:43, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:00, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:45, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:16, 2.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:53, 2.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:36, 2.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:23, 2.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:13, 2.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:04, 2.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<02:56, 2.96MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<02:50, 3.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<40:56, 213kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<30:47, 283kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<22:23, 388kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<16:17, 533kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<12:15, 708kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<09:10, 944kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<07:07, 1.22MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:52, 1.47MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:42, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:59, 2.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:40, 2.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<13:56, 619kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<11:22, 758kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<08:34, 1.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<07:02, 1.22MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<05:36, 1.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:36, 1.87MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:53, 2.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<03:14, 2.65MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:14, 2.65MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<02:55, 2.93MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<07:25, 1.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<08:43, 982kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<07:06, 1.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:38, 1.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:31, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:49, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:26, 2.48MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:03, 2.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<02:40, 3.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<02:31, 3.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<1:02:10, 137kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<45:15, 188kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<32:23, 262kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<23:19, 364kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<16:57, 499kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<12:23, 683kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<09:19, 907kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<07:01, 1.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<06:00, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:43, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<32:18, 261kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<26:06, 323kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<19:15, 437kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<14:00, 600kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<10:22, 810kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<08:00, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<06:08, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:05, 1.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:12, 1.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:36, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:05, 2.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<02:54, 2.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<49:45, 168kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<36:32, 229kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<26:09, 319kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<18:57, 440kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<13:49, 602kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<10:11, 817kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<07:42, 1.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<05:58, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:46, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<4:37:50, 29.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<3:17:35, 42.0kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<2:19:01, 59.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<1:37:51, 84.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<1:09:01, 120kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<48:50, 169kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<34:43, 238kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<24:51, 331kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<17:55, 459kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<21:40, 379kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<18:03, 455kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<13:30, 608kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<10:04, 814kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:34, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:48, 1.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<04:29, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:39, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:16, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<07:47, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:12, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:44, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:39, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:12, 2.53MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:41, 3.01MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<02:15, 3.58MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<07:40, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<07:44, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<06:01, 1.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:33, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:38, 2.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<02:52, 2.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<02:20, 3.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<07:29, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<09:10, 874kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<07:24, 1.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:36, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<04:12, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:18, 2.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<02:34, 3.08MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<1:56:30, 68.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<1:25:05, 93.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<1:00:22, 131kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<42:28, 186kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<29:56, 264kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<21:11, 372kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<21:04, 373kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<17:31, 449kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<12:52, 611kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<09:18, 843kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<06:45, 1.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<07:17, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<10:34, 738kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<08:40, 899kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:22, 1.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:40, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<03:30, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<33:00, 234kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<25:21, 305kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<18:17, 422kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<12:58, 593kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<09:14, 830kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<15:15, 503kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<14:30, 528kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<11:09, 686kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<08:03, 948kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:47, 1.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<08:02, 945kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<09:01, 842kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<07:10, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:15, 1.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:48, 1.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<08:49, 853kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<09:15, 813kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<07:14, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:16, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<03:48, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<1:40:55, 73.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<1:13:26, 102kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<52:03, 143kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<36:29, 203kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<25:32, 289kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<34:06, 217kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<26:28, 279kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<19:02, 388kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<13:27, 546kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<11:11, 655kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<10:04, 727kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<07:30, 973kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:24, 1.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<05:48, 1.25MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:10, 1.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:49, 1.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:30, 2.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:37, 1.55MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<05:11, 1.38MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:06, 1.74MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:59, 2.39MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:29, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:54, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:48, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:47, 2.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:54, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:22, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:27, 2.03MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:31, 2.77MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<05:28, 1.27MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<05:19, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:01, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:55, 2.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:41, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:03, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:58, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:52, 2.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:57, 2.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<4:44:47, 24.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<3:20:03, 34.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<2:19:48, 48.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<1:38:39, 68.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<1:10:17, 96.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<49:25, 137kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<34:26, 195kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<43:17, 155kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<31:57, 210kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<22:44, 294kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<15:55, 417kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<15:18, 434kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<11:49, 560kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<08:32, 775kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<07:04, 928kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<05:58, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<04:22, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<03:08, 2.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<06:01, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<05:36, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:12, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<03:00, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<06:30, 987kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:28, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:03, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:00, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:59, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:05, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:11, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:26, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:42, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:54, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:15, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:31, 2.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<01:53, 3.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:09, 1.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:25, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:38, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<01:57, 3.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:22, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:30, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:44, 2.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:54, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:10, 1.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:27, 2.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<01:49, 3.26MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:11, 1.86MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:24, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:37, 2.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:53, 3.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<12:30, 470kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<09:54, 593kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<07:11, 814kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<05:57, 975kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<05:15, 1.10MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:57, 1.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:48, 2.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<20:35, 278kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<15:54, 360kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<11:25, 501kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<08:06, 703kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<06:56, 816kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:47, 977kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:16, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:56, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:40, 1.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:48, 1.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:54, 1.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:56, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:16, 2.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:32, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:40, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:05, 2.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:53<01:30, 3.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<2:38:46, 33.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<1:51:56, 48.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<1:18:18, 68.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<54:28, 97.8kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<42:51, 124kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<30:50, 172kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<21:43, 244kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<15:09, 347kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<14:32, 361kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<11:01, 476kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<07:52, 664kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<05:32, 937kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<09:33, 542kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<07:34, 684kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<05:27, 945kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<03:52, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<05:37, 909kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<04:48, 1.06MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:31, 1.44MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:31, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<04:58, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:20, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:12, 1.57MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:17, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<05:38, 881kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:47, 1.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:33, 1.39MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:19, 1.48MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:09, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:24, 2.03MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:30, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:34, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<01:58, 2.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:25, 3.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:58, 958kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:17, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:09, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:15, 2.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<05:27, 862kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:29, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:20, 1.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:23, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:15, 1.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:45, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:47, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:59, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<06:18, 723kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:11, 878kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<03:47, 1.20MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:40, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<06:02, 744kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<04:58, 901kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<03:38, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<02:35, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:36, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<3:13:31, 22.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<2:15:36, 32.6kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<1:34:19, 46.5kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<1:06:52, 65.2kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<48:20, 90.2kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<34:08, 127kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<23:48, 181kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<17:47, 241kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<13:08, 326kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<09:20, 457kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<07:12, 586kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<05:43, 736kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:10, 1.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:36, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:12, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:24, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:22, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:19, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:47, 2.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:56, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:01, 1.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:32, 2.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:34<01:07, 3.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<04:10, 944kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:35, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:40, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:31, 1.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:25, 1.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:49, 2.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:20, 2.86MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<02:21, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:17, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:44, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:15, 2.98MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:42, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:32, 1.47MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:54, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:57, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:59, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:32, 2.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:41, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:47, 2.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:22, 2.59MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<00:59, 3.56MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<13:00, 272kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<09:41, 364kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<06:53, 509kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<05:21, 646kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<04:58, 696kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<03:45, 918kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:41, 1.28MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:50, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:33, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:55, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:54, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:53, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:27, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:34, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:39, 1.97MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:15, 2.56MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<00:55, 3.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:50, 1.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:31, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:52, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:20, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<03:32, 881kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<03:00, 1.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:12, 1.41MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<01:33, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<06:44, 453kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<05:12, 585kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:44, 811kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<02:36, 1.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<05:56, 502kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<04:38, 642kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<03:20, 887kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<02:20, 1.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<08:17, 352kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<06:16, 464kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<04:29, 645kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<03:35, 794kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:58, 956kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:10, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:31, 1.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<05:31, 503kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<04:18, 643kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:05, 889kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:10, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<05:20, 507kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<04:10, 648kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<03:00, 891kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:32, 1.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:13, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:38, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:09, 2.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:52, 896kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:26, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:47, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:15, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<03:58, 630kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<03:12, 780kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<02:19, 1.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<01:37, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<02:09, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<1:46:02, 23.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<1:14:09, 32.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<51:23, 46.7kB/s]  .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<36:03, 65.6kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<26:03, 90.8kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<18:22, 128kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<12:44, 183kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<09:28, 243kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<07:00, 328kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<04:57, 459kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<03:54, 570kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<04:07, 541kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<03:16, 680kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:21, 938kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:39, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<02:01, 1.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<01:47, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:18, 1.63MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:56, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:58, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:43, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:16, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:54, 2.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:12, 916kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:53, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:23, 1.44MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:17, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:13, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<00:56, 2.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:58, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:59, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:45, 2.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:32, 3.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:36, 1.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:25, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:03, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:44, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:52, 936kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:36, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:10, 1.47MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:49, 2.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:50, 915kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:34, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:08, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:49, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:11, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:05, 1.47MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:49, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:49, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:50, 1.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:38, 2.36MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:41, 2.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:40, 2.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:30, 2.85MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:37, 2.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:40, 2.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:31, 2.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:35, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:37, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:28, 2.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:21, 3.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:46, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:44, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:34, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:35, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:36, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:28, 2.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:30, 2.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:32, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:25, 2.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:28, 2.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:30, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:23, 2.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:16, 3.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:45, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:41, 1.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:30, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:21, 2.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:48, 1.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:43, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:31, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:21, 2.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:25, 597kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:08, 744kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:49, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:40, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:36, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:26, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:24, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:24, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:18, 2.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:12, 3.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:48, 802kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:40, 959kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:29, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:24, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:23, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:16, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:11, 2.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:34, 877kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:29, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:20, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:14, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:20, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:18, 1.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:13, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:12, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:12, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:09, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:05, 3.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:17, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:15, 1.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:10, 1.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:13, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:11, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:08, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:05, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:03, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:01, 3.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:19, 298kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:13, 398kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:08, 558kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:03, 786kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:02, 733kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:01, 893kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.21MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<152:31:54,  1.37s/it]  0%|          | 773/400000 [00:01<106:34:14,  1.04it/s]  0%|          | 1616/400000 [00:01<74:26:44,  1.49it/s]  1%|          | 2498/400000 [00:01<52:00:01,  2.12it/s]  1%|          | 3394/400000 [00:01<36:19:19,  3.03it/s]  1%|          | 4267/400000 [00:01<25:22:23,  4.33it/s]  1%|▏         | 5125/400000 [00:01<17:43:35,  6.19it/s]  2%|▏         | 6006/400000 [00:02<12:23:04,  8.84it/s]  2%|▏         | 6860/400000 [00:02<8:39:15, 12.62it/s]   2%|▏         | 7691/400000 [00:02<6:02:56, 18.01it/s]  2%|▏         | 8557/400000 [00:02<4:13:43, 25.71it/s]  2%|▏         | 9402/400000 [00:02<2:57:27, 36.68it/s]  3%|▎         | 10239/400000 [00:02<2:04:12, 52.30it/s]  3%|▎         | 11057/400000 [00:02<1:27:01, 74.49it/s]  3%|▎         | 11935/400000 [00:02<1:00:59, 106.03it/s]  3%|▎         | 12758/400000 [00:02<42:51, 150.62it/s]    3%|▎         | 13573/400000 [00:02<30:10, 213.39it/s]  4%|▎         | 14445/400000 [00:03<21:18, 301.68it/s]  4%|▍         | 15312/400000 [00:03<15:05, 424.64it/s]  4%|▍         | 16197/400000 [00:03<10:45, 594.40it/s]  4%|▍         | 17078/400000 [00:03<07:43, 825.27it/s]  4%|▍         | 17954/400000 [00:03<05:37, 1133.19it/s]  5%|▍         | 18819/400000 [00:03<04:08, 1531.96it/s]  5%|▍         | 19723/400000 [00:03<03:06, 2040.23it/s]  5%|▌         | 20612/400000 [00:03<02:22, 2653.54it/s]  5%|▌         | 21491/400000 [00:03<01:52, 3351.54it/s]  6%|▌         | 22367/400000 [00:04<01:33, 4043.47it/s]  6%|▌         | 23214/400000 [00:04<01:19, 4758.85it/s]  6%|▌         | 24111/400000 [00:04<01:07, 5537.85it/s]  6%|▌         | 24972/400000 [00:04<01:00, 6200.97it/s]  6%|▋         | 25828/400000 [00:04<00:56, 6610.82it/s]  7%|▋         | 26662/400000 [00:04<00:53, 6972.91it/s]  7%|▋         | 27511/400000 [00:04<00:50, 7366.81it/s]  7%|▋         | 28387/400000 [00:04<00:48, 7735.79it/s]  7%|▋         | 29279/400000 [00:04<00:46, 8056.14it/s]  8%|▊         | 30163/400000 [00:04<00:44, 8273.47it/s]  8%|▊         | 31029/400000 [00:05<00:44, 8334.37it/s]  8%|▊         | 31890/400000 [00:05<00:43, 8403.21it/s]  8%|▊         | 32771/400000 [00:05<00:43, 8520.08it/s]  8%|▊         | 33637/400000 [00:05<00:43, 8426.13it/s]  9%|▊         | 34521/400000 [00:05<00:42, 8543.42it/s]  9%|▉         | 35397/400000 [00:05<00:42, 8606.09it/s]  9%|▉         | 36263/400000 [00:05<00:42, 8527.35it/s]  9%|▉         | 37130/400000 [00:05<00:42, 8568.94it/s]  9%|▉         | 37990/400000 [00:05<00:42, 8572.94it/s] 10%|▉         | 38850/400000 [00:05<00:42, 8533.82it/s] 10%|▉         | 39709/400000 [00:06<00:42, 8550.03it/s] 10%|█         | 40565/400000 [00:06<00:42, 8439.58it/s] 10%|█         | 41410/400000 [00:06<00:42, 8355.48it/s] 11%|█         | 42302/400000 [00:06<00:42, 8514.53it/s] 11%|█         | 43168/400000 [00:06<00:41, 8555.60it/s] 11%|█         | 44025/400000 [00:06<00:41, 8542.66it/s] 11%|█         | 44880/400000 [00:06<00:42, 8417.97it/s] 11%|█▏        | 45768/400000 [00:06<00:41, 8547.99it/s] 12%|█▏        | 46658/400000 [00:06<00:40, 8649.34it/s] 12%|█▏        | 47535/400000 [00:06<00:40, 8682.63it/s] 12%|█▏        | 48405/400000 [00:07<00:40, 8622.38it/s] 12%|█▏        | 49297/400000 [00:07<00:40, 8706.87it/s] 13%|█▎        | 50176/400000 [00:07<00:40, 8730.30it/s] 13%|█▎        | 51065/400000 [00:07<00:39, 8775.32it/s] 13%|█▎        | 51956/400000 [00:07<00:39, 8814.91it/s] 13%|█▎        | 52838/400000 [00:07<00:39, 8760.13it/s] 13%|█▎        | 53715/400000 [00:07<00:41, 8441.40it/s] 14%|█▎        | 54591/400000 [00:07<00:40, 8533.82it/s] 14%|█▍        | 55460/400000 [00:07<00:40, 8577.37it/s] 14%|█▍        | 56320/400000 [00:07<00:40, 8581.53it/s] 14%|█▍        | 57180/400000 [00:08<00:40, 8568.21it/s] 15%|█▍        | 58045/400000 [00:08<00:39, 8590.76it/s] 15%|█▍        | 58925/400000 [00:08<00:39, 8652.32it/s] 15%|█▍        | 59791/400000 [00:08<00:39, 8624.73it/s] 15%|█▌        | 60719/400000 [00:08<00:38, 8810.09it/s] 15%|█▌        | 61602/400000 [00:08<00:38, 8757.53it/s] 16%|█▌        | 62479/400000 [00:08<00:39, 8602.73it/s] 16%|█▌        | 63356/400000 [00:08<00:38, 8652.05it/s] 16%|█▌        | 64223/400000 [00:08<00:38, 8648.17it/s] 16%|█▋        | 65119/400000 [00:08<00:38, 8739.01it/s] 16%|█▋        | 65994/400000 [00:09<00:38, 8726.31it/s] 17%|█▋        | 66868/400000 [00:09<00:38, 8603.10it/s] 17%|█▋        | 67761/400000 [00:09<00:38, 8698.13it/s] 17%|█▋        | 68635/400000 [00:09<00:38, 8710.50it/s] 17%|█▋        | 69507/400000 [00:09<00:38, 8678.01it/s] 18%|█▊        | 70376/400000 [00:09<00:38, 8518.44it/s] 18%|█▊        | 71287/400000 [00:09<00:37, 8686.06it/s] 18%|█▊        | 72197/400000 [00:09<00:37, 8805.16it/s] 18%|█▊        | 73081/400000 [00:09<00:37, 8814.86it/s] 18%|█▊        | 73964/400000 [00:10<00:37, 8782.62it/s] 19%|█▊        | 74843/400000 [00:10<00:37, 8633.17it/s] 19%|█▉        | 75708/400000 [00:10<00:38, 8457.13it/s] 19%|█▉        | 76556/400000 [00:10<00:38, 8357.63it/s] 19%|█▉        | 77441/400000 [00:10<00:37, 8496.50it/s] 20%|█▉        | 78293/400000 [00:10<00:38, 8373.96it/s] 20%|█▉        | 79134/400000 [00:10<00:38, 8382.48it/s] 20%|██        | 80008/400000 [00:10<00:37, 8485.40it/s] 20%|██        | 80897/400000 [00:10<00:37, 8592.90it/s] 20%|██        | 81762/400000 [00:10<00:36, 8607.40it/s] 21%|██        | 82628/400000 [00:11<00:36, 8621.13it/s] 21%|██        | 83491/400000 [00:11<00:37, 8533.41it/s] 21%|██        | 84387/400000 [00:11<00:36, 8655.67it/s] 21%|██▏       | 85254/400000 [00:11<00:37, 8501.07it/s] 22%|██▏       | 86130/400000 [00:11<00:36, 8574.42it/s] 22%|██▏       | 87027/400000 [00:11<00:36, 8687.50it/s] 22%|██▏       | 87897/400000 [00:11<00:36, 8654.38it/s] 22%|██▏       | 88785/400000 [00:11<00:35, 8719.22it/s] 22%|██▏       | 89658/400000 [00:11<00:35, 8707.88it/s] 23%|██▎       | 90546/400000 [00:11<00:35, 8758.34it/s] 23%|██▎       | 91445/400000 [00:12<00:34, 8826.13it/s] 23%|██▎       | 92329/400000 [00:12<00:34, 8804.63it/s] 23%|██▎       | 93247/400000 [00:12<00:34, 8913.53it/s] 24%|██▎       | 94139/400000 [00:12<00:35, 8715.84it/s] 24%|██▍       | 95045/400000 [00:12<00:34, 8810.93it/s] 24%|██▍       | 95928/400000 [00:12<00:34, 8801.38it/s] 24%|██▍       | 96809/400000 [00:12<00:34, 8778.69it/s] 24%|██▍       | 97688/400000 [00:12<00:34, 8747.74it/s] 25%|██▍       | 98564/400000 [00:12<00:34, 8694.67it/s] 25%|██▍       | 99434/400000 [00:12<00:35, 8564.77it/s] 25%|██▌       | 100302/400000 [00:13<00:34, 8596.30it/s] 25%|██▌       | 101163/400000 [00:13<00:34, 8591.26it/s] 26%|██▌       | 102054/400000 [00:13<00:34, 8684.27it/s] 26%|██▌       | 102949/400000 [00:13<00:33, 8759.21it/s] 26%|██▌       | 103840/400000 [00:13<00:33, 8801.13it/s] 26%|██▌       | 104722/400000 [00:13<00:33, 8803.64it/s] 26%|██▋       | 105603/400000 [00:13<00:33, 8734.51it/s] 27%|██▋       | 106497/400000 [00:13<00:33, 8793.48it/s] 27%|██▋       | 107380/400000 [00:13<00:33, 8802.66it/s] 27%|██▋       | 108278/400000 [00:13<00:32, 8853.20it/s] 27%|██▋       | 109164/400000 [00:14<00:32, 8818.12it/s] 28%|██▊       | 110047/400000 [00:14<00:32, 8801.39it/s] 28%|██▊       | 110930/400000 [00:14<00:32, 8808.78it/s] 28%|██▊       | 111811/400000 [00:14<00:32, 8775.13it/s] 28%|██▊       | 112689/400000 [00:14<00:33, 8610.08it/s] 28%|██▊       | 113563/400000 [00:14<00:33, 8648.28it/s] 29%|██▊       | 114439/400000 [00:14<00:32, 8681.00it/s] 29%|██▉       | 115323/400000 [00:14<00:32, 8727.84it/s] 29%|██▉       | 116197/400000 [00:14<00:32, 8721.27it/s] 29%|██▉       | 117070/400000 [00:14<00:32, 8707.07it/s] 29%|██▉       | 117941/400000 [00:15<00:32, 8585.26it/s] 30%|██▉       | 118801/400000 [00:15<00:33, 8457.21it/s] 30%|██▉       | 119648/400000 [00:15<00:33, 8435.18it/s] 30%|███       | 120541/400000 [00:15<00:32, 8575.53it/s] 30%|███       | 121429/400000 [00:15<00:32, 8663.79it/s] 31%|███       | 122297/400000 [00:15<00:32, 8602.35it/s] 31%|███       | 123158/400000 [00:15<00:33, 8366.13it/s] 31%|███       | 124031/400000 [00:15<00:32, 8469.51it/s] 31%|███       | 124906/400000 [00:15<00:32, 8548.39it/s] 31%|███▏      | 125787/400000 [00:15<00:31, 8623.59it/s] 32%|███▏      | 126651/400000 [00:16<00:31, 8574.52it/s] 32%|███▏      | 127510/400000 [00:16<00:32, 8263.47it/s] 32%|███▏      | 128395/400000 [00:16<00:32, 8430.27it/s] 32%|███▏      | 129271/400000 [00:16<00:31, 8524.67it/s] 33%|███▎      | 130136/400000 [00:16<00:31, 8560.04it/s] 33%|███▎      | 131001/400000 [00:16<00:31, 8584.27it/s] 33%|███▎      | 131861/400000 [00:16<00:31, 8586.38it/s] 33%|███▎      | 132759/400000 [00:16<00:30, 8698.65it/s] 33%|███▎      | 133630/400000 [00:16<00:31, 8423.33it/s] 34%|███▎      | 134530/400000 [00:17<00:30, 8585.82it/s] 34%|███▍      | 135392/400000 [00:17<00:30, 8568.27it/s] 34%|███▍      | 136263/400000 [00:17<00:30, 8608.51it/s] 34%|███▍      | 137168/400000 [00:17<00:30, 8736.29it/s] 35%|███▍      | 138054/400000 [00:17<00:29, 8770.33it/s] 35%|███▍      | 138964/400000 [00:17<00:29, 8866.05it/s] 35%|███▍      | 139855/400000 [00:17<00:29, 8878.34it/s] 35%|███▌      | 140744/400000 [00:17<00:29, 8750.28it/s] 35%|███▌      | 141626/400000 [00:17<00:29, 8769.86it/s] 36%|███▌      | 142511/400000 [00:17<00:29, 8792.78it/s] 36%|███▌      | 143411/400000 [00:18<00:28, 8851.34it/s] 36%|███▌      | 144297/400000 [00:18<00:28, 8835.34it/s] 36%|███▋      | 145181/400000 [00:18<00:29, 8572.89it/s] 37%|███▋      | 146041/400000 [00:18<00:29, 8477.98it/s] 37%|███▋      | 146948/400000 [00:18<00:29, 8646.45it/s] 37%|███▋      | 147858/400000 [00:18<00:28, 8776.38it/s] 37%|███▋      | 148738/400000 [00:18<00:29, 8643.20it/s] 37%|███▋      | 149621/400000 [00:18<00:28, 8698.13it/s] 38%|███▊      | 150516/400000 [00:18<00:28, 8771.79it/s] 38%|███▊      | 151404/400000 [00:18<00:28, 8801.76it/s] 38%|███▊      | 152286/400000 [00:19<00:28, 8806.82it/s] 38%|███▊      | 153181/400000 [00:19<00:27, 8849.08it/s] 39%|███▊      | 154070/400000 [00:19<00:27, 8859.00it/s] 39%|███▊      | 154979/400000 [00:19<00:27, 8926.85it/s] 39%|███▉      | 155874/400000 [00:19<00:27, 8931.77it/s] 39%|███▉      | 156768/400000 [00:19<00:27, 8883.70it/s] 39%|███▉      | 157657/400000 [00:19<00:27, 8880.39it/s] 40%|███▉      | 158546/400000 [00:19<00:27, 8849.85it/s] 40%|███▉      | 159457/400000 [00:19<00:26, 8924.84it/s] 40%|████      | 160350/400000 [00:19<00:27, 8822.67it/s] 40%|████      | 161233/400000 [00:20<00:27, 8734.15it/s] 41%|████      | 162107/400000 [00:20<00:27, 8708.85it/s] 41%|████      | 162997/400000 [00:20<00:27, 8762.95it/s] 41%|████      | 163874/400000 [00:20<00:27, 8539.26it/s] 41%|████      | 164766/400000 [00:20<00:27, 8648.81it/s] 41%|████▏     | 165635/400000 [00:20<00:27, 8660.99it/s] 42%|████▏     | 166503/400000 [00:20<00:27, 8501.24it/s] 42%|████▏     | 167355/400000 [00:20<00:28, 8174.31it/s] 42%|████▏     | 168211/400000 [00:20<00:27, 8285.28it/s] 42%|████▏     | 169081/400000 [00:20<00:27, 8404.71it/s] 42%|████▏     | 169924/400000 [00:21<00:27, 8408.60it/s] 43%|████▎     | 170802/400000 [00:21<00:26, 8516.16it/s] 43%|████▎     | 171657/400000 [00:21<00:26, 8523.14it/s] 43%|████▎     | 172532/400000 [00:21<00:26, 8588.07it/s] 43%|████▎     | 173392/400000 [00:21<00:26, 8568.33it/s] 44%|████▎     | 174250/400000 [00:21<00:27, 8353.75it/s] 44%|████▍     | 175087/400000 [00:21<00:27, 8270.04it/s] 44%|████▍     | 175974/400000 [00:21<00:26, 8439.37it/s] 44%|████▍     | 176853/400000 [00:21<00:26, 8541.23it/s] 44%|████▍     | 177755/400000 [00:22<00:25, 8678.98it/s] 45%|████▍     | 178637/400000 [00:22<00:25, 8719.99it/s] 45%|████▍     | 179521/400000 [00:22<00:25, 8753.06it/s] 45%|████▌     | 180404/400000 [00:22<00:25, 8773.41it/s] 45%|████▌     | 181282/400000 [00:22<00:25, 8594.08it/s] 46%|████▌     | 182143/400000 [00:22<00:25, 8590.53it/s] 46%|████▌     | 183003/400000 [00:22<00:25, 8438.53it/s] 46%|████▌     | 183888/400000 [00:22<00:25, 8556.15it/s] 46%|████▌     | 184745/400000 [00:22<00:25, 8553.95it/s] 46%|████▋     | 185602/400000 [00:22<00:25, 8270.69it/s] 47%|████▋     | 186432/400000 [00:23<00:25, 8276.80it/s] 47%|████▋     | 187290/400000 [00:23<00:25, 8364.49it/s] 47%|████▋     | 188156/400000 [00:23<00:25, 8448.27it/s] 47%|████▋     | 189003/400000 [00:23<00:25, 8417.04it/s] 47%|████▋     | 189878/400000 [00:23<00:24, 8513.27it/s] 48%|████▊     | 190743/400000 [00:23<00:24, 8553.16it/s] 48%|████▊     | 191610/400000 [00:23<00:24, 8585.90it/s] 48%|████▊     | 192506/400000 [00:23<00:23, 8692.28it/s] 48%|████▊     | 193376/400000 [00:23<00:23, 8693.06it/s] 49%|████▊     | 194246/400000 [00:23<00:24, 8476.67it/s] 49%|████▉     | 195096/400000 [00:24<00:24, 8386.60it/s] 49%|████▉     | 195968/400000 [00:24<00:24, 8482.56it/s] 49%|████▉     | 196872/400000 [00:24<00:23, 8640.52it/s] 49%|████▉     | 197738/400000 [00:24<00:24, 8305.15it/s] 50%|████▉     | 198573/400000 [00:24<00:24, 8183.37it/s] 50%|████▉     | 199434/400000 [00:24<00:24, 8304.71it/s] 50%|█████     | 200304/400000 [00:24<00:23, 8418.07it/s] 50%|█████     | 201177/400000 [00:24<00:23, 8507.97it/s] 51%|█████     | 202048/400000 [00:24<00:23, 8567.50it/s] 51%|█████     | 202907/400000 [00:24<00:23, 8533.79it/s] 51%|█████     | 203762/400000 [00:25<00:23, 8196.17it/s] 51%|█████     | 204606/400000 [00:25<00:23, 8266.98it/s] 51%|█████▏    | 205456/400000 [00:25<00:23, 8333.37it/s] 52%|█████▏    | 206292/400000 [00:25<00:23, 8211.02it/s] 52%|█████▏    | 207161/400000 [00:25<00:23, 8347.79it/s] 52%|█████▏    | 208023/400000 [00:25<00:22, 8425.45it/s] 52%|█████▏    | 208867/400000 [00:25<00:22, 8384.16it/s] 52%|█████▏    | 209753/400000 [00:25<00:22, 8521.41it/s] 53%|█████▎    | 210607/400000 [00:25<00:22, 8523.76it/s] 53%|█████▎    | 211483/400000 [00:25<00:21, 8591.83it/s] 53%|█████▎    | 212343/400000 [00:26<00:21, 8565.73it/s] 53%|█████▎    | 213201/400000 [00:26<00:21, 8536.76it/s] 54%|█████▎    | 214056/400000 [00:26<00:22, 8405.80it/s] 54%|█████▎    | 214898/400000 [00:26<00:22, 8156.72it/s] 54%|█████▍    | 215755/400000 [00:26<00:22, 8273.92it/s] 54%|█████▍    | 216600/400000 [00:26<00:22, 8323.17it/s] 54%|█████▍    | 217470/400000 [00:26<00:21, 8431.05it/s] 55%|█████▍    | 218354/400000 [00:26<00:21, 8548.58it/s] 55%|█████▍    | 219211/400000 [00:26<00:21, 8534.71it/s] 55%|█████▌    | 220066/400000 [00:27<00:21, 8426.83it/s] 55%|█████▌    | 220910/400000 [00:27<00:21, 8403.00it/s] 55%|█████▌    | 221779/400000 [00:27<00:21, 8485.75it/s] 56%|█████▌    | 222654/400000 [00:27<00:20, 8561.75it/s] 56%|█████▌    | 223519/400000 [00:27<00:20, 8586.60it/s] 56%|█████▌    | 224393/400000 [00:27<00:20, 8629.46it/s] 56%|█████▋    | 225257/400000 [00:27<00:20, 8604.12it/s] 57%|█████▋    | 226126/400000 [00:27<00:20, 8626.84it/s] 57%|█████▋    | 227006/400000 [00:27<00:19, 8677.79it/s] 57%|█████▋    | 227874/400000 [00:27<00:20, 8312.50it/s] 57%|█████▋    | 228709/400000 [00:28<00:20, 8197.20it/s] 57%|█████▋    | 229532/400000 [00:28<00:21, 7966.48it/s] 58%|█████▊    | 230332/400000 [00:28<00:21, 7873.47it/s] 58%|█████▊    | 231173/400000 [00:28<00:21, 8025.69it/s] 58%|█████▊    | 232012/400000 [00:28<00:20, 8130.26it/s] 58%|█████▊    | 232887/400000 [00:28<00:20, 8306.16it/s] 58%|█████▊    | 233763/400000 [00:28<00:19, 8436.53it/s] 59%|█████▊    | 234661/400000 [00:28<00:19, 8592.42it/s] 59%|█████▉    | 235523/400000 [00:28<00:19, 8501.37it/s] 59%|█████▉    | 236375/400000 [00:28<00:19, 8426.46it/s] 59%|█████▉    | 237278/400000 [00:29<00:18, 8596.97it/s] 60%|█████▉    | 238140/400000 [00:29<00:18, 8591.50it/s] 60%|█████▉    | 239048/400000 [00:29<00:18, 8730.64it/s] 60%|█████▉    | 239936/400000 [00:29<00:18, 8773.38it/s] 60%|██████    | 240815/400000 [00:29<00:18, 8747.41it/s] 60%|██████    | 241691/400000 [00:29<00:18, 8724.67it/s] 61%|██████    | 242565/400000 [00:29<00:18, 8690.82it/s] 61%|██████    | 243438/400000 [00:29<00:17, 8700.45it/s] 61%|██████    | 244330/400000 [00:29<00:17, 8764.66it/s] 61%|██████▏   | 245207/400000 [00:29<00:18, 8495.65it/s] 62%|██████▏   | 246059/400000 [00:30<00:18, 8465.30it/s] 62%|██████▏   | 246907/400000 [00:30<00:18, 8462.12it/s] 62%|██████▏   | 247777/400000 [00:30<00:17, 8531.68it/s] 62%|██████▏   | 248636/400000 [00:30<00:17, 8544.80it/s] 62%|██████▏   | 249492/400000 [00:30<00:17, 8534.29it/s] 63%|██████▎   | 250360/400000 [00:30<00:17, 8574.85it/s] 63%|██████▎   | 251218/400000 [00:30<00:17, 8519.93it/s] 63%|██████▎   | 252071/400000 [00:30<00:17, 8480.84it/s] 63%|██████▎   | 252954/400000 [00:30<00:17, 8581.28it/s] 63%|██████▎   | 253818/400000 [00:30<00:17, 8596.64it/s] 64%|██████▎   | 254678/400000 [00:31<00:17, 8491.66it/s] 64%|██████▍   | 255533/400000 [00:31<00:16, 8506.83it/s] 64%|██████▍   | 256423/400000 [00:31<00:16, 8620.22it/s] 64%|██████▍   | 257300/400000 [00:31<00:16, 8664.10it/s] 65%|██████▍   | 258175/400000 [00:31<00:16, 8689.33it/s] 65%|██████▍   | 259045/400000 [00:31<00:16, 8568.70it/s] 65%|██████▍   | 259903/400000 [00:31<00:17, 8138.18it/s] 65%|██████▌   | 260766/400000 [00:31<00:16, 8278.07it/s] 65%|██████▌   | 261598/400000 [00:31<00:17, 8043.96it/s] 66%|██████▌   | 262441/400000 [00:32<00:16, 8154.43it/s] 66%|██████▌   | 263301/400000 [00:32<00:16, 8282.27it/s] 66%|██████▌   | 264151/400000 [00:32<00:16, 8343.97it/s] 66%|██████▋   | 265041/400000 [00:32<00:15, 8501.76it/s] 66%|██████▋   | 265943/400000 [00:32<00:15, 8649.12it/s] 67%|██████▋   | 266811/400000 [00:32<00:15, 8630.56it/s] 67%|██████▋   | 267676/400000 [00:32<00:15, 8588.46it/s] 67%|██████▋   | 268536/400000 [00:32<00:15, 8541.62it/s] 67%|██████▋   | 269430/400000 [00:32<00:15, 8655.47it/s] 68%|██████▊   | 270297/400000 [00:32<00:15, 8637.17it/s] 68%|██████▊   | 271162/400000 [00:33<00:15, 8432.06it/s] 68%|██████▊   | 272031/400000 [00:33<00:15, 8505.58it/s] 68%|██████▊   | 272922/400000 [00:33<00:14, 8620.14it/s] 68%|██████▊   | 273809/400000 [00:33<00:14, 8691.39it/s] 69%|██████▊   | 274707/400000 [00:33<00:14, 8775.20it/s] 69%|██████▉   | 275586/400000 [00:33<00:14, 8719.58it/s] 69%|██████▉   | 276459/400000 [00:33<00:14, 8464.33it/s] 69%|██████▉   | 277360/400000 [00:33<00:14, 8619.82it/s] 70%|██████▉   | 278254/400000 [00:33<00:13, 8712.29it/s] 70%|██████▉   | 279127/400000 [00:33<00:13, 8645.40it/s] 70%|██████▉   | 279993/400000 [00:34<00:13, 8593.58it/s] 70%|███████   | 280854/400000 [00:34<00:13, 8568.41it/s] 70%|███████   | 281745/400000 [00:34<00:13, 8667.70it/s] 71%|███████   | 282616/400000 [00:34<00:13, 8677.43it/s] 71%|███████   | 283485/400000 [00:34<00:13, 8536.48it/s] 71%|███████   | 284350/400000 [00:34<00:13, 8567.43it/s] 71%|███████▏  | 285219/400000 [00:34<00:13, 8601.86it/s] 72%|███████▏  | 286086/400000 [00:34<00:13, 8621.32it/s] 72%|███████▏  | 286961/400000 [00:34<00:13, 8659.02it/s] 72%|███████▏  | 287830/400000 [00:34<00:12, 8667.33it/s] 72%|███████▏  | 288719/400000 [00:35<00:12, 8732.01it/s] 72%|███████▏  | 289593/400000 [00:35<00:12, 8719.66it/s] 73%|███████▎  | 290470/400000 [00:35<00:12, 8732.21it/s] 73%|███████▎  | 291344/400000 [00:35<00:12, 8711.46it/s] 73%|███████▎  | 292216/400000 [00:35<00:12, 8669.19it/s] 73%|███████▎  | 293084/400000 [00:35<00:12, 8621.69it/s] 73%|███████▎  | 293947/400000 [00:35<00:12, 8615.60it/s] 74%|███████▎  | 294826/400000 [00:35<00:12, 8665.87it/s] 74%|███████▍  | 295708/400000 [00:35<00:11, 8711.06it/s] 74%|███████▍  | 296580/400000 [00:35<00:11, 8710.76it/s] 74%|███████▍  | 297452/400000 [00:36<00:11, 8689.49it/s] 75%|███████▍  | 298322/400000 [00:36<00:11, 8656.35it/s] 75%|███████▍  | 299188/400000 [00:36<00:11, 8627.59it/s] 75%|███████▌  | 300051/400000 [00:36<00:11, 8472.76it/s] 75%|███████▌  | 300899/400000 [00:36<00:11, 8403.68it/s] 75%|███████▌  | 301778/400000 [00:36<00:11, 8515.29it/s] 76%|███████▌  | 302635/400000 [00:36<00:11, 8529.84it/s] 76%|███████▌  | 303501/400000 [00:36<00:11, 8567.76it/s] 76%|███████▌  | 304359/400000 [00:36<00:11, 8525.61it/s] 76%|███████▋  | 305212/400000 [00:36<00:11, 8387.85it/s] 77%|███████▋  | 306068/400000 [00:37<00:11, 8437.60it/s] 77%|███████▋  | 306956/400000 [00:37<00:10, 8562.95it/s] 77%|███████▋  | 307850/400000 [00:37<00:10, 8671.36it/s] 77%|███████▋  | 308757/400000 [00:37<00:10, 8786.64it/s] 77%|███████▋  | 309637/400000 [00:37<00:10, 8753.28it/s] 78%|███████▊  | 310514/400000 [00:37<00:10, 8707.04it/s] 78%|███████▊  | 311400/400000 [00:37<00:10, 8749.57it/s] 78%|███████▊  | 312279/400000 [00:37<00:10, 8759.72it/s] 78%|███████▊  | 313166/400000 [00:37<00:09, 8792.20it/s] 79%|███████▊  | 314046/400000 [00:37<00:09, 8766.05it/s] 79%|███████▊  | 314923/400000 [00:38<00:09, 8724.85it/s] 79%|███████▉  | 315796/400000 [00:38<00:09, 8713.61it/s] 79%|███████▉  | 316668/400000 [00:38<00:09, 8709.41it/s] 79%|███████▉  | 317540/400000 [00:38<00:09, 8409.95it/s] 80%|███████▉  | 318397/400000 [00:38<00:09, 8456.24it/s] 80%|███████▉  | 319245/400000 [00:38<00:09, 8418.00it/s] 80%|████████  | 320108/400000 [00:38<00:09, 8477.64it/s] 80%|████████  | 320987/400000 [00:38<00:09, 8567.83it/s] 80%|████████  | 321845/400000 [00:38<00:09, 8555.83it/s] 81%|████████  | 322702/400000 [00:39<00:09, 8516.51it/s] 81%|████████  | 323555/400000 [00:39<00:08, 8518.33it/s] 81%|████████  | 324449/400000 [00:39<00:08, 8638.89it/s] 81%|████████▏ | 325330/400000 [00:39<00:08, 8687.03it/s] 82%|████████▏ | 326211/400000 [00:39<00:08, 8721.69it/s] 82%|████████▏ | 327084/400000 [00:39<00:08, 8688.35it/s] 82%|████████▏ | 327954/400000 [00:39<00:08, 8554.47it/s] 82%|████████▏ | 328830/400000 [00:39<00:08, 8613.02it/s] 82%|████████▏ | 329692/400000 [00:39<00:08, 8602.83it/s] 83%|████████▎ | 330553/400000 [00:39<00:08, 8555.21it/s] 83%|████████▎ | 331409/400000 [00:40<00:08, 8455.63it/s] 83%|████████▎ | 332256/400000 [00:40<00:08, 8360.94it/s] 83%|████████▎ | 333097/400000 [00:40<00:07, 8373.86it/s] 83%|████████▎ | 333935/400000 [00:40<00:08, 8218.15it/s] 84%|████████▎ | 334783/400000 [00:40<00:07, 8294.36it/s] 84%|████████▍ | 335614/400000 [00:40<00:07, 8211.60it/s] 84%|████████▍ | 336436/400000 [00:40<00:07, 8176.87it/s] 84%|████████▍ | 337291/400000 [00:40<00:07, 8283.57it/s] 85%|████████▍ | 338158/400000 [00:40<00:07, 8394.96it/s] 85%|████████▍ | 339006/400000 [00:40<00:07, 8418.70it/s] 85%|████████▍ | 339849/400000 [00:41<00:07, 8380.73it/s] 85%|████████▌ | 340688/400000 [00:41<00:07, 8285.98it/s] 85%|████████▌ | 341518/400000 [00:41<00:07, 8210.76it/s] 86%|████████▌ | 342367/400000 [00:41<00:06, 8290.25it/s] 86%|████████▌ | 343221/400000 [00:41<00:06, 8362.95it/s] 86%|████████▌ | 344077/400000 [00:41<00:06, 8416.08it/s] 86%|████████▌ | 344920/400000 [00:41<00:06, 8323.62it/s] 86%|████████▋ | 345794/400000 [00:41<00:06, 8443.07it/s] 87%|████████▋ | 346653/400000 [00:41<00:06, 8483.71it/s] 87%|████████▋ | 347513/400000 [00:41<00:06, 8515.93it/s] 87%|████████▋ | 348382/400000 [00:42<00:06, 8566.86it/s] 87%|████████▋ | 349240/400000 [00:42<00:06, 8426.63it/s] 88%|████████▊ | 350084/400000 [00:42<00:05, 8324.52it/s] 88%|████████▊ | 350930/400000 [00:42<00:05, 8364.30it/s] 88%|████████▊ | 351788/400000 [00:42<00:05, 8425.61it/s] 88%|████████▊ | 352632/400000 [00:42<00:05, 8306.60it/s] 88%|████████▊ | 353487/400000 [00:42<00:05, 8375.89it/s] 89%|████████▊ | 354343/400000 [00:42<00:05, 8429.15it/s] 89%|████████▉ | 355199/400000 [00:42<00:05, 8465.49it/s] 89%|████████▉ | 356049/400000 [00:42<00:05, 8473.27it/s] 89%|████████▉ | 356897/400000 [00:43<00:05, 8431.62it/s] 89%|████████▉ | 357741/400000 [00:43<00:05, 8344.17it/s] 90%|████████▉ | 358587/400000 [00:43<00:04, 8376.23it/s] 90%|████████▉ | 359446/400000 [00:43<00:04, 8438.19it/s] 90%|█████████ | 360309/400000 [00:43<00:04, 8493.36it/s] 90%|█████████ | 361175/400000 [00:43<00:04, 8541.70it/s] 91%|█████████ | 362036/400000 [00:43<00:04, 8561.45it/s] 91%|█████████ | 362906/400000 [00:43<00:04, 8600.20it/s] 91%|█████████ | 363798/400000 [00:43<00:04, 8691.82it/s] 91%|█████████ | 364680/400000 [00:43<00:04, 8727.26it/s] 91%|█████████▏| 365554/400000 [00:44<00:03, 8713.63it/s] 92%|█████████▏| 366426/400000 [00:44<00:03, 8496.69it/s] 92%|█████████▏| 367301/400000 [00:44<00:03, 8570.38it/s] 92%|█████████▏| 368160/400000 [00:44<00:03, 8496.64it/s] 92%|█████████▏| 369049/400000 [00:44<00:03, 8608.61it/s] 92%|█████████▏| 369923/400000 [00:44<00:03, 8645.61it/s] 93%|█████████▎| 370799/400000 [00:44<00:03, 8677.85it/s] 93%|█████████▎| 371683/400000 [00:44<00:03, 8725.10it/s] 93%|█████████▎| 372572/400000 [00:44<00:03, 8771.07it/s] 93%|█████████▎| 373450/400000 [00:44<00:03, 8718.36it/s] 94%|█████████▎| 374323/400000 [00:45<00:02, 8616.98it/s] 94%|█████████▍| 375197/400000 [00:45<00:02, 8652.08it/s] 94%|█████████▍| 376088/400000 [00:45<00:02, 8726.35it/s] 94%|█████████▍| 376962/400000 [00:45<00:02, 8687.01it/s] 94%|█████████▍| 377832/400000 [00:45<00:02, 8547.93it/s] 95%|█████████▍| 378688/400000 [00:45<00:02, 8536.79it/s] 95%|█████████▍| 379543/400000 [00:45<00:02, 8477.49it/s] 95%|█████████▌| 380392/400000 [00:45<00:02, 8465.59it/s] 95%|█████████▌| 381277/400000 [00:45<00:02, 8575.41it/s] 96%|█████████▌| 382162/400000 [00:45<00:02, 8654.06it/s] 96%|█████████▌| 383028/400000 [00:46<00:01, 8498.02it/s] 96%|█████████▌| 383879/400000 [00:46<00:01, 8490.25it/s] 96%|█████████▌| 384759/400000 [00:46<00:01, 8580.32it/s] 96%|█████████▋| 385636/400000 [00:46<00:01, 8636.01it/s] 97%|█████████▋| 386501/400000 [00:46<00:01, 8614.98it/s] 97%|█████████▋| 387363/400000 [00:46<00:01, 8430.35it/s] 97%|█████████▋| 388233/400000 [00:46<00:01, 8506.93it/s] 97%|█████████▋| 389098/400000 [00:46<00:01, 8546.90it/s] 97%|█████████▋| 389974/400000 [00:46<00:01, 8609.69it/s] 98%|█████████▊| 390844/400000 [00:47<00:01, 8636.16it/s] 98%|█████████▊| 391709/400000 [00:47<00:00, 8362.06it/s] 98%|█████████▊| 392582/400000 [00:47<00:00, 8467.42it/s] 98%|█████████▊| 393454/400000 [00:47<00:00, 8430.19it/s] 99%|█████████▊| 394299/400000 [00:47<00:00, 8201.87it/s] 99%|█████████▉| 395193/400000 [00:47<00:00, 8408.90it/s] 99%|█████████▉| 396070/400000 [00:47<00:00, 8511.31it/s] 99%|█████████▉| 396942/400000 [00:47<00:00, 8570.26it/s] 99%|█████████▉| 397841/400000 [00:47<00:00, 8689.30it/s]100%|█████████▉| 398743/400000 [00:47<00:00, 8785.12it/s]100%|█████████▉| 399631/400000 [00:48<00:00, 8811.49it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8319.17it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f90912b3cc0>, <torchtext.data.dataset.TabularDataset object at 0x7f90912b3e10>, <torchtext.vocab.Vocab object at 0x7f90912b3d30>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  4.10 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  4.10 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  4.10 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  4.10 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.43 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.43 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.04 file/s]2020-07-21 18:18:42.103869: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-21 18:18:42.108019: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-07-21 18:18:42.108192: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a1b31ee830 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-21 18:18:42.108207: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 149796.90it/s] 68%|██████▊   | 6782976/9912422 [00:00<00:14, 213792.26it/s]9920512it [00:00, 41314943.67it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1263225.68it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 141347.49it/s]1654784it [00:00, 10256774.73it/s]                         
0it [00:00, ?it/s]8192it [00:00, 136344.92it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
