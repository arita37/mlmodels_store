
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1b9f93b9d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1b9f93b9d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1c0a5a6510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1c0a5a6510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1c297f5ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1c297f5ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bb78d7950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bb78d7950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bb78d7950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:16, 129371.84it/s] 25%|██▌       | 2514944/9912422 [00:00<00:40, 184407.41it/s] 64%|██████▎   | 6299648/9912422 [00:00<00:13, 262884.38it/s] 93%|█████████▎| 9248768/9912422 [00:00<00:01, 374110.95it/s]9920512it [00:00, 22703752.05it/s]                           
0it [00:00, ?it/s]32768it [00:00, 553968.19it/s]
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 982661.40it/s]1654784it [00:00, 11400195.71it/s]                         
0it [00:00, ?it/s]8192it [00:00, 241740.17it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bb5e2f5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bb5e49898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1bb78d7598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1bb78d7598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1bb78d7598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<00:59,  2.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:59,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:58,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:58,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:42,  3.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:42,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:42,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:41,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:41,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:41,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:41,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:40,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:40,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:28,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:28,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:28,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:27,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:27,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:27,  5.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:19,  7.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:19,  7.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:19,  7.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:19,  7.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:18,  7.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:01<00:14,  9.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:14,  9.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:14,  9.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:13,  9.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:13,  9.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:13,  9.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:01<00:10, 12.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:10, 12.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:10, 12.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:09, 12.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 12.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:09, 12.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:09, 12.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 12.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 22.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 22.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 22.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 22.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 22.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 22.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 22.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 22.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 27.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 34.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 34.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 34.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 34.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 40.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 40.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 40.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 40.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 40.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 40.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 45.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 47.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 47.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 47.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 47.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 47.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 47.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 47.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 47.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 47.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 54.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 61.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 61.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 61.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 61.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 61.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 61.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 61.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 61.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 61.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 61.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 67.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 67.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 67.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 67.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 67.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 67.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 67.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 67.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 67.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 57.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 43.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 43.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 43.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 43.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 43.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 43.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 43.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 45.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 45.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 45.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 45.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 45.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 45.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 45.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 41.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 41.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 41.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 41.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 41.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 41.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 41.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 43.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 43.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 43.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 43.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 43.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 43.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.51s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 43.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 43.06 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.02s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  3.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 43.06 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.02s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.02s/ file]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 23.07 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.02s/ url]
0 examples [00:00, ? examples/s]2020-07-21 06:11:54.157198: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-21 06:11:54.168254: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-21 06:11:54.168459: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a0a51ef8f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-21 06:11:54.168480: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  4.93 examples/s]59 examples [00:00,  7.02 examples/s]75 examples [00:00,  9.78 examples/s]145 examples [00:00, 13.88 examples/s]222 examples [00:00, 19.68 examples/s]268 examples [00:00, 27.58 examples/s]343 examples [00:00, 38.78 examples/s]437 examples [00:00, 54.43 examples/s]503 examples [00:01, 74.08 examples/s]591 examples [00:01, 102.14 examples/s]660 examples [00:01, 136.03 examples/s]728 examples [00:01, 178.91 examples/s]799 examples [00:01, 230.60 examples/s]887 examples [00:01, 295.90 examples/s]961 examples [00:01, 346.85 examples/s]1030 examples [00:01, 365.09 examples/s]1096 examples [00:02, 421.52 examples/s]1172 examples [00:02, 483.78 examples/s]1238 examples [00:02, 508.69 examples/s]1318 examples [00:02, 570.55 examples/s]1394 examples [00:02, 615.78 examples/s]1466 examples [00:02, 642.65 examples/s]1537 examples [00:02, 633.27 examples/s]1623 examples [00:02, 687.59 examples/s]1698 examples [00:02, 703.40 examples/s]1772 examples [00:02, 643.56 examples/s]1840 examples [00:03, 616.15 examples/s]1905 examples [00:03, 557.89 examples/s]1964 examples [00:03, 562.35 examples/s]2049 examples [00:03, 625.46 examples/s]2119 examples [00:03, 612.74 examples/s]2190 examples [00:03, 637.42 examples/s]2268 examples [00:03, 672.76 examples/s]2344 examples [00:03, 696.62 examples/s]2416 examples [00:04, 506.27 examples/s]2476 examples [00:04, 478.29 examples/s]2550 examples [00:04, 535.02 examples/s]2611 examples [00:04, 541.12 examples/s]2693 examples [00:04, 602.37 examples/s]2768 examples [00:04, 625.23 examples/s]2835 examples [00:04, 635.46 examples/s]2909 examples [00:04, 661.64 examples/s]2987 examples [00:04, 689.05 examples/s]3067 examples [00:05, 717.94 examples/s]3148 examples [00:05, 739.91 examples/s]3234 examples [00:05, 772.05 examples/s]3320 examples [00:05, 796.17 examples/s]3402 examples [00:05, 802.93 examples/s]3488 examples [00:05, 817.26 examples/s]3571 examples [00:05, 696.42 examples/s]3657 examples [00:05, 737.61 examples/s]3734 examples [00:05, 721.10 examples/s]3816 examples [00:06, 747.50 examples/s]3898 examples [00:06, 767.35 examples/s]3977 examples [00:06, 770.28 examples/s]4056 examples [00:06, 663.41 examples/s]4128 examples [00:06, 677.00 examples/s]4206 examples [00:06, 703.84 examples/s]4279 examples [00:06, 662.37 examples/s]4348 examples [00:06, 659.84 examples/s]4416 examples [00:06, 643.97 examples/s]4504 examples [00:07, 700.10 examples/s]4577 examples [00:07, 614.36 examples/s]4661 examples [00:07, 667.64 examples/s]4752 examples [00:07, 724.11 examples/s]4839 examples [00:07, 760.49 examples/s]4919 examples [00:07, 588.91 examples/s]4987 examples [00:07, 593.79 examples/s]5057 examples [00:07, 621.16 examples/s]5124 examples [00:08, 429.87 examples/s]5206 examples [00:08, 501.14 examples/s]5292 examples [00:08, 572.61 examples/s]5381 examples [00:08, 640.00 examples/s]5463 examples [00:08, 684.29 examples/s]5550 examples [00:08, 730.88 examples/s]5636 examples [00:08, 764.74 examples/s]5720 examples [00:08, 783.85 examples/s]5811 examples [00:09, 816.59 examples/s]5896 examples [00:09, 809.20 examples/s]5984 examples [00:09, 827.09 examples/s]6069 examples [00:09, 809.81 examples/s]6156 examples [00:09, 826.83 examples/s]6240 examples [00:09, 826.02 examples/s]6325 examples [00:09, 832.90 examples/s]6414 examples [00:09, 847.49 examples/s]6500 examples [00:09, 838.36 examples/s]6586 examples [00:09, 843.90 examples/s]6671 examples [00:10, 823.07 examples/s]6763 examples [00:10, 848.27 examples/s]6852 examples [00:10, 860.06 examples/s]6944 examples [00:10, 875.70 examples/s]7035 examples [00:10, 884.61 examples/s]7124 examples [00:10, 674.16 examples/s]7200 examples [00:10, 561.37 examples/s]7286 examples [00:10, 625.63 examples/s]7367 examples [00:11, 670.62 examples/s]7444 examples [00:11, 697.16 examples/s]7523 examples [00:11, 719.51 examples/s]7608 examples [00:11, 752.51 examples/s]7687 examples [00:11, 737.01 examples/s]7771 examples [00:11, 763.68 examples/s]7857 examples [00:11, 789.30 examples/s]7944 examples [00:11, 811.18 examples/s]8030 examples [00:11, 822.71 examples/s]8116 examples [00:11, 831.96 examples/s]8200 examples [00:12, 806.48 examples/s]8282 examples [00:12, 798.00 examples/s]8363 examples [00:12, 615.07 examples/s]8432 examples [00:12, 500.60 examples/s]8512 examples [00:12, 562.76 examples/s]8600 examples [00:12, 630.14 examples/s]8690 examples [00:12, 691.43 examples/s]8774 examples [00:12, 728.90 examples/s]8857 examples [00:13, 754.41 examples/s]8944 examples [00:13, 784.32 examples/s]9026 examples [00:13, 788.79 examples/s]9108 examples [00:13, 794.45 examples/s]9194 examples [00:13, 810.69 examples/s]9281 examples [00:13, 825.64 examples/s]9367 examples [00:13, 831.26 examples/s]9451 examples [00:13, 818.82 examples/s]9534 examples [00:13, 816.17 examples/s]9616 examples [00:13, 799.36 examples/s]9697 examples [00:14, 759.09 examples/s]9774 examples [00:14, 754.12 examples/s]9861 examples [00:14, 784.06 examples/s]9946 examples [00:14, 801.63 examples/s]10027 examples [00:14, 740.20 examples/s]10110 examples [00:14, 763.18 examples/s]10192 examples [00:14, 777.66 examples/s]10271 examples [00:14, 780.18 examples/s]10351 examples [00:14, 785.93 examples/s]10431 examples [00:15, 779.99 examples/s]10518 examples [00:15, 802.76 examples/s]10609 examples [00:15, 832.07 examples/s]10693 examples [00:15, 833.55 examples/s]10785 examples [00:15, 855.48 examples/s]10876 examples [00:15, 870.37 examples/s]10964 examples [00:15, 871.20 examples/s]11052 examples [00:15, 864.23 examples/s]11139 examples [00:15, 854.73 examples/s]11225 examples [00:15, 831.44 examples/s]11312 examples [00:16, 840.64 examples/s]11397 examples [00:16, 824.93 examples/s]11480 examples [00:16, 823.84 examples/s]11563 examples [00:16, 729.35 examples/s]11639 examples [00:16, 725.39 examples/s]11714 examples [00:16, 607.74 examples/s]11798 examples [00:16, 661.16 examples/s]11869 examples [00:16, 656.57 examples/s]11954 examples [00:17, 702.81 examples/s]12028 examples [00:17, 612.06 examples/s]12094 examples [00:17, 604.68 examples/s]12177 examples [00:17, 657.45 examples/s]12263 examples [00:17, 706.30 examples/s]12349 examples [00:17, 745.39 examples/s]12427 examples [00:17, 675.04 examples/s]12505 examples [00:17, 701.60 examples/s]12578 examples [00:17, 658.12 examples/s]12652 examples [00:18, 677.09 examples/s]12729 examples [00:18, 698.59 examples/s]12808 examples [00:18, 723.55 examples/s]12894 examples [00:18, 758.62 examples/s]12980 examples [00:18, 785.62 examples/s]13060 examples [00:18, 788.54 examples/s]13143 examples [00:18, 800.30 examples/s]13225 examples [00:18, 805.01 examples/s]13306 examples [00:18, 778.43 examples/s]13388 examples [00:18, 789.79 examples/s]13480 examples [00:19, 824.69 examples/s]13572 examples [00:19, 849.00 examples/s]13658 examples [00:19, 835.21 examples/s]13743 examples [00:19, 824.35 examples/s]13832 examples [00:19, 842.84 examples/s]13921 examples [00:19, 854.49 examples/s]14007 examples [00:19, 732.98 examples/s]14091 examples [00:19, 758.32 examples/s]14170 examples [00:19, 685.61 examples/s]14246 examples [00:20, 700.13 examples/s]14319 examples [00:20, 681.55 examples/s]14389 examples [00:20, 646.15 examples/s]14461 examples [00:20, 666.22 examples/s]14529 examples [00:20, 622.11 examples/s]14614 examples [00:20, 675.40 examples/s]14703 examples [00:20, 727.63 examples/s]14779 examples [00:20, 693.95 examples/s]14851 examples [00:20, 672.58 examples/s]14933 examples [00:21, 710.44 examples/s]15025 examples [00:21, 760.69 examples/s]15104 examples [00:21, 672.55 examples/s]15175 examples [00:21, 633.66 examples/s]15242 examples [00:21, 596.52 examples/s]15305 examples [00:21, 569.93 examples/s]15393 examples [00:21, 636.57 examples/s]15483 examples [00:21, 696.65 examples/s]15558 examples [00:22, 661.45 examples/s]15628 examples [00:22, 623.92 examples/s]15697 examples [00:22, 641.11 examples/s]15764 examples [00:22, 584.53 examples/s]15851 examples [00:22, 646.98 examples/s]15943 examples [00:22, 708.96 examples/s]16025 examples [00:22, 729.57 examples/s]16113 examples [00:22, 766.97 examples/s]16196 examples [00:22, 783.88 examples/s]16277 examples [00:23, 720.94 examples/s]16352 examples [00:23, 689.80 examples/s]16441 examples [00:23, 737.77 examples/s]16518 examples [00:23, 710.76 examples/s]16601 examples [00:23, 740.58 examples/s]16677 examples [00:23, 610.95 examples/s]16756 examples [00:23, 654.74 examples/s]16826 examples [00:23, 650.64 examples/s]16895 examples [00:23, 641.45 examples/s]16985 examples [00:24, 701.75 examples/s]17061 examples [00:24, 718.07 examples/s]17136 examples [00:24, 705.12 examples/s]17222 examples [00:24, 744.19 examples/s]17299 examples [00:24, 630.39 examples/s]17384 examples [00:24, 682.38 examples/s]17457 examples [00:24, 640.56 examples/s]17542 examples [00:24, 691.47 examples/s]17616 examples [00:25, 705.16 examples/s]17690 examples [00:25, 679.85 examples/s]17775 examples [00:25, 653.48 examples/s]17858 examples [00:25, 697.03 examples/s]17943 examples [00:25, 736.64 examples/s]18019 examples [00:25, 736.15 examples/s]18094 examples [00:25, 709.24 examples/s]18181 examples [00:25, 750.35 examples/s]18258 examples [00:25, 692.81 examples/s]18333 examples [00:26, 694.89 examples/s]18414 examples [00:26, 724.04 examples/s]18498 examples [00:26, 753.84 examples/s]18575 examples [00:26, 688.71 examples/s]18665 examples [00:26, 739.89 examples/s]18742 examples [00:26, 739.04 examples/s]18818 examples [00:26, 711.09 examples/s]18891 examples [00:26, 660.55 examples/s]18963 examples [00:26, 676.80 examples/s]19049 examples [00:27, 722.48 examples/s]19124 examples [00:27, 683.36 examples/s]19197 examples [00:27, 679.07 examples/s]19267 examples [00:27, 499.78 examples/s]19326 examples [00:27, 499.32 examples/s]19382 examples [00:27, 499.07 examples/s]19436 examples [00:27, 497.62 examples/s]19491 examples [00:27, 511.58 examples/s]19569 examples [00:28, 569.71 examples/s]19649 examples [00:28, 622.04 examples/s]19721 examples [00:28, 639.07 examples/s]19807 examples [00:28, 691.37 examples/s]19880 examples [00:28, 630.79 examples/s]19947 examples [00:28, 607.45 examples/s]20016 examples [00:28, 628.01 examples/s]20097 examples [00:28, 672.53 examples/s]20185 examples [00:28, 722.79 examples/s]20260 examples [00:29, 650.96 examples/s]20329 examples [00:29, 626.46 examples/s]20418 examples [00:29, 686.32 examples/s]20498 examples [00:29, 715.40 examples/s]20590 examples [00:29, 764.99 examples/s]20670 examples [00:29, 709.70 examples/s]20754 examples [00:29, 742.43 examples/s]20840 examples [00:29, 773.45 examples/s]20928 examples [00:29, 800.36 examples/s]21015 examples [00:29, 819.79 examples/s]21103 examples [00:30, 833.70 examples/s]21188 examples [00:30, 802.83 examples/s]21270 examples [00:30, 704.79 examples/s]21359 examples [00:30, 750.30 examples/s]21446 examples [00:30, 781.28 examples/s]21527 examples [00:30, 747.18 examples/s]21610 examples [00:30, 769.56 examples/s]21697 examples [00:30, 796.39 examples/s]21778 examples [00:30, 738.22 examples/s]21865 examples [00:31, 773.16 examples/s]21954 examples [00:31, 804.19 examples/s]22036 examples [00:31, 564.14 examples/s]22113 examples [00:31, 613.12 examples/s]22184 examples [00:31, 572.23 examples/s]22257 examples [00:31, 611.82 examples/s]22347 examples [00:31, 676.42 examples/s]22430 examples [00:31, 716.14 examples/s]22514 examples [00:32, 746.83 examples/s]22600 examples [00:32, 776.41 examples/s]22688 examples [00:32, 802.62 examples/s]22771 examples [00:32, 691.87 examples/s]22858 examples [00:32, 737.07 examples/s]22941 examples [00:32, 762.67 examples/s]23028 examples [00:32, 789.69 examples/s]23110 examples [00:32, 774.86 examples/s]23190 examples [00:33, 665.09 examples/s]23266 examples [00:33, 690.60 examples/s]23339 examples [00:33, 581.28 examples/s]23417 examples [00:33, 628.04 examples/s]23485 examples [00:33, 519.90 examples/s]23563 examples [00:33, 577.32 examples/s]23636 examples [00:33, 615.56 examples/s]23712 examples [00:33, 635.83 examples/s]23780 examples [00:33, 633.59 examples/s]23847 examples [00:34, 620.69 examples/s]23924 examples [00:34, 658.41 examples/s]24007 examples [00:34, 700.62 examples/s]24080 examples [00:34, 699.38 examples/s]24166 examples [00:34, 739.99 examples/s]24256 examples [00:34, 779.89 examples/s]24344 examples [00:34, 806.52 examples/s]24433 examples [00:34, 829.18 examples/s]24522 examples [00:34, 845.48 examples/s]24608 examples [00:35, 836.55 examples/s]24693 examples [00:35, 836.41 examples/s]24778 examples [00:35, 809.85 examples/s]24860 examples [00:35, 778.02 examples/s]24939 examples [00:35, 778.40 examples/s]25025 examples [00:35, 799.43 examples/s]25112 examples [00:35, 818.10 examples/s]25199 examples [00:35, 830.55 examples/s]25286 examples [00:35, 841.37 examples/s]25371 examples [00:36, 698.27 examples/s]25452 examples [00:36, 726.63 examples/s]25540 examples [00:36, 766.40 examples/s]25626 examples [00:36, 791.42 examples/s]25708 examples [00:36, 755.48 examples/s]25789 examples [00:36, 770.85 examples/s]25868 examples [00:36, 744.81 examples/s]25944 examples [00:36, 702.92 examples/s]26016 examples [00:36, 664.96 examples/s]26084 examples [00:37, 613.06 examples/s]26148 examples [00:37, 593.88 examples/s]26236 examples [00:37, 657.52 examples/s]26323 examples [00:37, 707.67 examples/s]26397 examples [00:37, 660.05 examples/s]26478 examples [00:37, 680.80 examples/s]26564 examples [00:37, 726.12 examples/s]26655 examples [00:37, 772.63 examples/s]26744 examples [00:37, 802.20 examples/s]26827 examples [00:38, 670.31 examples/s]26909 examples [00:38, 709.01 examples/s]26992 examples [00:38, 738.71 examples/s]27071 examples [00:38, 753.37 examples/s]27158 examples [00:38, 784.76 examples/s]27248 examples [00:38, 815.46 examples/s]27340 examples [00:38, 843.30 examples/s]27427 examples [00:38, 850.50 examples/s]27517 examples [00:38, 863.19 examples/s]27605 examples [00:38, 847.97 examples/s]27692 examples [00:39, 852.81 examples/s]27778 examples [00:39, 669.89 examples/s]27852 examples [00:39, 575.16 examples/s]27937 examples [00:39, 635.29 examples/s]28012 examples [00:39, 664.29 examples/s]28084 examples [00:39, 534.35 examples/s]28146 examples [00:39, 543.81 examples/s]28222 examples [00:40, 593.50 examples/s]28287 examples [00:40, 601.10 examples/s]28351 examples [00:40, 522.67 examples/s]28417 examples [00:40, 555.91 examples/s]28490 examples [00:40, 598.67 examples/s]28581 examples [00:40, 665.52 examples/s]28669 examples [00:40, 717.78 examples/s]28755 examples [00:40, 755.08 examples/s]28835 examples [00:40, 613.86 examples/s]28920 examples [00:41, 605.13 examples/s]28995 examples [00:41, 640.71 examples/s]29064 examples [00:41, 619.49 examples/s]29129 examples [00:41, 619.51 examples/s]29216 examples [00:41, 676.27 examples/s]29296 examples [00:41, 707.91 examples/s]29370 examples [00:41, 705.41 examples/s]29455 examples [00:41, 742.99 examples/s]29532 examples [00:41, 738.26 examples/s]29623 examples [00:42, 781.77 examples/s]29712 examples [00:42, 810.57 examples/s]29795 examples [00:42, 798.67 examples/s]29887 examples [00:42, 830.13 examples/s]29972 examples [00:42, 733.35 examples/s]30049 examples [00:42, 626.59 examples/s]30117 examples [00:42, 611.47 examples/s]30182 examples [00:42, 586.12 examples/s]30266 examples [00:43, 644.35 examples/s]30356 examples [00:43, 703.98 examples/s]30435 examples [00:43, 706.62 examples/s]30509 examples [00:43, 666.80 examples/s]30579 examples [00:43, 438.17 examples/s]30652 examples [00:43, 497.15 examples/s]30716 examples [00:43, 531.69 examples/s]30778 examples [00:44, 448.78 examples/s]30845 examples [00:44, 497.78 examples/s]30903 examples [00:44, 464.06 examples/s]30980 examples [00:44, 491.47 examples/s]31034 examples [00:44, 426.18 examples/s]31096 examples [00:44, 469.17 examples/s]31166 examples [00:44, 520.15 examples/s]31223 examples [00:44, 451.68 examples/s]31274 examples [00:45, 370.34 examples/s]31317 examples [00:45, 361.44 examples/s]31394 examples [00:45, 421.92 examples/s]31444 examples [00:45, 412.22 examples/s]31494 examples [00:45, 378.87 examples/s]31536 examples [00:45, 380.55 examples/s]31591 examples [00:45, 416.61 examples/s]31636 examples [00:46, 392.26 examples/s]31678 examples [00:46, 377.58 examples/s]31722 examples [00:46, 359.91 examples/s]31761 examples [00:46, 362.64 examples/s]31802 examples [00:46, 374.60 examples/s]31848 examples [00:46, 395.76 examples/s]31905 examples [00:46, 434.57 examples/s]31951 examples [00:46, 425.28 examples/s]32010 examples [00:46, 462.78 examples/s]32059 examples [00:47, 441.91 examples/s]32105 examples [00:47, 436.38 examples/s]32161 examples [00:47, 462.01 examples/s]32239 examples [00:47, 526.16 examples/s]32296 examples [00:47, 504.76 examples/s]32350 examples [00:47, 462.67 examples/s]32399 examples [00:47, 414.34 examples/s]32470 examples [00:47, 472.17 examples/s]32523 examples [00:47, 471.08 examples/s]32574 examples [00:48, 436.55 examples/s]32642 examples [00:48, 489.05 examples/s]32727 examples [00:48, 559.91 examples/s]32808 examples [00:48, 615.59 examples/s]32893 examples [00:48, 670.13 examples/s]32976 examples [00:48, 709.93 examples/s]33064 examples [00:48, 751.64 examples/s]33144 examples [00:48, 738.76 examples/s]33221 examples [00:48, 689.06 examples/s]33293 examples [00:49, 696.34 examples/s]33380 examples [00:49, 738.94 examples/s]33456 examples [00:49, 507.91 examples/s]33528 examples [00:49, 555.54 examples/s]33593 examples [00:49, 472.19 examples/s]33649 examples [00:49, 424.42 examples/s]33731 examples [00:49, 496.22 examples/s]33791 examples [00:50, 507.58 examples/s]33849 examples [00:50, 499.49 examples/s]33913 examples [00:50, 534.39 examples/s]33971 examples [00:50, 544.16 examples/s]34057 examples [00:50, 610.39 examples/s]34140 examples [00:50, 662.50 examples/s]34211 examples [00:50, 538.58 examples/s]34272 examples [00:50, 548.21 examples/s]34332 examples [00:51, 520.73 examples/s]34416 examples [00:51, 586.81 examples/s]34482 examples [00:51, 604.43 examples/s]34547 examples [00:51, 517.96 examples/s]34604 examples [00:51, 478.24 examples/s]34691 examples [00:51, 552.20 examples/s]34754 examples [00:51, 480.19 examples/s]34809 examples [00:52, 393.95 examples/s]34856 examples [00:52, 397.92 examples/s]34924 examples [00:52, 451.15 examples/s]34976 examples [00:52, 455.36 examples/s]35026 examples [00:52, 376.83 examples/s]35095 examples [00:52, 435.64 examples/s]35146 examples [00:52, 416.77 examples/s]35214 examples [00:52, 444.27 examples/s]35263 examples [00:52, 454.41 examples/s]35334 examples [00:53, 508.47 examples/s]35389 examples [00:53, 480.65 examples/s]35441 examples [00:53, 446.50 examples/s]35507 examples [00:53, 493.87 examples/s]35577 examples [00:53, 540.66 examples/s]35666 examples [00:53, 612.47 examples/s]35737 examples [00:53, 636.73 examples/s]35817 examples [00:53, 676.73 examples/s]35891 examples [00:53, 689.59 examples/s]35963 examples [00:54, 679.29 examples/s]36041 examples [00:54, 706.06 examples/s]36114 examples [00:54, 712.18 examples/s]36203 examples [00:54, 757.14 examples/s]36281 examples [00:54, 735.29 examples/s]36361 examples [00:54, 752.40 examples/s]36438 examples [00:54, 719.09 examples/s]36511 examples [00:54, 713.48 examples/s]36594 examples [00:54, 742.61 examples/s]36673 examples [00:55, 755.97 examples/s]36750 examples [00:55, 750.92 examples/s]36826 examples [00:55, 694.80 examples/s]36897 examples [00:55, 635.04 examples/s]36963 examples [00:55, 548.68 examples/s]37050 examples [00:55, 616.76 examples/s]37126 examples [00:55, 651.73 examples/s]37201 examples [00:55, 676.77 examples/s]37285 examples [00:55, 717.23 examples/s]37360 examples [00:56, 724.57 examples/s]37444 examples [00:56, 754.26 examples/s]37534 examples [00:56, 792.38 examples/s]37618 examples [00:56, 804.38 examples/s]37700 examples [00:56, 789.71 examples/s]37780 examples [00:56, 771.36 examples/s]37858 examples [00:56, 770.35 examples/s]37936 examples [00:56, 758.09 examples/s]38023 examples [00:56, 788.29 examples/s]38109 examples [00:56, 808.07 examples/s]38198 examples [00:57, 829.39 examples/s]38285 examples [00:57, 838.65 examples/s]38370 examples [00:57, 808.69 examples/s]38453 examples [00:57, 806.13 examples/s]38534 examples [00:57, 800.94 examples/s]38615 examples [00:57, 773.52 examples/s]38693 examples [00:57, 575.39 examples/s]38759 examples [00:57, 551.73 examples/s]38845 examples [00:58, 617.89 examples/s]38922 examples [00:58, 655.51 examples/s]39011 examples [00:58, 709.96 examples/s]39101 examples [00:58, 757.36 examples/s]39182 examples [00:58, 762.96 examples/s]39264 examples [00:58, 776.94 examples/s]39344 examples [00:58, 622.89 examples/s]39413 examples [00:58, 619.35 examples/s]39480 examples [00:59, 555.85 examples/s]39562 examples [00:59, 614.91 examples/s]39629 examples [00:59, 562.66 examples/s]39705 examples [00:59, 608.90 examples/s]39790 examples [00:59, 663.84 examples/s]39865 examples [00:59, 686.32 examples/s]39937 examples [00:59, 688.07 examples/s]40009 examples [00:59, 685.34 examples/s]40088 examples [00:59, 712.32 examples/s]40167 examples [00:59, 733.79 examples/s]40254 examples [01:00, 769.48 examples/s]40337 examples [01:00, 786.58 examples/s]40425 examples [01:00, 810.33 examples/s]40507 examples [01:00, 752.52 examples/s]40595 examples [01:00, 784.60 examples/s]40682 examples [01:00, 806.25 examples/s]40764 examples [01:00, 770.24 examples/s]40852 examples [01:00, 800.14 examples/s]40934 examples [01:00, 776.74 examples/s]41013 examples [01:01, 779.78 examples/s]41096 examples [01:01, 793.82 examples/s]41180 examples [01:01, 806.67 examples/s]41266 examples [01:01, 820.44 examples/s]41349 examples [01:01, 720.92 examples/s]41434 examples [01:01, 753.99 examples/s]41517 examples [01:01, 775.17 examples/s]41597 examples [01:01, 715.50 examples/s]41671 examples [01:01, 701.91 examples/s]41743 examples [01:02, 611.49 examples/s]41829 examples [01:02, 669.18 examples/s]41900 examples [01:02, 599.23 examples/s]41975 examples [01:02, 637.01 examples/s]42056 examples [01:02, 657.30 examples/s]42125 examples [01:02, 639.70 examples/s]42212 examples [01:02, 693.12 examples/s]42287 examples [01:02, 695.10 examples/s]42359 examples [01:02, 692.66 examples/s]42430 examples [01:03, 622.38 examples/s]42495 examples [01:03, 619.59 examples/s]42578 examples [01:03, 669.23 examples/s]42649 examples [01:03, 669.42 examples/s]42724 examples [01:03, 691.07 examples/s]42810 examples [01:03, 732.28 examples/s]42897 examples [01:03, 768.26 examples/s]42986 examples [01:03, 801.05 examples/s]43068 examples [01:03, 791.66 examples/s]43156 examples [01:04, 813.67 examples/s]43239 examples [01:04, 681.63 examples/s]43318 examples [01:04, 708.93 examples/s]43394 examples [01:04, 722.89 examples/s]43478 examples [01:04, 753.96 examples/s]43563 examples [01:04, 779.59 examples/s]43649 examples [01:04, 800.09 examples/s]43731 examples [01:04, 778.63 examples/s]43810 examples [01:04, 752.91 examples/s]43897 examples [01:05, 783.89 examples/s]43977 examples [01:05, 769.67 examples/s]44055 examples [01:05, 641.46 examples/s]44132 examples [01:05, 674.74 examples/s]44203 examples [01:05, 590.80 examples/s]44287 examples [01:05, 647.22 examples/s]44366 examples [01:05, 682.55 examples/s]44456 examples [01:05, 733.40 examples/s]44534 examples [01:05, 737.94 examples/s]44611 examples [01:06, 740.58 examples/s]44698 examples [01:06, 774.69 examples/s]44782 examples [01:06, 791.91 examples/s]44869 examples [01:06, 813.51 examples/s]44956 examples [01:06, 829.64 examples/s]45043 examples [01:06, 841.23 examples/s]45132 examples [01:06, 853.99 examples/s]45218 examples [01:06, 854.15 examples/s]45306 examples [01:06, 860.09 examples/s]45393 examples [01:06, 823.81 examples/s]45476 examples [01:07, 700.77 examples/s]45550 examples [01:07, 710.91 examples/s]45624 examples [01:07, 716.71 examples/s]45698 examples [01:07, 655.13 examples/s]45783 examples [01:07, 703.50 examples/s]45870 examples [01:07, 745.35 examples/s]45955 examples [01:07, 772.75 examples/s]46037 examples [01:07, 785.22 examples/s]46117 examples [01:07, 765.80 examples/s]46206 examples [01:08, 798.49 examples/s]46295 examples [01:08, 822.53 examples/s]46383 examples [01:08, 838.31 examples/s]46468 examples [01:08, 825.47 examples/s]46552 examples [01:08, 814.88 examples/s]46634 examples [01:08, 789.47 examples/s]46718 examples [01:08, 803.62 examples/s]46806 examples [01:08, 823.56 examples/s]46896 examples [01:08, 843.76 examples/s]46981 examples [01:09, 817.45 examples/s]47070 examples [01:09, 836.14 examples/s]47155 examples [01:09, 835.05 examples/s]47239 examples [01:09, 828.34 examples/s]47326 examples [01:09, 839.15 examples/s]47413 examples [01:09, 848.13 examples/s]47498 examples [01:09, 694.73 examples/s]47573 examples [01:09, 611.73 examples/s]47655 examples [01:09, 660.57 examples/s]47726 examples [01:10, 531.31 examples/s]47800 examples [01:10, 578.91 examples/s]47865 examples [01:10, 561.73 examples/s]47957 examples [01:10, 635.07 examples/s]48044 examples [01:10, 690.97 examples/s]48120 examples [01:10, 704.97 examples/s]48195 examples [01:10, 700.07 examples/s]48268 examples [01:10, 655.00 examples/s]48347 examples [01:11, 688.81 examples/s]48419 examples [01:11, 634.48 examples/s]48500 examples [01:11, 676.80 examples/s]48575 examples [01:11, 695.23 examples/s]48662 examples [01:11, 738.65 examples/s]48740 examples [01:11, 748.61 examples/s]48824 examples [01:11, 773.11 examples/s]48907 examples [01:11, 787.11 examples/s]48994 examples [01:11, 808.18 examples/s]49076 examples [01:12, 733.61 examples/s]49154 examples [01:12, 746.67 examples/s]49233 examples [01:12, 695.09 examples/s]49308 examples [01:12, 710.57 examples/s]49383 examples [01:12, 721.86 examples/s]49465 examples [01:12, 738.84 examples/s]49545 examples [01:12, 754.83 examples/s]49627 examples [01:12, 773.25 examples/s]49705 examples [01:12, 773.34 examples/s]49783 examples [01:12, 764.34 examples/s]49860 examples [01:13, 764.74 examples/s]49937 examples [01:13, 757.96 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s]  4%|▎         | 1780/50000 [00:00<00:02, 17799.55 examples/s] 22%|██▏       | 11134/50000 [00:00<00:01, 23510.48 examples/s] 39%|███▉      | 19627/50000 [00:00<00:01, 29003.27 examples/s] 61%|██████    | 30556/50000 [00:00<00:00, 37201.90 examples/s] 73%|███████▎  | 36732/50000 [00:00<00:00, 41761.41 examples/s] 92%|█████████▏| 46005/50000 [00:00<00:00, 48406.98 examples/s]                                                               0 examples [00:00, ? examples/s]12 examples [00:00, 119.94 examples/s]63 examples [00:00, 154.74 examples/s]128 examples [00:00, 194.54 examples/s]175 examples [00:00, 235.17 examples/s]223 examples [00:00, 263.07 examples/s]262 examples [00:00, 276.34 examples/s]323 examples [00:00, 327.70 examples/s]374 examples [00:00, 366.62 examples/s]418 examples [00:01, 377.31 examples/s]490 examples [00:01, 439.27 examples/s]542 examples [00:01, 416.67 examples/s]621 examples [00:01, 465.05 examples/s]674 examples [00:01, 437.59 examples/s]756 examples [00:01, 506.34 examples/s]839 examples [00:01, 572.17 examples/s]930 examples [00:01, 642.84 examples/s]1022 examples [00:01, 705.43 examples/s]1106 examples [00:02, 740.25 examples/s]1191 examples [00:02, 769.35 examples/s]1281 examples [00:02, 788.70 examples/s]1364 examples [00:02, 778.97 examples/s]1449 examples [00:02, 797.40 examples/s]1535 examples [00:02, 813.94 examples/s]1623 examples [00:02, 831.13 examples/s]1708 examples [00:02, 729.77 examples/s]1784 examples [00:02, 680.58 examples/s]1855 examples [00:03, 532.65 examples/s]1916 examples [00:03, 535.38 examples/s]1975 examples [00:03, 443.14 examples/s]2026 examples [00:03, 449.49 examples/s]2076 examples [00:03, 461.44 examples/s]2126 examples [00:03, 440.09 examples/s]2178 examples [00:03, 439.31 examples/s]2224 examples [00:04, 355.00 examples/s]2294 examples [00:04, 416.16 examples/s]2373 examples [00:04, 484.93 examples/s]2450 examples [00:04, 544.38 examples/s]2514 examples [00:04, 525.72 examples/s]2584 examples [00:04, 566.69 examples/s]2669 examples [00:04, 628.95 examples/s]2755 examples [00:04, 683.83 examples/s]2842 examples [00:04, 730.37 examples/s]2929 examples [00:04, 766.45 examples/s]3013 examples [00:05, 786.55 examples/s]3095 examples [00:05, 714.65 examples/s]3177 examples [00:05, 741.71 examples/s]3265 examples [00:05, 776.69 examples/s]3345 examples [00:05, 711.32 examples/s]3423 examples [00:05, 729.82 examples/s]3511 examples [00:05, 767.67 examples/s]3596 examples [00:05, 790.43 examples/s]3684 examples [00:05, 813.98 examples/s]3767 examples [00:06, 787.75 examples/s]3847 examples [00:06, 765.00 examples/s]3935 examples [00:06, 793.73 examples/s]4023 examples [00:06, 815.74 examples/s]4106 examples [00:06, 816.36 examples/s]4189 examples [00:06, 812.38 examples/s]4271 examples [00:06, 529.10 examples/s]4357 examples [00:06, 596.94 examples/s]4429 examples [00:07, 477.54 examples/s]4517 examples [00:07, 552.64 examples/s]4605 examples [00:07, 620.76 examples/s]4679 examples [00:07, 633.97 examples/s]4758 examples [00:07, 672.85 examples/s]4845 examples [00:07, 721.24 examples/s]4932 examples [00:07, 758.90 examples/s]5016 examples [00:07, 780.67 examples/s]5098 examples [00:08, 626.82 examples/s]5171 examples [00:08, 652.87 examples/s]5242 examples [00:08, 619.32 examples/s]5308 examples [00:08, 558.91 examples/s]5393 examples [00:08, 622.87 examples/s]5481 examples [00:08, 682.68 examples/s]5573 examples [00:08, 739.09 examples/s]5662 examples [00:08, 778.29 examples/s]5744 examples [00:09, 765.48 examples/s]5824 examples [00:09, 760.74 examples/s]5903 examples [00:09, 674.35 examples/s]5974 examples [00:09, 508.28 examples/s]6040 examples [00:09, 535.55 examples/s]6100 examples [00:09, 524.10 examples/s]6175 examples [00:09, 575.55 examples/s]6257 examples [00:09, 630.82 examples/s]6342 examples [00:10, 683.13 examples/s]6427 examples [00:10, 725.62 examples/s]6514 examples [00:10, 762.45 examples/s]6596 examples [00:10, 777.14 examples/s]6677 examples [00:10, 764.93 examples/s]6766 examples [00:10, 796.23 examples/s]6852 examples [00:10, 812.01 examples/s]6937 examples [00:10, 822.26 examples/s]7022 examples [00:10, 828.25 examples/s]7113 examples [00:10, 850.49 examples/s]7199 examples [00:11, 822.73 examples/s]7282 examples [00:11, 770.76 examples/s]7361 examples [00:11, 703.06 examples/s]7441 examples [00:11, 727.72 examples/s]7525 examples [00:11, 757.07 examples/s]7611 examples [00:11, 782.13 examples/s]7696 examples [00:11, 797.05 examples/s]7783 examples [00:11, 815.33 examples/s]7869 examples [00:11, 826.17 examples/s]7953 examples [00:12, 763.69 examples/s]8031 examples [00:12, 764.77 examples/s]8117 examples [00:12, 789.21 examples/s]8198 examples [00:12, 794.34 examples/s]8279 examples [00:12, 603.79 examples/s]8353 examples [00:12, 638.23 examples/s]8423 examples [00:12, 534.56 examples/s]8508 examples [00:12, 600.98 examples/s]8581 examples [00:13, 634.33 examples/s]8666 examples [00:13, 685.60 examples/s]8752 examples [00:13, 727.79 examples/s]8830 examples [00:13, 730.15 examples/s]8914 examples [00:13, 758.94 examples/s]8999 examples [00:13, 781.80 examples/s]9084 examples [00:13, 800.71 examples/s]9172 examples [00:13, 820.50 examples/s]9262 examples [00:13, 841.74 examples/s]9350 examples [00:13, 850.58 examples/s]9436 examples [00:14, 846.08 examples/s]9525 examples [00:14, 856.13 examples/s]9611 examples [00:14, 836.19 examples/s]9695 examples [00:14, 826.60 examples/s]9783 examples [00:14, 840.56 examples/s]9870 examples [00:14, 846.91 examples/s]9955 examples [00:14, 831.17 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteD8UZUS/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteD8UZUS/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bb78d7950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bb78d7950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bb78d7950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b3dc27208>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b3dc27cf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bb78d7950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bb78d7950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bb78d7950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bb1c492b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bb5e494a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1b302a6e18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1b302a6e18> 

  function with postional parmater data_info <function split_train_valid at 0x7f1b302a6e18> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1b302a6f28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1b302a6f28> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1b302a6f28> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=23f542977342ed4d311e5a739828e46d795746c50a71142eec7d976e7e84134e
  Stored in directory: /tmp/pip-ephem-wheel-cache-ks69cz6a/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:29:35, 11.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:34:44, 16.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:15:41, 23.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:01<7:11:23, 33.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.15M/862M [00:01<5:01:51, 47.5kB/s].vector_cache/glove.6B.zip:   1%|          | 4.46M/862M [00:01<3:30:58, 67.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.79M/862M [00:01<2:26:50, 96.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.6M/862M [00:01<1:42:28, 138kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.9M/862M [00:01<1:11:22, 197kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.3M/862M [00:01<49:52, 281kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.5M/862M [00:01<34:46, 400kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.9M/862M [00:01<24:22, 568kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:02<17:02, 808kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:02<11:59, 1.14MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.3M/862M [00:02<08:25, 1.62MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.0M/862M [00:02<05:59, 2.26MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<04:44, 2.84MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.5M/862M [00:03<03:26, 3.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<10:08, 1.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<08:58, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<06:39, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:05<04:48, 2.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<1:04:45, 206kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<46:57, 284kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<33:14, 401kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<25:57, 512kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<19:42, 674kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<14:06, 940kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<12:40, 1.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<11:36, 1.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<08:47, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<08:16, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<07:07, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<05:17, 2.48MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<06:43, 1.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<06:38, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:15<05:06, 2.56MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<06:00, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<06:04, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:17<04:42, 2.76MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<05:41, 2.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<05:50, 2.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:19<04:29, 2.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<05:32, 2.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<05:46, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:20<04:26, 2.89MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<05:29, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<05:41, 2.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<04:26, 2.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<05:28, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<05:41, 2.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<04:27, 2.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<03:24, 3.73MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:12, 1.76MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:53, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:12, 2.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<03:53, 3.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:44, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:14, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<05:31, 2.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:10, 2.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:08, 2.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:44, 2.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:38, 2.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:45, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:28, 2.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:25, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:36, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:18, 2.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:18, 2.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:30, 2.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:18, 2.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:16, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:27, 2.25MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:10, 2.93MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<03:03, 3.99MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<44:52, 271kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<33:09, 367kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<23:33, 516kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<18:42, 648kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<14:50, 816kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<10:45, 1.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<07:38, 1.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<35:12, 342kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<26:24, 456kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<18:50, 638kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<13:16, 903kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:20:25, 149kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<57:59, 206kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<40:54, 292kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<28:42, 415kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<30:15, 394kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<22:57, 519kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<16:52, 705kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<11:55, 995kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<14:39, 808kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<12:13, 968kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<08:54, 1.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<06:27, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:55, 1.32MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:00, 1.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:01, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:21, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:18, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:49, 2.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:32, 2.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:32, 2.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:51, 2.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:41, 3.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:53, 1.96MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:56, 1.94MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:39, 2.48MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<05:18, 2.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:22, 2.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:13, 2.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<03:05, 3.69MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<14:02, 814kB/s] .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<11:27, 996kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<08:29, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:00, 1.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:09, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:23, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:55, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:41, 1.98MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:21, 2.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:02, 2.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<02:59, 3.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<1:01:49, 181kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<43:33, 257kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<32:27, 344kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<24:18, 458kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<17:20, 641kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<12:13, 906kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<28:30, 389kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<21:31, 514kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<15:23, 719kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<10:59, 1.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<11:48, 932kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<09:44, 1.13MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<07:13, 1.52MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:03, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:30, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:52, 2.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:32, 3.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<22:08, 491kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<17:15, 630kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<12:26, 872kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<10:40, 1.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:59, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:39, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:39, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:19, 1.70MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:48, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:19, 2.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:15, 2.03MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:03, 2.62MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<04:48, 2.21MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:41, 2.26MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<03:42, 2.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<02:47, 3.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:47, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:34, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:14, 2.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<03:08, 3.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<12:40, 825kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<10:22, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:34, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<05:23, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<36:20, 286kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<26:58, 385kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<19:13, 539kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<15:19, 673kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<12:13, 844kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<08:50, 1.16MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<06:22, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<08:33, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:28, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:35, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:47, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:32, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:09, 2.44MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<03:06, 3.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:06, 1.66MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:43, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:18, 2.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:07, 3.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<39:34, 254kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<29:07, 345kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<20:39, 485kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<14:36, 684kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<14:36, 683kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<11:29, 867kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<08:20, 1.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:52, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:56, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:12, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:27, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:11, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:56, 2.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<02:54, 3.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<07:38, 1.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:46, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:04, 1.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:21, 1.81MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<05:09, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:53, 2.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<02:50, 3.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<19:53, 484kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<15:18, 629kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<10:59, 874kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<07:47, 1.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<15:47, 606kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<12:25, 770kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<08:58, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<06:29, 1.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:21, 1.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<07:03, 1.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:14, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<03:47, 2.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<09:22, 1.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<07:52, 1.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:48, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:48, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:25, 1.72MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:05, 2.28MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<02:58, 3.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<11:21, 818kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<09:26, 983kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<06:55, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:31, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:40, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:13, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:52, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:30, 2.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:23, 2.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:14, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:16, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:16, 2.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:58, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:04, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:09, 2.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<03:52, 2.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:00, 2.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:07, 2.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:50, 2.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:58, 2.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:05, 2.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<03:46, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:00, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:06, 2.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<02:17, 3.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<11:14, 777kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<09:06, 959kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<06:37, 1.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<04:42, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<8:41:28, 16.6kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<6:07:32, 23.6kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<4:17:31, 33.6kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<3:00:16, 47.9kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<2:07:11, 67.6kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<1:30:13, 95.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<1:03:17, 135kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<44:10, 193kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<53:09, 160kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<39:40, 215kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<28:28, 299kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<20:00, 424kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<17:04, 495kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<13:09, 642kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<09:27, 892kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<06:45, 1.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<08:16, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:06, 1.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:15, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:49, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:48, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:21, 1.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:02, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:20, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:17, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:14, 2.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:21, 3.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<11:25, 716kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:19, 877kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:47, 1.20MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:14, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:33, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:07, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<02:58, 2.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<27:16, 295kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<20:15, 397kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<14:26, 555kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<10:09, 785kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<16:40, 478kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<12:37, 631kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<09:23, 847kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<06:39, 1.19MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<11:34, 683kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<09:14, 854kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:42, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<06:08, 1.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:13, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:52, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:18, 1.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:08, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:10, 2.44MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<03:38, 2.11MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<03:39, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:47, 2.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:01, 3.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<1:32:09, 82.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<1:05:36, 116kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<46:05, 165kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<33:28, 226kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<24:31, 308kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<17:21, 434kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<12:15, 613kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<11:24, 656kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<09:03, 826kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<06:33, 1.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<04:43, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:09, 1.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<05:23, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:30, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:18, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:00, 1.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<04:40, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:31, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:33, 2.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<08:18, 877kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<07:01, 1.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<05:10, 1.40MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:44, 1.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:54, 1.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:39, 1.55MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:22, 2.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:06, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:49, 1.23MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:38, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:21, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:56, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:29, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:21, 2.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:26, 2.88MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<07:02, 996kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<05:46, 1.21MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<04:26, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:10, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:23, 826kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:41, 1.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<04:51, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<03:30, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<10:52, 631kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<08:36, 797kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<06:13, 1.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:27, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:25, 1.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:29, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<04:04, 1.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:05, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:50, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:53, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:15, 2.04MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:14, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:30, 2.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:57, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:06, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:24, 2.72MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<01:46, 3.69MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<30:03, 217kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<21:56, 297kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<15:30, 419kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:22<10:53, 594kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<14:04, 459kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<10:37, 607kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<07:36, 844kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:36, 966kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:32, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:05, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:57, 2.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:50, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:17, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:19, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:26, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:18, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:32, 2.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<01:51, 3.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:23, 1.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:40, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:27, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:29, 2.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<06:41, 912kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<05:34, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:05, 1.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:57, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:42, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:47, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<02:01, 2.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<11:57, 499kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<09:03, 658kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<06:29, 915kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:44, 1.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:25, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:06, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:59, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:33, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:11, 1.83MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:28, 2.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:52, 3.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:57, 1.95MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:52, 2.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:11, 2.63MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<01:38, 3.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:27, 1.65MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:08, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:38, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:00, 2.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<01:28, 3.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<22:24, 251kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<16:19, 344kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<11:32, 485kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<09:10, 605kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<07:05, 783kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<05:04, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<03:37, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<08:57, 613kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<07:03, 776kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<05:04, 1.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<03:39, 1.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:43, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:56, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:55, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:08, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:49, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:06, 2.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:34, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:19, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:43, 1.93MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:59, 2.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:27, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:17, 1.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:31, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<01:48, 2.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<2:27:12, 34.9kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<1:43:41, 49.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<1:12:44, 70.4kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<50:40, 100kB/s]   .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<37:14, 136kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<26:52, 188kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<19:02, 265kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<13:24, 374kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<11:08, 449kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<08:15, 605kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<05:50, 850kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<05:19, 925kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<04:42, 1.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:39, 1.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:38, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:26, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:36, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:43, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:30, 1.90MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:07, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:34, 3.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:31, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:43, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:56, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<02:08, 2.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:56, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:37, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:57, 2.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:18, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:41, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:09, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:33, 2.91MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<04:01, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:28, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:34, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:37, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:28, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:51, 2.37MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:20, 3.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<24:22, 180kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<17:40, 247kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<12:28, 349kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<09:26, 456kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<07:13, 596kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<05:11, 824kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<04:23, 966kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:40, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:44, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:58, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:33, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:24, 1.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:35, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:57, 2.10MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:25, 2.87MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<06:43, 609kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<05:17, 774kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:56, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:48, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:52, 1.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:25, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:33, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:51, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<04:38, 853kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:49, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:47, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:01, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:44, 1.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:28, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:07, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:32, 2.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:28, 1.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:16, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:43, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:53, 1.98MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:58, 1.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:42, 2.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:15, 2.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:57, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:33, 1.43MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<02:03, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:35, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:43, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:44, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:20, 2.65MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:35, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:36, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:13, 2.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<00:55, 3.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<03:03, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:34, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:54, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:00, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:54, 1.78MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:36, 2.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:11, 2.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:43, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:40, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:20, 2.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<00:59, 3.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:45, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:39, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:14, 2.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:29, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:58, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:37, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:10, 2.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:55, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:47, 1.73MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:31, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:08, 2.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:50, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:43, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:17, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:27, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:33, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:11, 2.48MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<00:52, 3.37MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:55, 996kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:27, 1.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:49, 1.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:18, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:42, 1.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:18, 1.23MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:43, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:41, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:27, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:05, 2.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<00:47, 3.40MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:23, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:03, 1.31MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:32, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:05, 2.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<14:24, 183kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<10:26, 252kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<07:19, 357kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<05:07, 504kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<04:35, 559kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:31, 727kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<02:31, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:15, 1.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:56, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:26, 1.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:00, 2.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<24:33, 98.9kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<17:52, 136kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<12:37, 192kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<08:52, 271kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<06:32, 361kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<04:54, 479kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<03:29, 668kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:21<02:25, 944kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<47:35, 48.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<34:03, 67.2kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<23:55, 95.4kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<16:30, 136kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<12:13, 182kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<08:48, 252kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<06:09, 356kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<04:20, 502kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<03:48, 565kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<02:58, 722kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<02:09, 990kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<01:32, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:40, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:28, 1.41MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:05, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:47, 2.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:14, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:06, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:49, 2.43MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:36, 3.20MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:16, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:27, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:12, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:54, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:38, 2.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:41, 696kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:09, 868kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:33, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:06, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:36, 1.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:23, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:01, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:01, 1.69MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:55, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:41, 2.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:29, 3.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<03:27, 482kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:37, 635kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:51, 886kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:36, 999kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:16, 1.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:02, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:44, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:59, 1.56MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:54, 1.68MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:41, 2.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:44, 1.98MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:43, 2.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:32, 2.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:38, 2.19MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:39, 2.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:34, 2.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:25, 3.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:50, 1.56MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:47, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:35, 2.21MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:37, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:37, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:28, 2.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:20, 3.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:42, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:39, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:29, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:33, 2.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:34, 1.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:26, 2.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:28, 2.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:29, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:21, 2.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:15, 3.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:46, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:53, 1.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:40, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:28, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:21, 2.54MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:51, 1.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:42, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:32, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:22, 2.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:32, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:28, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:20, 2.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:23, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:23, 2.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:17, 2.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<00:12, 3.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:29, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:26, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:13, 2.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:52, 719kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:42, 895kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:29, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:25, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:23, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:17, 1.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:16, 1.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:15, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:11, 2.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:07, 3.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<25:25, 16.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<17:40, 23.9kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<11:55, 34.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<07:48, 48.6kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<05:16, 67.7kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<03:41, 95.6kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<02:30, 136kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<01:36, 193kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<01:08, 252kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:49, 345kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:32, 485kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:21, 605kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:16, 769kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:11, 1.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:23<00:07, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:06, 1.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:05, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:01, 2.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:08, 563kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:06, 721kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 998kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:01, 503kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 651kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<109:46:10,  1.01it/s]  0%|          | 66/400000 [00:01<76:53:06,  1.44it/s]  0%|          | 276/400000 [00:01<53:48:26,  2.06it/s]  0%|          | 456/400000 [00:01<37:40:01,  2.95it/s]  0%|          | 708/400000 [00:01<26:21:52,  4.21it/s]  0%|          | 1190/400000 [00:01<18:26:23,  6.01it/s]  0%|          | 1840/400000 [00:01<12:53:30,  8.58it/s]  1%|          | 2549/400000 [00:01<9:00:46, 12.25it/s]   1%|          | 3019/400000 [00:01<6:19:04, 17.45it/s]  1%|          | 3409/400000 [00:02<4:25:50, 24.86it/s]  1%|          | 3990/400000 [00:02<3:06:09, 35.45it/s]  1%|          | 4410/400000 [00:02<2:10:56, 50.35it/s]  1%|          | 4781/400000 [00:02<1:32:11, 71.45it/s]  1%|▏         | 5137/400000 [00:02<1:05:20, 100.72it/s]  1%|▏         | 5608/400000 [00:02<46:06, 142.58it/s]    1%|▏         | 5972/400000 [00:02<32:56, 199.36it/s]  2%|▏         | 6542/400000 [00:03<23:36, 277.73it/s]  2%|▏         | 6880/400000 [00:03<17:18, 378.67it/s]  2%|▏         | 7200/400000 [00:03<12:52, 508.26it/s]  2%|▏         | 7494/400000 [00:03<10:09, 644.44it/s]  2%|▏         | 7944/400000 [00:03<07:34, 862.55it/s]  2%|▏         | 8650/400000 [00:03<05:34, 1170.86it/s]  2%|▏         | 9074/400000 [00:03<04:25, 1470.57it/s]  2%|▏         | 9481/400000 [00:03<04:01, 1620.29it/s]  2%|▏         | 9830/400000 [00:04<03:29, 1865.18it/s]  3%|▎         | 10520/400000 [00:04<02:43, 2387.66it/s]  3%|▎         | 11220/400000 [00:04<02:10, 2975.67it/s]  3%|▎         | 11941/400000 [00:04<01:47, 3611.72it/s]  3%|▎         | 12697/400000 [00:04<01:30, 4282.47it/s]  3%|▎         | 13402/400000 [00:04<01:19, 4853.47it/s]  4%|▎         | 14128/400000 [00:04<01:11, 5389.02it/s]  4%|▎         | 14841/400000 [00:04<01:06, 5811.97it/s]  4%|▍         | 15548/400000 [00:04<01:02, 6138.90it/s]  4%|▍         | 16241/400000 [00:05<01:01, 6216.99it/s]  4%|▍         | 16989/400000 [00:05<00:58, 6547.29it/s]  4%|▍         | 17687/400000 [00:05<01:04, 5889.13it/s]  5%|▍         | 18364/400000 [00:05<01:02, 6128.20it/s]  5%|▍         | 19010/400000 [00:05<01:03, 5997.72it/s]  5%|▍         | 19705/400000 [00:05<01:00, 6254.19it/s]  5%|▌         | 20350/400000 [00:05<01:01, 6221.88it/s]  5%|▌         | 20986/400000 [00:05<01:09, 5432.53it/s]  5%|▌         | 21557/400000 [00:05<01:18, 4837.22it/s]  6%|▌         | 22230/400000 [00:06<01:11, 5282.95it/s]  6%|▌         | 22878/400000 [00:06<01:07, 5591.15it/s]  6%|▌         | 23515/400000 [00:06<01:04, 5802.15it/s]  6%|▌         | 24217/400000 [00:06<01:01, 6119.45it/s]  6%|▌         | 24850/400000 [00:06<01:03, 5868.72it/s]  6%|▋         | 25552/400000 [00:06<01:00, 6171.86it/s]  7%|▋         | 26185/400000 [00:06<01:00, 6131.96it/s]  7%|▋         | 26923/400000 [00:06<00:57, 6458.90it/s]  7%|▋         | 27602/400000 [00:06<00:56, 6553.44it/s]  7%|▋         | 28267/400000 [00:07<00:58, 6327.49it/s]  7%|▋         | 28994/400000 [00:07<00:56, 6583.21it/s]  7%|▋         | 29699/400000 [00:07<00:55, 6715.16it/s]  8%|▊         | 30416/400000 [00:07<00:53, 6844.85it/s]  8%|▊         | 31161/400000 [00:07<00:52, 7015.09it/s]  8%|▊         | 31909/400000 [00:07<00:51, 7146.88it/s]  8%|▊         | 32628/400000 [00:07<00:51, 7143.48it/s]  8%|▊         | 33346/400000 [00:07<00:51, 7127.01it/s]  9%|▊         | 34073/400000 [00:07<00:51, 7168.87it/s]  9%|▊         | 34824/400000 [00:07<00:50, 7259.02it/s]  9%|▉         | 35552/400000 [00:08<01:04, 5670.13it/s]  9%|▉         | 36174/400000 [00:08<01:09, 5222.87it/s]  9%|▉         | 36852/400000 [00:08<01:04, 5609.31it/s]  9%|▉         | 37550/400000 [00:08<01:00, 5958.89it/s] 10%|▉         | 38251/400000 [00:08<00:57, 6238.11it/s] 10%|▉         | 38977/400000 [00:08<00:55, 6512.08it/s] 10%|▉         | 39651/400000 [00:08<00:56, 6359.20it/s] 10%|█         | 40355/400000 [00:08<00:54, 6547.04it/s] 10%|█         | 41088/400000 [00:08<00:53, 6761.97it/s] 10%|█         | 41775/400000 [00:09<01:01, 5780.02it/s] 11%|█         | 42386/400000 [00:09<01:09, 5112.95it/s] 11%|█         | 43103/400000 [00:09<01:03, 5594.27it/s] 11%|█         | 43846/400000 [00:09<00:58, 6041.25it/s] 11%|█         | 44488/400000 [00:09<00:58, 6072.76it/s] 11%|█▏        | 45219/400000 [00:09<00:55, 6395.03it/s] 11%|█▏        | 45916/400000 [00:09<00:53, 6557.30it/s] 12%|█▏        | 46589/400000 [00:09<00:55, 6379.70it/s] 12%|█▏        | 47251/400000 [00:09<00:54, 6448.97it/s] 12%|█▏        | 47906/400000 [00:10<00:54, 6435.04it/s] 12%|█▏        | 48651/400000 [00:10<00:52, 6708.15it/s] 12%|█▏        | 49330/400000 [00:10<00:55, 6307.17it/s] 13%|█▎        | 50004/400000 [00:10<00:54, 6429.95it/s] 13%|█▎        | 50710/400000 [00:10<00:52, 6605.56it/s] 13%|█▎        | 51378/400000 [00:10<00:53, 6531.50it/s] 13%|█▎        | 52090/400000 [00:10<00:51, 6695.87it/s] 13%|█▎        | 52764/400000 [00:10<00:52, 6640.49it/s] 13%|█▎        | 53483/400000 [00:10<00:50, 6795.82it/s] 14%|█▎        | 54199/400000 [00:11<00:50, 6899.39it/s] 14%|█▎        | 54911/400000 [00:11<00:49, 6961.57it/s] 14%|█▍        | 55610/400000 [00:11<00:50, 6881.17it/s] 14%|█▍        | 56331/400000 [00:11<00:49, 6974.43it/s] 14%|█▍        | 57030/400000 [00:11<00:49, 6921.29it/s] 14%|█▍        | 57731/400000 [00:11<00:49, 6946.27it/s] 15%|█▍        | 58427/400000 [00:11<01:02, 5462.39it/s] 15%|█▍        | 59024/400000 [00:11<01:09, 4940.33it/s] 15%|█▍        | 59741/400000 [00:11<01:02, 5446.72it/s] 15%|█▌        | 60396/400000 [00:12<00:59, 5735.84it/s] 15%|█▌        | 61100/400000 [00:12<00:57, 5937.82it/s] 15%|█▌        | 61722/400000 [00:12<00:56, 5999.75it/s] 16%|█▌        | 62342/400000 [00:12<00:55, 6030.97it/s] 16%|█▌        | 62959/400000 [00:12<00:55, 6036.20it/s] 16%|█▌        | 63707/400000 [00:12<00:52, 6405.36it/s] 16%|█▌        | 64417/400000 [00:12<00:50, 6598.15it/s] 16%|█▋        | 65117/400000 [00:12<00:49, 6713.62it/s] 16%|█▋        | 65833/400000 [00:12<00:48, 6840.12it/s] 17%|█▋        | 66536/400000 [00:12<00:48, 6893.42it/s] 17%|█▋        | 67230/400000 [00:13<00:51, 6516.58it/s] 17%|█▋        | 67889/400000 [00:13<00:51, 6409.93it/s] 17%|█▋        | 68584/400000 [00:13<00:50, 6562.83it/s] 17%|█▋        | 69282/400000 [00:13<00:49, 6682.57it/s] 17%|█▋        | 69955/400000 [00:13<00:49, 6688.05it/s] 18%|█▊        | 70647/400000 [00:13<00:48, 6754.41it/s] 18%|█▊        | 71359/400000 [00:13<00:47, 6859.36it/s] 18%|█▊        | 72074/400000 [00:13<00:47, 6943.39it/s] 18%|█▊        | 72770/400000 [00:13<00:48, 6803.15it/s] 18%|█▊        | 73453/400000 [00:14<00:48, 6767.23it/s] 19%|█▊        | 74191/400000 [00:14<00:46, 6939.12it/s] 19%|█▊        | 74898/400000 [00:14<00:46, 6974.32it/s] 19%|█▉        | 75597/400000 [00:14<00:57, 5661.46it/s] 19%|█▉        | 76205/400000 [00:14<00:56, 5727.79it/s] 19%|█▉        | 76807/400000 [00:14<01:09, 4657.09it/s] 19%|█▉        | 77364/400000 [00:14<01:05, 4896.90it/s] 19%|█▉        | 77980/400000 [00:14<01:01, 5217.62it/s] 20%|█▉        | 78536/400000 [00:14<01:02, 5123.36it/s] 20%|█▉        | 79237/400000 [00:15<00:57, 5571.92it/s] 20%|█▉        | 79823/400000 [00:15<01:01, 5186.43it/s] 20%|██        | 80526/400000 [00:15<00:56, 5627.43it/s] 20%|██        | 81182/400000 [00:15<00:54, 5877.27it/s] 20%|██        | 81894/400000 [00:15<00:51, 6200.13it/s] 21%|██        | 82647/400000 [00:15<00:48, 6546.27it/s] 21%|██        | 83376/400000 [00:15<00:46, 6750.96it/s] 21%|██        | 84067/400000 [00:15<00:48, 6569.42it/s] 21%|██        | 84762/400000 [00:15<00:47, 6678.30it/s] 21%|██▏       | 85439/400000 [00:16<00:50, 6252.12it/s] 22%|██▏       | 86140/400000 [00:16<00:48, 6461.04it/s] 22%|██▏       | 86854/400000 [00:16<00:47, 6650.47it/s] 22%|██▏       | 87587/400000 [00:16<00:45, 6839.93it/s] 22%|██▏       | 88307/400000 [00:16<00:44, 6944.08it/s] 22%|██▏       | 89040/400000 [00:16<00:44, 7054.89it/s] 22%|██▏       | 89790/400000 [00:16<00:43, 7181.70it/s] 23%|██▎       | 90512/400000 [00:16<00:43, 7090.61it/s] 23%|██▎       | 91224/400000 [00:16<00:46, 6642.57it/s] 23%|██▎       | 91896/400000 [00:17<00:47, 6431.03it/s] 23%|██▎       | 92546/400000 [00:17<00:48, 6380.11it/s] 23%|██▎       | 93261/400000 [00:17<00:46, 6591.41it/s] 23%|██▎       | 93997/400000 [00:17<00:44, 6803.63it/s] 24%|██▎       | 94741/400000 [00:17<00:43, 6980.91it/s] 24%|██▍       | 95444/400000 [00:17<00:44, 6909.22it/s] 24%|██▍       | 96139/400000 [00:17<00:44, 6803.65it/s] 24%|██▍       | 96886/400000 [00:17<00:43, 6990.72it/s] 24%|██▍       | 97623/400000 [00:17<00:42, 7100.03it/s] 25%|██▍       | 98365/400000 [00:17<00:41, 7190.80it/s] 25%|██▍       | 99122/400000 [00:18<00:41, 7299.50it/s] 25%|██▍       | 99854/400000 [00:18<00:45, 6549.94it/s] 25%|██▌       | 100525/400000 [00:18<00:49, 6100.53it/s] 25%|██▌       | 101153/400000 [00:18<00:48, 6122.67it/s] 25%|██▌       | 101876/400000 [00:18<00:46, 6415.60it/s] 26%|██▌       | 102542/400000 [00:18<00:45, 6486.51it/s] 26%|██▌       | 103274/400000 [00:18<00:44, 6715.33it/s] 26%|██▌       | 104034/400000 [00:18<00:42, 6957.86it/s] 26%|██▌       | 104738/400000 [00:18<00:42, 6948.62it/s] 26%|██▋       | 105475/400000 [00:18<00:41, 7067.43it/s] 27%|██▋       | 106187/400000 [00:19<00:51, 5675.03it/s] 27%|██▋       | 106932/400000 [00:19<00:47, 6109.44it/s] 27%|██▋       | 107628/400000 [00:19<00:46, 6341.14it/s] 27%|██▋       | 108294/400000 [00:19<00:55, 5236.46it/s] 27%|██▋       | 108902/400000 [00:19<00:53, 5462.50it/s] 27%|██▋       | 109586/400000 [00:19<00:49, 5812.37it/s] 28%|██▊       | 110319/400000 [00:19<00:46, 6196.05it/s] 28%|██▊       | 111030/400000 [00:19<00:44, 6444.42it/s] 28%|██▊       | 111763/400000 [00:20<00:43, 6686.09it/s] 28%|██▊       | 112451/400000 [00:20<00:42, 6727.62it/s] 28%|██▊       | 113138/400000 [00:20<00:42, 6768.04it/s] 28%|██▊       | 113843/400000 [00:20<00:41, 6848.87it/s] 29%|██▊       | 114548/400000 [00:20<00:41, 6906.75it/s] 29%|██▉       | 115283/400000 [00:20<00:40, 7033.27it/s] 29%|██▉       | 115991/400000 [00:20<00:41, 6880.12it/s] 29%|██▉       | 116699/400000 [00:20<00:40, 6938.48it/s] 29%|██▉       | 117432/400000 [00:20<00:40, 7048.74it/s] 30%|██▉       | 118154/400000 [00:20<00:39, 7097.66it/s] 30%|██▉       | 118877/400000 [00:21<00:39, 7133.27it/s] 30%|██▉       | 119592/400000 [00:21<00:40, 6889.75it/s] 30%|███       | 120320/400000 [00:21<00:39, 7000.35it/s] 30%|███       | 121023/400000 [00:21<00:40, 6950.22it/s] 30%|███       | 121757/400000 [00:21<00:39, 7061.98it/s] 31%|███       | 122465/400000 [00:21<00:39, 6957.77it/s] 31%|███       | 123243/400000 [00:21<00:38, 7184.22it/s] 31%|███       | 123965/400000 [00:21<00:38, 7123.61it/s] 31%|███       | 124703/400000 [00:21<00:38, 7196.19it/s] 31%|███▏      | 125430/400000 [00:21<00:38, 7217.86it/s] 32%|███▏      | 126153/400000 [00:22<00:38, 7111.83it/s] 32%|███▏      | 126866/400000 [00:22<00:38, 7025.37it/s] 32%|███▏      | 127570/400000 [00:22<00:40, 6741.91it/s] 32%|███▏      | 128248/400000 [00:22<00:45, 5963.64it/s] 32%|███▏      | 128864/400000 [00:22<00:47, 5679.75it/s] 32%|███▏      | 129448/400000 [00:22<00:54, 4977.93it/s] 32%|███▏      | 129973/400000 [00:22<01:10, 3851.81it/s] 33%|███▎      | 130545/400000 [00:23<01:04, 4192.08it/s] 33%|███▎      | 131014/400000 [00:23<01:16, 3496.22it/s] 33%|███▎      | 131473/400000 [00:23<01:11, 3764.73it/s] 33%|███▎      | 132007/400000 [00:23<01:04, 4129.72it/s] 33%|███▎      | 132615/400000 [00:23<00:58, 4568.81it/s] 33%|███▎      | 133316/400000 [00:23<00:52, 5100.61it/s] 33%|███▎      | 133877/400000 [00:23<00:50, 5228.75it/s] 34%|███▎      | 134634/400000 [00:23<00:46, 5762.20it/s] 34%|███▍      | 135342/400000 [00:23<00:43, 6101.33it/s] 34%|███▍      | 136069/400000 [00:24<00:41, 6409.84it/s] 34%|███▍      | 136813/400000 [00:24<00:39, 6686.25it/s] 34%|███▍      | 137525/400000 [00:24<00:38, 6809.41it/s] 35%|███▍      | 138259/400000 [00:24<00:37, 6955.54it/s] 35%|███▍      | 138978/400000 [00:24<00:37, 7022.84it/s] 35%|███▍      | 139697/400000 [00:24<00:36, 7070.71it/s] 35%|███▌      | 140411/400000 [00:24<00:37, 7007.59it/s] 35%|███▌      | 141117/400000 [00:24<00:38, 6774.48it/s] 35%|███▌      | 141800/400000 [00:24<00:38, 6629.26it/s] 36%|███▌      | 142468/400000 [00:24<00:39, 6539.40it/s] 36%|███▌      | 143221/400000 [00:25<00:37, 6807.29it/s] 36%|███▌      | 143907/400000 [00:25<00:40, 6315.85it/s] 36%|███▌      | 144550/400000 [00:25<00:46, 5493.06it/s] 36%|███▋      | 145126/400000 [00:25<00:50, 5040.96it/s] 36%|███▋      | 145832/400000 [00:25<00:47, 5366.94it/s] 37%|███▋      | 146549/400000 [00:25<00:43, 5804.65it/s] 37%|███▋      | 147305/400000 [00:25<00:40, 6237.89it/s] 37%|███▋      | 147957/400000 [00:25<00:44, 5679.43it/s] 37%|███▋      | 148555/400000 [00:26<00:51, 4863.20it/s] 37%|███▋      | 149103/400000 [00:26<00:53, 4649.31it/s] 37%|███▋      | 149599/400000 [00:26<01:08, 3676.35it/s] 38%|███▊      | 150021/400000 [00:26<01:06, 3786.27it/s] 38%|███▊      | 150438/400000 [00:26<01:18, 3176.31it/s] 38%|███▊      | 150800/400000 [00:26<01:17, 3229.46it/s] 38%|███▊      | 151154/400000 [00:26<01:20, 3077.83it/s] 38%|███▊      | 151485/400000 [00:27<01:20, 3074.44it/s] 38%|███▊      | 152066/400000 [00:27<01:09, 3579.98it/s] 38%|███▊      | 152467/400000 [00:27<01:32, 2663.06it/s] 38%|███▊      | 153051/400000 [00:27<01:17, 3181.74it/s] 38%|███▊      | 153458/400000 [00:27<01:12, 3389.69it/s] 38%|███▊      | 153913/400000 [00:27<01:07, 3670.49it/s] 39%|███▊      | 154451/400000 [00:27<01:00, 4056.42it/s] 39%|███▉      | 155178/400000 [00:27<00:52, 4675.81it/s] 39%|███▉      | 155919/400000 [00:27<00:46, 5257.23it/s] 39%|███▉      | 156612/400000 [00:28<00:42, 5667.45it/s] 39%|███▉      | 157336/400000 [00:28<00:40, 6061.26it/s] 39%|███▉      | 157998/400000 [00:28<00:39, 6110.95it/s] 40%|███▉      | 158699/400000 [00:28<00:38, 6349.86it/s] 40%|███▉      | 159415/400000 [00:28<00:36, 6571.65it/s] 40%|████      | 160109/400000 [00:28<00:35, 6676.89it/s] 40%|████      | 160823/400000 [00:28<00:35, 6806.44it/s] 40%|████      | 161515/400000 [00:28<00:37, 6353.96it/s] 41%|████      | 162165/400000 [00:28<00:37, 6270.44it/s] 41%|████      | 162822/400000 [00:29<00:37, 6355.44it/s] 41%|████      | 163465/400000 [00:29<00:37, 6319.51it/s] 41%|████      | 164172/400000 [00:29<00:36, 6525.77it/s] 41%|████      | 164877/400000 [00:29<00:35, 6674.16it/s] 41%|████▏     | 165613/400000 [00:29<00:34, 6864.41it/s] 42%|████▏     | 166330/400000 [00:29<00:33, 6951.43it/s] 42%|████▏     | 167037/400000 [00:29<00:33, 6984.86it/s] 42%|████▏     | 167759/400000 [00:29<00:32, 7053.40it/s] 42%|████▏     | 168475/400000 [00:29<00:32, 7083.38it/s] 42%|████▏     | 169203/400000 [00:29<00:32, 7140.35it/s] 42%|████▏     | 169919/400000 [00:30<00:32, 7118.43it/s] 43%|████▎     | 170681/400000 [00:30<00:31, 7260.80it/s] 43%|████▎     | 171409/400000 [00:30<00:32, 7087.69it/s] 43%|████▎     | 172120/400000 [00:30<00:36, 6323.75it/s] 43%|████▎     | 172769/400000 [00:30<00:40, 5659.61it/s] 43%|████▎     | 173360/400000 [00:30<00:40, 5581.85it/s] 43%|████▎     | 173936/400000 [00:30<00:43, 5220.40it/s] 44%|████▎     | 174476/400000 [00:30<00:43, 5151.92it/s] 44%|████▍     | 175199/400000 [00:30<00:39, 5636.06it/s] 44%|████▍     | 175804/400000 [00:31<00:38, 5754.16it/s] 44%|████▍     | 176396/400000 [00:31<00:42, 5283.99it/s] 44%|████▍     | 176944/400000 [00:31<00:57, 3881.25it/s] 44%|████▍     | 177619/400000 [00:31<00:49, 4448.29it/s] 45%|████▍     | 178144/400000 [00:31<00:48, 4534.60it/s] 45%|████▍     | 178652/400000 [00:31<00:49, 4515.05it/s] 45%|████▍     | 179351/400000 [00:31<00:43, 5050.58it/s] 45%|████▌     | 180052/400000 [00:31<00:39, 5511.74it/s] 45%|████▌     | 180758/400000 [00:32<00:37, 5899.01it/s] 45%|████▌     | 181464/400000 [00:32<00:35, 6156.87it/s] 46%|████▌     | 182111/400000 [00:32<00:37, 5821.05it/s] 46%|████▌     | 182766/400000 [00:32<00:36, 6020.43it/s] 46%|████▌     | 183388/400000 [00:32<00:39, 5475.08it/s] 46%|████▌     | 183959/400000 [00:32<00:48, 4487.66it/s] 46%|████▌     | 184454/400000 [00:32<00:48, 4425.18it/s] 46%|████▌     | 184998/400000 [00:32<00:45, 4687.30it/s] 46%|████▋     | 185494/400000 [00:33<00:45, 4669.07it/s] 47%|████▋     | 186070/400000 [00:33<00:45, 4706.86it/s] 47%|████▋     | 186554/400000 [00:33<00:51, 4146.35it/s] 47%|████▋     | 187218/400000 [00:33<00:45, 4672.25it/s] 47%|████▋     | 187767/400000 [00:33<00:43, 4889.97it/s] 47%|████▋     | 188445/400000 [00:33<00:39, 5334.89it/s] 47%|████▋     | 189011/400000 [00:33<00:44, 4785.13it/s] 47%|████▋     | 189523/400000 [00:33<00:44, 4697.47it/s] 48%|████▊     | 190043/400000 [00:33<00:43, 4835.67it/s] 48%|████▊     | 190545/400000 [00:34<00:56, 3678.68it/s] 48%|████▊     | 191234/400000 [00:34<00:48, 4275.96it/s] 48%|████▊     | 191736/400000 [00:34<01:04, 3232.36it/s] 48%|████▊     | 192149/400000 [00:34<01:09, 2996.88it/s] 48%|████▊     | 192623/400000 [00:34<01:01, 3368.41it/s] 48%|████▊     | 193340/400000 [00:34<00:51, 4005.09it/s] 49%|████▊     | 194013/400000 [00:34<00:45, 4558.81it/s] 49%|████▊     | 194717/400000 [00:35<00:40, 5097.45it/s] 49%|████▉     | 195417/400000 [00:35<00:36, 5549.90it/s] 49%|████▉     | 196161/400000 [00:35<00:33, 6006.36it/s] 49%|████▉     | 196853/400000 [00:35<00:32, 6251.24it/s] 49%|████▉     | 197536/400000 [00:35<00:31, 6413.10it/s] 50%|████▉     | 198264/400000 [00:35<00:30, 6650.03it/s] 50%|████▉     | 199005/400000 [00:35<00:29, 6860.25it/s] 50%|████▉     | 199748/400000 [00:35<00:29, 6819.01it/s] 50%|█████     | 200444/400000 [00:35<00:37, 5364.07it/s] 50%|█████     | 201039/400000 [00:36<00:44, 4421.70it/s] 50%|█████     | 201550/400000 [00:36<00:47, 4195.98it/s] 51%|█████     | 202260/400000 [00:36<00:41, 4781.68it/s] 51%|█████     | 203005/400000 [00:36<00:39, 4988.24it/s] 51%|█████     | 203551/400000 [00:36<00:47, 4150.13it/s] 51%|█████     | 204023/400000 [00:37<01:07, 2885.45it/s] 51%|█████     | 204404/400000 [00:37<01:15, 2593.63it/s] 51%|█████▏    | 205073/400000 [00:37<01:01, 3176.99it/s] 51%|█████▏    | 205760/400000 [00:37<00:51, 3787.48it/s] 52%|█████▏    | 206469/400000 [00:37<00:43, 4402.06it/s] 52%|█████▏    | 207212/400000 [00:37<00:38, 5014.22it/s] 52%|█████▏    | 207856/400000 [00:37<00:35, 5369.00it/s] 52%|█████▏    | 208571/400000 [00:37<00:32, 5801.64it/s] 52%|█████▏    | 209225/400000 [00:37<00:32, 5912.86it/s] 52%|█████▏    | 209938/400000 [00:37<00:30, 6231.29it/s] 53%|█████▎    | 210648/400000 [00:38<00:29, 6467.86it/s] 53%|█████▎    | 211348/400000 [00:38<00:28, 6618.22it/s] 53%|█████▎    | 212077/400000 [00:38<00:27, 6804.81it/s] 53%|█████▎    | 212778/400000 [00:38<00:27, 6864.38it/s] 53%|█████▎    | 213500/400000 [00:38<00:26, 6967.07it/s] 54%|█████▎    | 214253/400000 [00:38<00:26, 7126.23it/s] 54%|█████▎    | 214973/400000 [00:38<00:26, 7092.33it/s] 54%|█████▍    | 215688/400000 [00:38<00:27, 6800.67it/s] 54%|█████▍    | 216422/400000 [00:38<00:26, 6953.49it/s] 54%|█████▍    | 217159/400000 [00:39<00:25, 7072.34it/s] 54%|█████▍    | 217871/400000 [00:39<00:32, 5524.85it/s] 55%|█████▍    | 218479/400000 [00:39<00:35, 5053.12it/s] 55%|█████▍    | 219189/400000 [00:39<00:32, 5527.26it/s] 55%|█████▍    | 219894/400000 [00:39<00:30, 5910.06it/s] 55%|█████▌    | 220548/400000 [00:39<00:29, 6082.38it/s] 55%|█████▌    | 221217/400000 [00:39<00:28, 6251.01it/s] 55%|█████▌    | 221864/400000 [00:39<00:31, 5737.74it/s] 56%|█████▌    | 222463/400000 [00:39<00:30, 5810.11it/s] 56%|█████▌    | 223061/400000 [00:40<00:35, 4958.90it/s] 56%|█████▌    | 223601/400000 [00:40<00:34, 5083.38it/s] 56%|█████▌    | 224287/400000 [00:40<00:31, 5511.28it/s] 56%|█████▌    | 224953/400000 [00:40<00:30, 5811.84it/s] 56%|█████▋    | 225688/400000 [00:40<00:28, 6199.09it/s] 57%|█████▋    | 226343/400000 [00:40<00:27, 6220.32it/s] 57%|█████▋    | 227061/400000 [00:40<00:26, 6477.35it/s] 57%|█████▋    | 227806/400000 [00:40<00:25, 6741.14it/s] 57%|█████▋    | 228554/400000 [00:40<00:24, 6946.39it/s] 57%|█████▋    | 229278/400000 [00:41<00:24, 7030.90it/s] 57%|█████▊    | 230000/400000 [00:41<00:23, 7084.36it/s] 58%|█████▊    | 230728/400000 [00:41<00:23, 7139.76it/s] 58%|█████▊    | 231446/400000 [00:41<00:23, 7124.30it/s] 58%|█████▊    | 232162/400000 [00:41<00:24, 6905.87it/s] 58%|█████▊    | 232856/400000 [00:41<00:25, 6565.71it/s] 58%|█████▊    | 233519/400000 [00:41<00:26, 6382.44it/s] 59%|█████▊    | 234254/400000 [00:41<00:24, 6643.13it/s] 59%|█████▊    | 234925/400000 [00:41<00:25, 6391.25it/s] 59%|█████▉    | 235571/400000 [00:42<00:25, 6350.42it/s] 59%|█████▉    | 236300/400000 [00:42<00:24, 6604.76it/s] 59%|█████▉    | 237025/400000 [00:42<00:24, 6785.78it/s] 59%|█████▉    | 237709/400000 [00:42<00:23, 6792.61it/s] 60%|█████▉    | 238393/400000 [00:42<00:29, 5489.48it/s] 60%|█████▉    | 238985/400000 [00:42<00:35, 4598.73it/s] 60%|█████▉    | 239583/400000 [00:42<00:32, 4939.85it/s] 60%|██████    | 240310/400000 [00:42<00:29, 5464.54it/s] 60%|██████    | 241010/400000 [00:42<00:27, 5848.76it/s] 60%|██████    | 241695/400000 [00:43<00:25, 6115.00it/s] 61%|██████    | 242388/400000 [00:43<00:24, 6337.03it/s] 61%|██████    | 243118/400000 [00:43<00:23, 6597.88it/s] 61%|██████    | 243831/400000 [00:43<00:23, 6745.57it/s] 61%|██████    | 244521/400000 [00:43<00:24, 6451.03it/s] 61%|██████▏   | 245180/400000 [00:43<00:32, 4831.26it/s] 61%|██████▏   | 245777/400000 [00:43<00:30, 5122.27it/s] 62%|██████▏   | 246343/400000 [00:43<00:32, 4721.92it/s] 62%|██████▏   | 246859/400000 [00:44<00:32, 4685.11it/s] 62%|██████▏   | 247605/400000 [00:44<00:28, 5270.83it/s] 62%|██████▏   | 248294/400000 [00:44<00:26, 5669.89it/s] 62%|██████▏   | 248957/400000 [00:44<00:25, 5927.05it/s] 62%|██████▏   | 249592/400000 [00:44<00:24, 6047.76it/s] 63%|██████▎   | 250220/400000 [00:44<00:25, 5965.64it/s] 63%|██████▎   | 250844/400000 [00:44<00:24, 6044.28it/s] 63%|██████▎   | 251533/400000 [00:44<00:23, 6274.38it/s] 63%|██████▎   | 252251/400000 [00:44<00:22, 6519.81it/s] 63%|██████▎   | 252973/400000 [00:44<00:21, 6714.42it/s] 63%|██████▎   | 253693/400000 [00:45<00:21, 6852.41it/s] 64%|██████▎   | 254403/400000 [00:45<00:21, 6923.31it/s] 64%|██████▍   | 255130/400000 [00:45<00:20, 7019.28it/s] 64%|██████▍   | 255845/400000 [00:45<00:20, 7055.68it/s] 64%|██████▍   | 256554/400000 [00:45<00:21, 6678.86it/s] 64%|██████▍   | 257279/400000 [00:45<00:20, 6839.36it/s] 64%|██████▍   | 257969/400000 [00:45<00:20, 6852.61it/s] 65%|██████▍   | 258658/400000 [00:45<00:21, 6661.66it/s] 65%|██████▍   | 259336/400000 [00:45<00:21, 6695.26it/s] 65%|██████▌   | 260026/400000 [00:45<00:20, 6753.56it/s] 65%|██████▌   | 260723/400000 [00:46<00:20, 6816.19it/s] 65%|██████▌   | 261407/400000 [00:46<00:20, 6736.80it/s] 66%|██████▌   | 262086/400000 [00:46<00:20, 6750.01it/s] 66%|██████▌   | 262775/400000 [00:46<00:20, 6790.17it/s] 66%|██████▌   | 263455/400000 [00:46<00:20, 6774.63it/s] 66%|██████▌   | 264133/400000 [00:46<00:26, 5102.63it/s] 66%|██████▌   | 264750/400000 [00:46<00:25, 5381.14it/s] 66%|██████▋   | 265336/400000 [00:46<00:25, 5229.02it/s] 66%|██████▋   | 265893/400000 [00:47<00:26, 5000.81it/s] 67%|██████▋   | 266525/400000 [00:47<00:25, 5334.14it/s] 67%|██████▋   | 267270/400000 [00:47<00:22, 5830.52it/s] 67%|██████▋   | 267953/400000 [00:47<00:21, 6097.68it/s] 67%|██████▋   | 268588/400000 [00:47<00:21, 6084.27it/s] 67%|██████▋   | 269269/400000 [00:47<00:20, 6284.52it/s] 68%|██████▊   | 270000/400000 [00:47<00:19, 6559.87it/s] 68%|██████▊   | 270733/400000 [00:47<00:19, 6771.76it/s] 68%|██████▊   | 271458/400000 [00:47<00:18, 6907.30it/s] 68%|██████▊   | 272157/400000 [00:47<00:19, 6481.30it/s] 68%|██████▊   | 272817/400000 [00:48<00:22, 5701.69it/s] 68%|██████▊   | 273457/400000 [00:48<00:21, 5892.78it/s] 69%|██████▊   | 274178/400000 [00:48<00:20, 6233.10it/s] 69%|██████▊   | 274844/400000 [00:48<00:19, 6354.34it/s] 69%|██████▉   | 275494/400000 [00:48<00:20, 6070.22it/s] 69%|██████▉   | 276172/400000 [00:48<00:19, 6266.23it/s] 69%|██████▉   | 276834/400000 [00:48<00:19, 6363.43it/s] 69%|██████▉   | 277525/400000 [00:48<00:18, 6516.54it/s] 70%|██████▉   | 278258/400000 [00:48<00:18, 6738.12it/s] 70%|██████▉   | 278997/400000 [00:49<00:17, 6919.74it/s] 70%|██████▉   | 279758/400000 [00:49<00:16, 7110.94it/s] 70%|███████   | 280492/400000 [00:49<00:16, 7177.37it/s] 70%|███████   | 281229/400000 [00:49<00:16, 7231.25it/s] 70%|███████   | 281955/400000 [00:49<00:16, 6974.76it/s] 71%|███████   | 282657/400000 [00:49<00:17, 6762.62it/s] 71%|███████   | 283415/400000 [00:49<00:16, 6986.65it/s] 71%|███████   | 284119/400000 [00:49<00:16, 6912.96it/s] 71%|███████   | 284814/400000 [00:49<00:22, 5197.75it/s] 71%|███████▏  | 285400/400000 [00:50<00:23, 4810.50it/s] 71%|███████▏  | 285933/400000 [00:50<00:23, 4794.78it/s] 72%|███████▏  | 286651/400000 [00:50<00:21, 5325.09it/s] 72%|███████▏  | 287266/400000 [00:50<00:20, 5546.66it/s] 72%|███████▏  | 287972/400000 [00:50<00:18, 5926.15it/s] 72%|███████▏  | 288703/400000 [00:50<00:17, 6282.67it/s] 72%|███████▏  | 289427/400000 [00:50<00:16, 6540.78it/s] 73%|███████▎  | 290150/400000 [00:50<00:16, 6731.91it/s] 73%|███████▎  | 290887/400000 [00:50<00:15, 6909.88it/s] 73%|███████▎  | 291628/400000 [00:51<00:15, 7049.00it/s] 73%|███████▎  | 292343/400000 [00:51<00:15, 7045.72it/s] 73%|███████▎  | 293055/400000 [00:51<00:15, 7001.51it/s] 73%|███████▎  | 293760/400000 [00:51<00:15, 7007.02it/s] 74%|███████▎  | 294465/400000 [00:51<00:17, 5871.21it/s] 74%|███████▍  | 295086/400000 [00:51<00:19, 5393.71it/s] 74%|███████▍  | 295657/400000 [00:51<00:22, 4739.27it/s] 74%|███████▍  | 296348/400000 [00:51<00:19, 5229.24it/s] 74%|███████▍  | 297032/400000 [00:52<00:18, 5626.15it/s] 74%|███████▍  | 297632/400000 [00:52<00:18, 5676.90it/s] 75%|███████▍  | 298354/400000 [00:52<00:16, 6064.91it/s] 75%|███████▍  | 299061/400000 [00:52<00:15, 6334.76it/s] 75%|███████▍  | 299778/400000 [00:52<00:15, 6563.45it/s] 75%|███████▌  | 300505/400000 [00:52<00:14, 6758.72it/s] 75%|███████▌  | 301212/400000 [00:52<00:14, 6847.62it/s] 75%|███████▌  | 301949/400000 [00:52<00:14, 6996.19it/s] 76%|███████▌  | 302705/400000 [00:52<00:13, 7155.36it/s] 76%|███████▌  | 303452/400000 [00:52<00:13, 7245.52it/s] 76%|███████▌  | 304197/400000 [00:53<00:13, 7303.44it/s] 76%|███████▌  | 304931/400000 [00:53<00:13, 7202.80it/s] 76%|███████▋  | 305662/400000 [00:53<00:13, 7234.61it/s] 77%|███████▋  | 306388/400000 [00:53<00:13, 7164.10it/s] 77%|███████▋  | 307112/400000 [00:53<00:12, 7185.96it/s] 77%|███████▋  | 307832/400000 [00:53<00:13, 7043.47it/s] 77%|███████▋  | 308544/400000 [00:53<00:12, 7065.06it/s] 77%|███████▋  | 309271/400000 [00:53<00:12, 7124.06it/s] 77%|███████▋  | 309985/400000 [00:53<00:13, 6848.89it/s] 78%|███████▊  | 310712/400000 [00:53<00:12, 6969.85it/s] 78%|███████▊  | 311412/400000 [00:54<00:16, 5514.61it/s] 78%|███████▊  | 312013/400000 [00:54<00:15, 5596.93it/s] 78%|███████▊  | 312608/400000 [00:54<00:18, 4794.57it/s] 78%|███████▊  | 313299/400000 [00:54<00:16, 5278.63it/s] 79%|███████▊  | 314047/400000 [00:54<00:14, 5789.08it/s] 79%|███████▊  | 314737/400000 [00:54<00:14, 6081.28it/s] 79%|███████▉  | 315383/400000 [00:54<00:13, 6111.87it/s] 79%|███████▉  | 316021/400000 [00:54<00:13, 6138.73it/s] 79%|███████▉  | 316717/400000 [00:55<00:13, 6362.37it/s] 79%|███████▉  | 317413/400000 [00:55<00:12, 6529.82it/s] 80%|███████▉  | 318144/400000 [00:55<00:12, 6745.74it/s] 80%|███████▉  | 318845/400000 [00:55<00:11, 6820.80it/s] 80%|███████▉  | 319535/400000 [00:55<00:11, 6832.50it/s] 80%|████████  | 320224/400000 [00:55<00:14, 5395.28it/s] 80%|████████  | 320815/400000 [00:55<00:14, 5492.69it/s] 80%|████████  | 321484/400000 [00:55<00:13, 5802.21it/s] 81%|████████  | 322095/400000 [00:55<00:13, 5693.03it/s] 81%|████████  | 322747/400000 [00:56<00:13, 5918.19it/s] 81%|████████  | 323444/400000 [00:56<00:12, 6197.89it/s] 81%|████████  | 324080/400000 [00:56<00:12, 6191.21it/s] 81%|████████  | 324811/400000 [00:56<00:11, 6488.06it/s] 81%|████████▏ | 325472/400000 [00:56<00:11, 6402.11it/s] 82%|████████▏ | 326136/400000 [00:56<00:11, 6470.06it/s] 82%|████████▏ | 326789/400000 [00:56<00:11, 6487.37it/s] 82%|████████▏ | 327442/400000 [00:56<00:13, 5391.43it/s] 82%|████████▏ | 328016/400000 [00:56<00:13, 5394.39it/s] 82%|████████▏ | 328580/400000 [00:57<00:14, 5008.21it/s] 82%|████████▏ | 329129/400000 [00:57<00:13, 5142.03it/s] 82%|████████▏ | 329660/400000 [00:57<00:13, 5058.34it/s] 83%|████████▎ | 330242/400000 [00:57<00:13, 5263.95it/s] 83%|████████▎ | 330829/400000 [00:57<00:12, 5430.91it/s] 83%|████████▎ | 331381/400000 [00:57<00:12, 5331.51it/s] 83%|████████▎ | 331921/400000 [00:57<00:16, 4230.75it/s] 83%|████████▎ | 332385/400000 [00:57<00:18, 3675.29it/s] 83%|████████▎ | 332838/400000 [00:58<00:17, 3894.87it/s] 83%|████████▎ | 333260/400000 [00:58<00:17, 3783.29it/s] 83%|████████▎ | 333888/400000 [00:58<00:15, 4294.86it/s] 84%|████████▎ | 334590/400000 [00:58<00:13, 4860.96it/s] 84%|████████▍ | 335192/400000 [00:58<00:12, 5156.71it/s] 84%|████████▍ | 335866/400000 [00:58<00:11, 5546.47it/s] 84%|████████▍ | 336462/400000 [00:58<00:11, 5663.64it/s] 84%|████████▍ | 337152/400000 [00:58<00:10, 5984.86it/s] 84%|████████▍ | 337844/400000 [00:58<00:09, 6236.41it/s] 85%|████████▍ | 338488/400000 [00:58<00:10, 6092.33it/s] 85%|████████▍ | 339204/400000 [00:59<00:09, 6371.00it/s] 85%|████████▍ | 339855/400000 [00:59<00:09, 6344.00it/s] 85%|████████▌ | 340535/400000 [00:59<00:09, 6472.90it/s] 85%|████████▌ | 341190/400000 [00:59<00:09, 6465.79it/s] 85%|████████▌ | 341947/400000 [00:59<00:08, 6761.46it/s] 86%|████████▌ | 342655/400000 [00:59<00:08, 6852.56it/s] 86%|████████▌ | 343346/400000 [00:59<00:08, 6635.88it/s] 86%|████████▌ | 344015/400000 [00:59<00:09, 6183.02it/s] 86%|████████▌ | 344644/400000 [00:59<00:09, 5959.86it/s] 86%|████████▋ | 345262/400000 [00:59<00:09, 6023.47it/s] 86%|████████▋ | 345871/400000 [01:00<00:09, 5832.81it/s] 87%|████████▋ | 346511/400000 [01:00<00:09, 5910.96it/s] 87%|████████▋ | 347115/400000 [01:00<00:08, 5946.67it/s] 87%|████████▋ | 347713/400000 [01:00<00:09, 5622.73it/s] 87%|████████▋ | 348282/400000 [01:00<00:09, 5401.91it/s] 87%|████████▋ | 348829/400000 [01:00<00:11, 4502.56it/s] 87%|████████▋ | 349548/400000 [01:00<00:09, 5070.70it/s] 88%|████████▊ | 350100/400000 [01:00<00:09, 5059.28it/s] 88%|████████▊ | 350703/400000 [01:01<00:09, 5316.01it/s] 88%|████████▊ | 351260/400000 [01:01<00:10, 4651.69it/s] 88%|████████▊ | 351889/400000 [01:01<00:09, 4960.37it/s] 88%|████████▊ | 352414/400000 [01:01<00:09, 4994.05it/s] 88%|████████▊ | 352934/400000 [01:01<00:09, 4865.00it/s] 88%|████████▊ | 353452/400000 [01:01<00:09, 4955.39it/s] 88%|████████▊ | 353964/400000 [01:01<00:09, 5003.01it/s] 89%|████████▊ | 354576/400000 [01:01<00:08, 5292.20it/s] 89%|████████▉ | 355165/400000 [01:01<00:08, 5457.12it/s] 89%|████████▉ | 355785/400000 [01:02<00:07, 5659.37it/s] 89%|████████▉ | 356470/400000 [01:02<00:07, 5968.31it/s] 89%|████████▉ | 357184/400000 [01:02<00:06, 6275.96it/s] 89%|████████▉ | 357911/400000 [01:02<00:06, 6543.12it/s] 90%|████████▉ | 358615/400000 [01:02<00:06, 6683.08it/s] 90%|████████▉ | 359361/400000 [01:02<00:05, 6895.81it/s] 90%|█████████ | 360068/400000 [01:02<00:05, 6946.86it/s] 90%|█████████ | 360809/400000 [01:02<00:05, 7078.29it/s] 90%|█████████ | 361564/400000 [01:02<00:05, 7210.85it/s] 91%|█████████ | 362304/400000 [01:02<00:05, 7264.06it/s] 91%|█████████ | 363034/400000 [01:03<00:05, 7062.30it/s] 91%|█████████ | 363770/400000 [01:03<00:05, 7146.64it/s] 91%|█████████ | 364516/400000 [01:03<00:04, 7236.52it/s] 91%|█████████▏| 365279/400000 [01:03<00:04, 7347.54it/s] 92%|█████████▏| 366016/400000 [01:03<00:04, 7018.64it/s] 92%|█████████▏| 366723/400000 [01:03<00:04, 6792.58it/s] 92%|█████████▏| 367419/400000 [01:03<00:04, 6841.85it/s] 92%|█████████▏| 368162/400000 [01:03<00:04, 7007.18it/s] 92%|█████████▏| 368867/400000 [01:03<00:04, 6533.68it/s] 92%|█████████▏| 369530/400000 [01:03<00:04, 6193.90it/s] 93%|█████████▎| 370160/400000 [01:04<00:05, 5852.84it/s] 93%|█████████▎| 370767/400000 [01:04<00:04, 5916.06it/s] 93%|█████████▎| 371407/400000 [01:04<00:04, 6051.42it/s] 93%|█████████▎| 372125/400000 [01:04<00:04, 6348.96it/s] 93%|█████████▎| 372813/400000 [01:04<00:04, 6344.13it/s] 93%|█████████▎| 373454/400000 [01:04<00:04, 6254.92it/s] 94%|█████████▎| 374087/400000 [01:04<00:04, 5788.66it/s] 94%|█████████▎| 374676/400000 [01:04<00:04, 5404.35it/s] 94%|█████████▍| 375229/400000 [01:05<00:05, 4718.26it/s] 94%|█████████▍| 375725/400000 [01:05<00:05, 4746.45it/s] 94%|█████████▍| 376217/400000 [01:05<00:05, 4633.02it/s] 94%|█████████▍| 376693/400000 [01:05<00:05, 4357.96it/s] 94%|█████████▍| 377418/400000 [01:05<00:04, 4949.20it/s] 94%|█████████▍| 377949/400000 [01:05<00:04, 4425.97it/s] 95%|█████████▍| 378428/400000 [01:05<00:04, 4354.93it/s] 95%|█████████▍| 378889/400000 [01:05<00:05, 4174.72it/s] 95%|█████████▍| 379373/400000 [01:05<00:04, 4354.12it/s] 95%|█████████▍| 379824/400000 [01:06<00:05, 4022.57it/s] 95%|█████████▌| 380329/400000 [01:06<00:04, 4284.00it/s] 95%|█████████▌| 380870/400000 [01:06<00:04, 4568.01it/s] 95%|█████████▌| 381541/400000 [01:06<00:03, 5050.91it/s] 96%|█████████▌| 382146/400000 [01:06<00:03, 5312.50it/s] 96%|█████████▌| 382700/400000 [01:06<00:03, 5218.01it/s] 96%|█████████▌| 383270/400000 [01:06<00:03, 5265.09it/s] 96%|█████████▌| 383962/400000 [01:06<00:02, 5671.73it/s] 96%|█████████▌| 384553/400000 [01:06<00:02, 5739.79it/s] 96%|█████████▋| 385140/400000 [01:07<00:02, 5237.16it/s] 96%|█████████▋| 385767/400000 [01:07<00:02, 5509.36it/s] 97%|█████████▋| 386334/400000 [01:07<00:02, 5098.16it/s] 97%|█████████▋| 387028/400000 [01:07<00:02, 5537.91it/s] 97%|█████████▋| 387738/400000 [01:07<00:02, 5928.54it/s] 97%|█████████▋| 388378/400000 [01:07<00:01, 6062.33it/s] 97%|█████████▋| 389067/400000 [01:07<00:01, 6287.24it/s] 97%|█████████▋| 389787/400000 [01:07<00:01, 6535.26it/s] 98%|█████████▊| 390492/400000 [01:07<00:01, 6681.23it/s] 98%|█████████▊| 391170/400000 [01:07<00:01, 6622.39it/s] 98%|█████████▊| 391883/400000 [01:08<00:01, 6765.96it/s] 98%|█████████▊| 392612/400000 [01:08<00:01, 6913.81it/s] 98%|█████████▊| 393350/400000 [01:08<00:00, 7045.47it/s] 99%|█████████▊| 394059/400000 [01:08<00:01, 5583.72it/s] 99%|█████████▊| 394815/400000 [01:08<00:00, 6057.43it/s] 99%|█████████▉| 395527/400000 [01:08<00:00, 6339.25it/s] 99%|█████████▉| 396237/400000 [01:08<00:00, 6547.19it/s] 99%|█████████▉| 396919/400000 [01:08<00:00, 6592.97it/s] 99%|█████████▉| 397640/400000 [01:08<00:00, 6766.03it/s]100%|█████████▉| 398381/400000 [01:09<00:00, 6944.78it/s]100%|█████████▉| 399087/400000 [01:09<00:00, 6933.95it/s]100%|█████████▉| 399789/400000 [01:09<00:00, 6316.44it/s]100%|█████████▉| 399999/400000 [01:09<00:00, 5767.30it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1b3c0ecc50>, <torchtext.data.dataset.TabularDataset object at 0x7f1b3c0ecda0>, <torchtext.vocab.Vocab object at 0x7f1b3c0eccc0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.72 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.72 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  5.80 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  5.80 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.64 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.29 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.29 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.53 file/s]2020-07-21 06:22:06.871162: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-21 06:22:06.875318: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-21 06:22:06.875501: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ea36160d50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-21 06:22:06.875519: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3284992/9912422 [00:00<00:00, 32280049.87it/s]9920512it [00:00, 34943476.21it/s]                             
0it [00:00, ?it/s]32768it [00:00, 438510.87it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 976284.68it/s]1654784it [00:00, 12512357.29it/s]                          
0it [00:00, ?it/s]8192it [00:00, 234047.00it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
