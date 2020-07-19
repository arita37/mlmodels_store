
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f8a76eca9d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f8a76eca9d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f8ae1b35510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f8ae1b35510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f8b00d84ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f8b00d84ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8a8ee69950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8a8ee69950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8a8ee69950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 15%|█▌        | 1490944/9912422 [00:00<00:00, 14771260.89it/s] 54%|█████▍    | 5357568/9912422 [00:00<00:00, 18093296.24it/s] 69%|██████▉   | 6815744/9912422 [00:00<00:00, 15611332.92it/s]9920512it [00:00, 21488422.01it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1120404.94it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 464727.95it/s]1654784it [00:00, 11558974.50it/s]                         
0it [00:00, ?it/s]8192it [00:00, 214378.56it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8a8d3cb588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8a8d3d77f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f8a8ee69598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f8a8ee69598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f8a8ee69598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:15,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:14,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:13,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:13,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  2.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:50,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:49,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:49,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:49,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:32,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:21,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:21,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  8.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 34.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 34.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 42.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 49.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 58.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 65.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 65.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.25s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.25s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.25s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.05s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.25s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.42 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.05s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.05s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.99 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.05s/ url]
0 examples [00:00, ? examples/s]2020-07-19 18:07:53.184388: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-19 18:07:53.434288: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-19 18:07:53.434555: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56401fc05630 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-19 18:07:53.434589: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.32 examples/s]119 examples [00:00,  3.31 examples/s]238 examples [00:00,  4.72 examples/s]344 examples [00:00,  6.73 examples/s]450 examples [00:00,  9.59 examples/s]562 examples [00:00, 13.65 examples/s]680 examples [00:01, 19.40 examples/s]796 examples [00:01, 27.51 examples/s]905 examples [00:01, 38.88 examples/s]1024 examples [00:01, 54.77 examples/s]1144 examples [00:01, 76.74 examples/s]1263 examples [00:01, 106.66 examples/s]1378 examples [00:01, 146.52 examples/s]1493 examples [00:01, 198.25 examples/s]1607 examples [00:01, 263.56 examples/s]1725 examples [00:01, 343.48 examples/s]1843 examples [00:02, 436.25 examples/s]1962 examples [00:02, 538.45 examples/s]2079 examples [00:02, 634.18 examples/s]2194 examples [00:02, 725.97 examples/s]2313 examples [00:02, 821.96 examples/s]2432 examples [00:02, 904.14 examples/s]2551 examples [00:02, 972.93 examples/s]2668 examples [00:02, 1010.81 examples/s]2783 examples [00:02, 1003.91 examples/s]2898 examples [00:02, 1043.48 examples/s]3010 examples [00:03, 1051.27 examples/s]3125 examples [00:03, 1076.69 examples/s]3241 examples [00:03, 1099.89 examples/s]3358 examples [00:03, 1118.92 examples/s]3477 examples [00:03, 1137.94 examples/s]3596 examples [00:03, 1150.56 examples/s]3713 examples [00:03, 1152.84 examples/s]3830 examples [00:03, 1145.84 examples/s]3946 examples [00:03, 1146.27 examples/s]4061 examples [00:03, 1127.13 examples/s]4180 examples [00:04, 1143.65 examples/s]4295 examples [00:04, 1143.76 examples/s]4412 examples [00:04, 1149.80 examples/s]4532 examples [00:04, 1161.21 examples/s]4649 examples [00:04, 1163.34 examples/s]4768 examples [00:04, 1170.36 examples/s]4886 examples [00:04, 1166.00 examples/s]5003 examples [00:04, 1153.97 examples/s]5119 examples [00:04, 1143.18 examples/s]5238 examples [00:05, 1154.29 examples/s]5354 examples [00:05, 1152.43 examples/s]5470 examples [00:05, 1140.21 examples/s]5585 examples [00:05, 1133.06 examples/s]5704 examples [00:05, 1147.37 examples/s]5819 examples [00:05, 1142.29 examples/s]5934 examples [00:05, 1128.55 examples/s]6048 examples [00:05, 1130.64 examples/s]6162 examples [00:05, 1124.31 examples/s]6275 examples [00:05, 1125.47 examples/s]6388 examples [00:06, 1110.90 examples/s]6508 examples [00:06, 1133.90 examples/s]6622 examples [00:06, 1132.47 examples/s]6739 examples [00:06, 1141.64 examples/s]6858 examples [00:06, 1153.71 examples/s]6975 examples [00:06, 1158.08 examples/s]7094 examples [00:06, 1165.27 examples/s]7214 examples [00:06, 1171.84 examples/s]7332 examples [00:06, 1137.49 examples/s]7447 examples [00:06, 1139.96 examples/s]7562 examples [00:07, 1130.03 examples/s]7676 examples [00:07, 1119.47 examples/s]7789 examples [00:07, 1100.39 examples/s]7908 examples [00:07, 1124.38 examples/s]8027 examples [00:07, 1141.68 examples/s]8145 examples [00:07, 1151.79 examples/s]8264 examples [00:07, 1162.14 examples/s]8381 examples [00:07, 1143.94 examples/s]8500 examples [00:07, 1156.71 examples/s]8616 examples [00:07, 1154.43 examples/s]8735 examples [00:08, 1162.56 examples/s]8853 examples [00:08, 1167.22 examples/s]8970 examples [00:08, 1152.91 examples/s]9086 examples [00:08, 1136.23 examples/s]9205 examples [00:08, 1149.20 examples/s]9323 examples [00:08, 1155.76 examples/s]9440 examples [00:08, 1157.49 examples/s]9556 examples [00:08, 1132.29 examples/s]9670 examples [00:08, 1118.16 examples/s]9789 examples [00:08, 1136.82 examples/s]9904 examples [00:09, 1137.97 examples/s]10018 examples [00:09, 1083.23 examples/s]10127 examples [00:09, 1076.64 examples/s]10246 examples [00:09, 1107.08 examples/s]10367 examples [00:09, 1133.94 examples/s]10482 examples [00:09, 1137.84 examples/s]10599 examples [00:09, 1146.93 examples/s]10714 examples [00:09, 1134.87 examples/s]10833 examples [00:09, 1148.99 examples/s]10949 examples [00:10, 1150.71 examples/s]11067 examples [00:10, 1159.01 examples/s]11186 examples [00:10, 1167.90 examples/s]11303 examples [00:10, 1140.77 examples/s]11420 examples [00:10, 1149.13 examples/s]11539 examples [00:10, 1159.38 examples/s]11656 examples [00:10, 1159.23 examples/s]11775 examples [00:10, 1168.14 examples/s]11892 examples [00:10, 1124.81 examples/s]12005 examples [00:10, 1097.28 examples/s]12120 examples [00:11, 1110.42 examples/s]12232 examples [00:11, 1103.30 examples/s]12343 examples [00:11, 1092.30 examples/s]12453 examples [00:11, 1092.62 examples/s]12571 examples [00:11, 1116.77 examples/s]12688 examples [00:11, 1131.80 examples/s]12808 examples [00:11, 1149.22 examples/s]12925 examples [00:11, 1154.54 examples/s]13041 examples [00:11, 1115.24 examples/s]13154 examples [00:11, 1117.54 examples/s]13267 examples [00:12, 1118.63 examples/s]13385 examples [00:12, 1134.95 examples/s]13503 examples [00:12, 1147.69 examples/s]13618 examples [00:12, 1128.59 examples/s]13738 examples [00:12, 1147.99 examples/s]13857 examples [00:12, 1159.72 examples/s]13975 examples [00:12, 1164.96 examples/s]14095 examples [00:12, 1175.25 examples/s]14213 examples [00:12, 1155.02 examples/s]14329 examples [00:13, 1109.06 examples/s]14447 examples [00:13, 1128.63 examples/s]14564 examples [00:13, 1139.57 examples/s]14682 examples [00:13, 1149.38 examples/s]14798 examples [00:13, 1139.80 examples/s]14914 examples [00:13, 1145.60 examples/s]15029 examples [00:13, 1146.45 examples/s]15147 examples [00:13, 1154.21 examples/s]15266 examples [00:13, 1164.57 examples/s]15383 examples [00:13, 1133.87 examples/s]15505 examples [00:14, 1156.19 examples/s]15622 examples [00:14, 1160.01 examples/s]15740 examples [00:14, 1163.99 examples/s]15859 examples [00:14, 1170.70 examples/s]15977 examples [00:14, 1148.23 examples/s]16094 examples [00:14, 1152.39 examples/s]16210 examples [00:14, 1150.11 examples/s]16328 examples [00:14, 1156.73 examples/s]16444 examples [00:14, 1157.34 examples/s]16560 examples [00:14, 1134.65 examples/s]16674 examples [00:15, 1131.01 examples/s]16789 examples [00:15, 1133.74 examples/s]16903 examples [00:15, 1130.07 examples/s]17022 examples [00:15, 1145.64 examples/s]17137 examples [00:15, 1133.78 examples/s]17254 examples [00:15, 1143.50 examples/s]17372 examples [00:15, 1153.05 examples/s]17491 examples [00:15, 1161.92 examples/s]17610 examples [00:15, 1168.95 examples/s]17727 examples [00:15, 1145.68 examples/s]17844 examples [00:16, 1150.26 examples/s]17964 examples [00:16, 1162.00 examples/s]18081 examples [00:16, 1138.48 examples/s]18196 examples [00:16, 1102.35 examples/s]18309 examples [00:16, 1109.92 examples/s]18424 examples [00:16, 1118.79 examples/s]18540 examples [00:16, 1128.83 examples/s]18654 examples [00:16, 1130.58 examples/s]18768 examples [00:16, 1119.66 examples/s]18881 examples [00:17, 1048.58 examples/s]19000 examples [00:17, 1087.24 examples/s]19118 examples [00:17, 1110.57 examples/s]19237 examples [00:17, 1131.67 examples/s]19351 examples [00:17, 1126.10 examples/s]19466 examples [00:17, 1131.29 examples/s]19584 examples [00:17, 1143.47 examples/s]19702 examples [00:17, 1152.27 examples/s]19818 examples [00:17, 1154.23 examples/s]19934 examples [00:17, 1102.96 examples/s]20045 examples [00:18, 1060.90 examples/s]20164 examples [00:18, 1095.99 examples/s]20281 examples [00:18, 1115.68 examples/s]20398 examples [00:18, 1130.24 examples/s]20512 examples [00:18, 1090.41 examples/s]20627 examples [00:18, 1106.51 examples/s]20745 examples [00:18, 1127.56 examples/s]20863 examples [00:18, 1141.15 examples/s]20982 examples [00:18, 1153.16 examples/s]21098 examples [00:18, 1141.63 examples/s]21213 examples [00:19, 1137.31 examples/s]21331 examples [00:19, 1147.03 examples/s]21446 examples [00:19, 1135.93 examples/s]21564 examples [00:19, 1148.30 examples/s]21679 examples [00:19, 1136.77 examples/s]21798 examples [00:19, 1150.26 examples/s]21918 examples [00:19, 1163.03 examples/s]22037 examples [00:19, 1168.82 examples/s]22157 examples [00:19, 1176.50 examples/s]22275 examples [00:19, 1139.66 examples/s]22390 examples [00:20, 1132.77 examples/s]22508 examples [00:20, 1146.26 examples/s]22623 examples [00:20, 1117.05 examples/s]22740 examples [00:20, 1132.07 examples/s]22854 examples [00:20, 1116.09 examples/s]22970 examples [00:20, 1127.98 examples/s]23088 examples [00:20, 1142.66 examples/s]23205 examples [00:20, 1149.27 examples/s]23323 examples [00:20, 1157.18 examples/s]23439 examples [00:21, 1123.49 examples/s]23552 examples [00:21, 1107.23 examples/s]23668 examples [00:21, 1121.88 examples/s]23785 examples [00:21, 1135.61 examples/s]23899 examples [00:21, 1121.03 examples/s]24012 examples [00:21, 1101.27 examples/s]24123 examples [00:21, 1100.45 examples/s]24240 examples [00:21, 1119.18 examples/s]24357 examples [00:21, 1133.93 examples/s]24473 examples [00:21, 1138.94 examples/s]24588 examples [00:22, 1131.89 examples/s]24707 examples [00:22, 1145.77 examples/s]24825 examples [00:22, 1152.39 examples/s]24943 examples [00:22, 1160.32 examples/s]25060 examples [00:22, 1158.85 examples/s]25176 examples [00:22, 1122.57 examples/s]25293 examples [00:22, 1135.59 examples/s]25412 examples [00:22, 1151.31 examples/s]25530 examples [00:22, 1158.62 examples/s]25647 examples [00:22, 1120.54 examples/s]25760 examples [00:23, 1107.16 examples/s]25872 examples [00:23, 1097.72 examples/s]25989 examples [00:23, 1117.34 examples/s]26106 examples [00:23, 1130.93 examples/s]26220 examples [00:23, 1122.39 examples/s]26336 examples [00:23, 1132.90 examples/s]26455 examples [00:23, 1149.18 examples/s]26575 examples [00:23, 1162.03 examples/s]26694 examples [00:23, 1168.45 examples/s]26811 examples [00:23, 1150.50 examples/s]26929 examples [00:24, 1157.63 examples/s]27047 examples [00:24, 1163.12 examples/s]27166 examples [00:24, 1168.75 examples/s]27283 examples [00:24, 1168.05 examples/s]27400 examples [00:24, 1150.51 examples/s]27519 examples [00:24, 1161.82 examples/s]27637 examples [00:24, 1166.99 examples/s]27754 examples [00:24, 1152.75 examples/s]27873 examples [00:24, 1163.17 examples/s]27990 examples [00:25, 1146.00 examples/s]28105 examples [00:25, 1129.48 examples/s]28225 examples [00:25, 1148.61 examples/s]28341 examples [00:25, 1147.00 examples/s]28461 examples [00:25, 1160.09 examples/s]28578 examples [00:25, 1146.75 examples/s]28696 examples [00:25, 1153.91 examples/s]28816 examples [00:25, 1164.44 examples/s]28935 examples [00:25, 1171.06 examples/s]29053 examples [00:25, 1170.03 examples/s]29171 examples [00:26, 1117.47 examples/s]29284 examples [00:26, 1112.30 examples/s]29402 examples [00:26, 1131.58 examples/s]29516 examples [00:26, 1122.83 examples/s]29635 examples [00:26, 1140.87 examples/s]29750 examples [00:26, 1127.16 examples/s]29863 examples [00:26, 1103.57 examples/s]29975 examples [00:26, 1106.99 examples/s]30086 examples [00:26, 1063.09 examples/s]30202 examples [00:26, 1090.07 examples/s]30312 examples [00:27, 1080.92 examples/s]30421 examples [00:27, 1044.21 examples/s]30534 examples [00:27, 1067.97 examples/s]30649 examples [00:27, 1089.61 examples/s]30766 examples [00:27, 1112.11 examples/s]30878 examples [00:27, 1101.01 examples/s]30992 examples [00:27, 1112.26 examples/s]31104 examples [00:27, 1112.22 examples/s]31219 examples [00:27, 1122.33 examples/s]31332 examples [00:27, 1122.35 examples/s]31445 examples [00:28, 1104.00 examples/s]31558 examples [00:28, 1110.51 examples/s]31676 examples [00:28, 1128.97 examples/s]31795 examples [00:28, 1144.61 examples/s]31911 examples [00:28, 1148.33 examples/s]32026 examples [00:28, 1106.07 examples/s]32140 examples [00:28, 1115.98 examples/s]32255 examples [00:28, 1123.72 examples/s]32370 examples [00:28, 1129.59 examples/s]32484 examples [00:29, 1110.15 examples/s]32596 examples [00:29, 1087.72 examples/s]32706 examples [00:29, 1089.81 examples/s]32816 examples [00:29, 1075.02 examples/s]32934 examples [00:29, 1102.36 examples/s]33045 examples [00:29, 1090.72 examples/s]33157 examples [00:29, 1097.24 examples/s]33270 examples [00:29, 1106.06 examples/s]33381 examples [00:29, 1104.58 examples/s]33492 examples [00:29, 1104.75 examples/s]33603 examples [00:30, 1104.86 examples/s]33714 examples [00:30, 1104.17 examples/s]33833 examples [00:30, 1128.07 examples/s]33952 examples [00:30, 1145.10 examples/s]34067 examples [00:30, 1142.72 examples/s]34183 examples [00:30, 1147.18 examples/s]34298 examples [00:30, 1144.89 examples/s]34417 examples [00:30, 1158.02 examples/s]34533 examples [00:30, 1157.34 examples/s]34651 examples [00:30, 1161.87 examples/s]34768 examples [00:31, 1155.72 examples/s]34884 examples [00:31, 1127.04 examples/s]34997 examples [00:31, 1058.11 examples/s]35113 examples [00:31, 1083.64 examples/s]35231 examples [00:31, 1109.65 examples/s]35343 examples [00:31, 1108.27 examples/s]35455 examples [00:31, 1097.66 examples/s]35566 examples [00:31, 1083.77 examples/s]35675 examples [00:31, 1072.70 examples/s]35783 examples [00:31, 1068.68 examples/s]35891 examples [00:32, 1065.97 examples/s]36007 examples [00:32, 1090.72 examples/s]36125 examples [00:32, 1114.91 examples/s]36243 examples [00:32, 1131.61 examples/s]36361 examples [00:32, 1145.36 examples/s]36476 examples [00:32, 1134.83 examples/s]36595 examples [00:32, 1148.72 examples/s]36711 examples [00:32, 1134.97 examples/s]36825 examples [00:32, 1133.27 examples/s]36939 examples [00:33, 1124.04 examples/s]37052 examples [00:33, 1107.17 examples/s]37163 examples [00:33, 1102.78 examples/s]37278 examples [00:33, 1115.10 examples/s]37393 examples [00:33, 1123.69 examples/s]37511 examples [00:33, 1138.10 examples/s]37625 examples [00:33, 1120.81 examples/s]37745 examples [00:33, 1140.87 examples/s]37865 examples [00:33, 1156.18 examples/s]37985 examples [00:33, 1167.29 examples/s]38102 examples [00:34, 1157.98 examples/s]38218 examples [00:34, 1141.66 examples/s]38337 examples [00:34, 1153.90 examples/s]38453 examples [00:34, 1151.10 examples/s]38569 examples [00:34, 1152.48 examples/s]38687 examples [00:34, 1158.97 examples/s]38803 examples [00:34, 1144.36 examples/s]38922 examples [00:34, 1156.92 examples/s]39042 examples [00:34, 1166.87 examples/s]39161 examples [00:34, 1170.94 examples/s]39279 examples [00:35, 1162.46 examples/s]39396 examples [00:35, 1143.02 examples/s]39515 examples [00:35, 1156.24 examples/s]39631 examples [00:35, 1118.39 examples/s]39750 examples [00:35, 1138.91 examples/s]39869 examples [00:35, 1151.83 examples/s]39985 examples [00:35, 1140.52 examples/s]40100 examples [00:35, 1091.81 examples/s]40219 examples [00:35, 1117.63 examples/s]40332 examples [00:35, 1110.58 examples/s]40444 examples [00:36, 1100.95 examples/s]40559 examples [00:36, 1112.92 examples/s]40674 examples [00:36, 1123.15 examples/s]40787 examples [00:36, 1124.24 examples/s]40906 examples [00:36, 1141.56 examples/s]41021 examples [00:36, 1139.57 examples/s]41136 examples [00:36, 1140.40 examples/s]41255 examples [00:36, 1153.79 examples/s]41371 examples [00:36, 1116.88 examples/s]41484 examples [00:37, 1080.27 examples/s]41594 examples [00:37, 1083.77 examples/s]41713 examples [00:37, 1112.92 examples/s]41830 examples [00:37, 1127.26 examples/s]41944 examples [00:37, 1098.01 examples/s]42061 examples [00:37, 1117.74 examples/s]42174 examples [00:37, 1095.29 examples/s]42289 examples [00:37, 1110.95 examples/s]42408 examples [00:37, 1131.86 examples/s]42522 examples [00:37, 1133.25 examples/s]42636 examples [00:38, 1122.17 examples/s]42749 examples [00:38, 1118.57 examples/s]42868 examples [00:38, 1136.56 examples/s]42987 examples [00:38, 1150.05 examples/s]43103 examples [00:38, 1150.17 examples/s]43219 examples [00:38, 1139.63 examples/s]43334 examples [00:38, 1127.10 examples/s]43453 examples [00:38, 1144.60 examples/s]43572 examples [00:38, 1157.22 examples/s]43690 examples [00:38, 1163.19 examples/s]43807 examples [00:39, 1159.17 examples/s]43923 examples [00:39, 1118.66 examples/s]44041 examples [00:39, 1134.65 examples/s]44159 examples [00:39, 1146.06 examples/s]44274 examples [00:39, 1128.15 examples/s]44392 examples [00:39, 1141.99 examples/s]44507 examples [00:39, 1138.09 examples/s]44625 examples [00:39, 1148.20 examples/s]44744 examples [00:39, 1159.30 examples/s]44862 examples [00:39, 1164.42 examples/s]44979 examples [00:40, 1156.00 examples/s]45095 examples [00:40, 1134.85 examples/s]45211 examples [00:40, 1142.04 examples/s]45329 examples [00:40, 1151.90 examples/s]45447 examples [00:40, 1159.83 examples/s]45565 examples [00:40, 1165.29 examples/s]45682 examples [00:40, 1129.29 examples/s]45796 examples [00:40, 1124.26 examples/s]45909 examples [00:40, 1110.58 examples/s]46024 examples [00:41, 1120.99 examples/s]46137 examples [00:41, 1118.39 examples/s]46249 examples [00:41, 1069.19 examples/s]46364 examples [00:41, 1091.61 examples/s]46478 examples [00:41, 1105.17 examples/s]46589 examples [00:41, 1086.39 examples/s]46707 examples [00:41, 1110.27 examples/s]46819 examples [00:41, 1102.27 examples/s]46942 examples [00:41, 1135.12 examples/s]47058 examples [00:41, 1139.95 examples/s]47173 examples [00:42, 1132.54 examples/s]47287 examples [00:42, 1112.40 examples/s]47399 examples [00:42, 1095.90 examples/s]47510 examples [00:42, 1098.89 examples/s]47629 examples [00:42, 1122.15 examples/s]47746 examples [00:42, 1135.24 examples/s]47863 examples [00:42, 1144.54 examples/s]47978 examples [00:42, 1131.61 examples/s]48097 examples [00:42, 1146.98 examples/s]48216 examples [00:42, 1158.88 examples/s]48333 examples [00:43, 1151.70 examples/s]48449 examples [00:43, 1151.53 examples/s]48565 examples [00:43, 1131.28 examples/s]48684 examples [00:43, 1146.46 examples/s]48799 examples [00:43, 1142.75 examples/s]48914 examples [00:43, 1142.87 examples/s]49029 examples [00:43, 1137.37 examples/s]49143 examples [00:43, 1127.23 examples/s]49256 examples [00:43, 1122.84 examples/s]49371 examples [00:43, 1129.60 examples/s]49485 examples [00:44, 1120.53 examples/s]49602 examples [00:44, 1134.42 examples/s]49716 examples [00:44, 1121.49 examples/s]49835 examples [00:44, 1139.24 examples/s]49953 examples [00:44, 1150.02 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 20%|█▉        | 9989/50000 [00:00<00:00, 99885.81 examples/s] 48%|████▊     | 24072/50000 [00:00<00:00, 109428.39 examples/s] 78%|███████▊  | 38842/50000 [00:00<00:00, 118649.11 examples/s]                                                                0 examples [00:00, ? examples/s]96 examples [00:00, 957.48 examples/s]212 examples [00:00, 1009.46 examples/s]327 examples [00:00, 1045.79 examples/s]420 examples [00:00, 995.03 examples/s] 531 examples [00:00, 1024.98 examples/s]646 examples [00:00, 1057.81 examples/s]767 examples [00:00, 1099.14 examples/s]880 examples [00:00, 1107.21 examples/s]991 examples [00:00, 1105.54 examples/s]1111 examples [00:01, 1130.20 examples/s]1230 examples [00:01, 1146.70 examples/s]1348 examples [00:01, 1156.05 examples/s]1463 examples [00:01, 1150.75 examples/s]1578 examples [00:01, 1146.97 examples/s]1698 examples [00:01, 1160.17 examples/s]1817 examples [00:01, 1168.35 examples/s]1934 examples [00:01, 1159.73 examples/s]2050 examples [00:01, 1149.46 examples/s]2169 examples [00:01, 1159.03 examples/s]2289 examples [00:02, 1169.36 examples/s]2406 examples [00:02, 1141.55 examples/s]2521 examples [00:02, 1143.38 examples/s]2636 examples [00:02, 1082.66 examples/s]2746 examples [00:02, 1055.94 examples/s]2865 examples [00:02, 1091.72 examples/s]2984 examples [00:02, 1118.41 examples/s]3100 examples [00:02, 1129.88 examples/s]3214 examples [00:02, 1118.60 examples/s]3333 examples [00:02, 1137.06 examples/s]3448 examples [00:03, 1136.56 examples/s]3567 examples [00:03, 1150.13 examples/s]3687 examples [00:03, 1162.71 examples/s]3804 examples [00:03, 1147.98 examples/s]3922 examples [00:03, 1155.14 examples/s]4041 examples [00:03, 1164.65 examples/s]4158 examples [00:03, 1165.57 examples/s]4275 examples [00:03, 1146.11 examples/s]4390 examples [00:03, 1135.11 examples/s]4508 examples [00:03, 1147.33 examples/s]4626 examples [00:04, 1154.50 examples/s]4746 examples [00:04, 1165.08 examples/s]4866 examples [00:04, 1173.27 examples/s]4984 examples [00:04, 1143.83 examples/s]5100 examples [00:04, 1145.78 examples/s]5220 examples [00:04, 1160.51 examples/s]5337 examples [00:04, 1158.81 examples/s]5458 examples [00:04, 1171.74 examples/s]5576 examples [00:04, 1156.77 examples/s]5692 examples [00:05, 1155.62 examples/s]5808 examples [00:05, 1149.93 examples/s]5924 examples [00:05, 1149.34 examples/s]6039 examples [00:05, 1148.09 examples/s]6154 examples [00:05, 1119.35 examples/s]6273 examples [00:05, 1139.56 examples/s]6390 examples [00:05, 1147.20 examples/s]6509 examples [00:05, 1157.72 examples/s]6630 examples [00:05, 1170.58 examples/s]6748 examples [00:05, 1149.24 examples/s]6864 examples [00:06, 1149.29 examples/s]6982 examples [00:06, 1156.92 examples/s]7099 examples [00:06, 1160.14 examples/s]7216 examples [00:06, 1157.37 examples/s]7332 examples [00:06, 1090.78 examples/s]7452 examples [00:06, 1120.34 examples/s]7572 examples [00:06, 1142.07 examples/s]7693 examples [00:06, 1158.69 examples/s]7810 examples [00:06, 1149.20 examples/s]7926 examples [00:06, 1111.77 examples/s]8044 examples [00:07, 1130.33 examples/s]8163 examples [00:07, 1145.29 examples/s]8283 examples [00:07, 1158.64 examples/s]8403 examples [00:07, 1168.66 examples/s]8521 examples [00:07, 1123.90 examples/s]8634 examples [00:07, 1122.72 examples/s]8751 examples [00:07, 1135.91 examples/s]8868 examples [00:07, 1143.99 examples/s]8985 examples [00:07, 1150.74 examples/s]9101 examples [00:08, 1093.43 examples/s]9212 examples [00:08, 1060.02 examples/s]9320 examples [00:08, 1065.82 examples/s]9431 examples [00:08, 1078.14 examples/s]9543 examples [00:08, 1086.85 examples/s]9652 examples [00:08, 1046.32 examples/s]9765 examples [00:08, 1069.36 examples/s]9878 examples [00:08, 1086.81 examples/s]9992 examples [00:08, 1101.63 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete6AJ51R/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete6AJ51R/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8a8ee69950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8a8ee69950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8a8ee69950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8a8d3cb588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8a191cbe80>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8a8ee69950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8a8ee69950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8a8ee69950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8a191cbdd8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8a8d3d7358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f8a077c8e18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f8a077c8e18> 

  function with postional parmater data_info <function split_train_valid at 0x7f8a077c8e18> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f8a077c8f28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f8a077c8f28> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f8a077c8f28> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=bbce6854d3fa19f8cf2a5ec7c9ec5f60cb54722245c7df079606e8fc814a2b78
  Stored in directory: /tmp/pip-ephem-wheel-cache-3wyhvrlr/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:40:50, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:42:27, 16.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:20:54, 23.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:15:08, 33.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:03:49, 47.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.27M/862M [00:01<3:31:21, 67.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 12.9M/862M [00:01<2:27:26, 96.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:42:43, 137kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:01<1:11:38, 195kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:01<49:57, 279kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.5M/862M [00:01<34:55, 397kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:01<24:24, 565kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.1M/862M [00:02<17:06, 802kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.8M/862M [00:02<11:59, 1.14MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:02<08:27, 1.60MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<06:20, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<06:20, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<05:58, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<04:52, 2.75MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:05<03:36, 3.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<07:42, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<07:06, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:06<05:20, 2.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:07<03:52, 3.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<3:19:16, 66.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<2:21:00, 94.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:08<1:38:53, 134kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<1:11:45, 184kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<53:29, 247kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<38:07, 347kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:10<26:50, 491kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<23:02, 571kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<17:26, 754kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<12:31, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<11:50, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<09:36, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:14<06:59, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<07:59, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:41, 1.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:16<05:01, 2.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:16<03:41, 3.51MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<1:39:12, 131kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<1:12:02, 180kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<50:59, 254kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:18<35:42, 361kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<2:11:16, 98.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<1:32:58, 139kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<1:05:26, 197kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:20<45:47, 280kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<1:23:45, 153kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<59:53, 214kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:22<42:06, 303kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<32:26, 393kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<25:18, 504kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<18:16, 696kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<12:57, 980kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<14:34, 869kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:30, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<08:21, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:48, 1.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:26, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<05:30, 2.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:50, 1.83MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:51, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:27, 2.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<03:15, 3.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<49:14, 253kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<36:58, 337kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<26:24, 472kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<18:33, 669kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<23:37, 525kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<17:48, 696kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<12:43, 973kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<11:46, 1.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:44, 1.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:07, 1.51MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:39, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:36, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:52, 2.51MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:17, 1.94MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:32, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:17, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<03:59, 3.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:36, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:11, 2.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<03:55, 3.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:34, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:07, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:53, 3.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:31, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:04, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<03:51, 3.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:30, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:03, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:50, 3.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:29, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<04:52, 2.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<03:44, 3.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<02:45, 4.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<49:20, 239kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<36:56, 319kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<26:20, 447kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<18:37, 630kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<16:32, 708kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<12:47, 916kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<09:13, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<09:10, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:46, 1.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:39, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:46, 2.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<17:35, 658kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<13:28, 859kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:42, 1.19MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<09:28, 1.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:59, 1.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [00:59<06:52, 1.67MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:55, 2.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<17:04, 670kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<13:07, 871kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<09:27, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:15, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:46, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<06:39, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:47, 2.36MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<10:56:10, 17.2kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<7:40:15, 24.5kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<5:21:42, 35.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<3:47:06, 49.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<2:40:01, 70.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<1:51:59, 100kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<1:20:49, 138kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<57:40, 194kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<40:33, 275kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<30:56, 359kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<23:56, 464kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<17:16, 642kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<12:09, 908kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<21:58, 502kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<16:29, 668kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<11:48, 932kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<10:48, 1.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<08:40, 1.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<06:18, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:59, 1.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:05, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:30, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:36, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:14, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:52, 2.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 216M/862M [01:19<03:34, 3.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:40, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:48, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:19, 2.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:31, 1.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:56, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:43, 2.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:08, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:44, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:27, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:15, 3.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:03, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:01, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:25, 2.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:34, 1.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:57, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:44, 2.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:04, 2.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:35, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:26, 3.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:50, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:28, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:21, 2.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<04:42, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:22, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:16, 3.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:41, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:18, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:15, 3.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:41, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:19, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:09, 2.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<03:01, 3.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<07:58, 1.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:38, 1.52MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:53, 2.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:45, 1.74MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:01, 1.99MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:43, 2.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:58, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:29, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:20, 2.29MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:37, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:14, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:10, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:02, 3.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<1:22:02, 119kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<58:22, 168kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<41:00, 238kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<30:52, 315kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<23:39, 411kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<16:57, 573kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<12:02, 804kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<11:09, 866kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:48, 1.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<06:21, 1.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:40, 1.44MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:36, 1.45MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<05:02, 1.90MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:54<03:37, 2.63MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<17:45, 536kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<13:24, 710kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<09:35, 989kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<08:55, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:12, 1.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<05:16, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:54, 1.59MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:02, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:38, 2.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<03:20, 2.79MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<09:46, 953kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<07:49, 1.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<05:41, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<06:08, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:13, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:52, 2.37MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:53, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:16, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:05, 2.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<02:57, 3.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<12:04, 754kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<10:20, 881kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:38, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<05:29, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:57, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:47, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<04:16, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:05, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:24, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:14, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<03:04, 2.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<1:05:49, 135kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<46:58, 189kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<32:59, 269kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<25:04, 352kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<18:26, 479kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<13:05, 673kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<11:12, 783kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<09:39, 908kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<07:08, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<05:08, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<06:26, 1.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:22, 1.62MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:58, 2.18MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:47, 1.80MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:14, 2.03MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:10, 2.70MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:14, 2.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:42, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<03:43, 2.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<02:40, 3.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<20:15, 419kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<15:01, 565kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<10:40, 793kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<09:25, 893kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<07:26, 1.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:24, 1.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:45, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:44, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:34, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<02:35, 3.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<33:03, 251kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<23:57, 346kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<16:53, 489kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<13:44, 598kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<10:25, 788kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<07:27, 1.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:08, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:50, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<04:15, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<04:53, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:14, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:07, 2.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:04, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:15, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:23, 2.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:27, 3.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<06:31, 1.22MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:23, 1.47MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:55, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:35, 1.72MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:01, 1.96MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:00, 2.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:57, 1.98MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<21:32, 362kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<15:46, 491kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<11:53, 651kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<08:30, 906kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:36, 1.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:52, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<05:07, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:40, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:14, 1.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:08, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:46, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:24, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:38, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:33, 2.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<02:35, 2.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:56, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:13, 1.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:08, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:55, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:15, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:17, 2.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<02:27, 2.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:43, 1.97MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:22, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:30, 2.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [02:59<01:50, 3.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<28:14, 257kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<20:29, 354kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<14:28, 499kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<11:47, 610kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<09:41, 741kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<07:08, 1.00MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<05:03, 1.41MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:47, 810kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<06:52, 1.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<04:56, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:06, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:59, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:51, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<02:45, 2.54MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<6:46:28, 17.2kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<4:44:59, 24.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<3:18:56, 34.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<2:20:09, 49.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<1:38:42, 70.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<1:08:59, 99.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<49:40, 138kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<36:08, 189kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<25:32, 267kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<17:52, 380kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<16:05, 421kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<11:57, 566kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<08:29, 795kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<07:29, 896kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:54, 1.13MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<04:17, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:33, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:31, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:26, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<02:29, 2.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:40, 1.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:56, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:54, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:37, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:52, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:59, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:09, 2.98MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:23, 1.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:25, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:15, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:45, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:56, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:02, 2.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:13, 2.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:43, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:16, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:25, 2.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:09, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:29, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:45, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<01:59, 3.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<07:04, 871kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:34, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:02, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:17, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:14, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:13, 1.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:19, 2.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<05:07, 1.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:12, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:05, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:32, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:42, 1.61MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:50, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:06, 2.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:04, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:44, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:04, 2.83MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:48, 2.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:08, 1.85MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:27, 2.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<01:45, 3.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<11:50, 486kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<08:53, 646kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<06:20, 902kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:44, 990kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:29, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<03:16, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:20, 2.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<12:59, 432kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<09:39, 581kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<06:51, 814kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<06:05, 911kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<05:21, 1.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<04:00, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:53, 1.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:42, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:10, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:18, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<01:41, 3.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<21:16, 254kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<15:20, 352kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<10:57, 492kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<07:41, 696kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<09:02, 591kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<07:25, 718kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<05:27, 976kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<03:50, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<08:30, 620kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<06:28, 812kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<04:38, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:29, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:02, 1.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:05, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:15, 2.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:07, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:42, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<02:00, 2.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:35, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:51, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:13, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:38, 3.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:42, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:24, 2.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:47, 2.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:24, 2.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:39, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:06, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:31, 3.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:16, 1.48MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:47, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:04, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:34, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:49, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:10, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:36, 2.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:28, 1.90MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:12, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:39, 2.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:14, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:01, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:31, 3.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:09, 2.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<01:58, 2.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:28, 3.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:05, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:22, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:52, 2.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:21, 3.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<06:03, 734kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:41, 947kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:22, 1.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<03:22, 1.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:15, 1.34MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:27, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:49, 2.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:19, 1.85MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:08, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:35, 2.69MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<02:00, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:21, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:50, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:22, 3.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:05, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<01:54, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:25, 2.90MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:54, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:09, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:43, 2.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:50, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:41, 2.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:16, 3.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:49, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:04, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:37, 2.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:11, 3.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:27, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:07, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:34, 2.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:59, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:43, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<01:20, 2.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<00:57, 3.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<14:50, 253kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<10:45, 349kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<07:33, 493kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<06:07, 603kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<05:01, 734kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<03:40, 1.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<02:34, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<3:31:08, 17.2kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<2:27:54, 24.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<1:42:52, 34.9kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<1:12:07, 49.3kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<51:09, 69.4kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<35:50, 98.7kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<24:48, 141kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<20:30, 170kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<14:37, 238kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<10:17, 336kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<07:10, 477kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<07:55, 432kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<05:52, 580kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<04:10, 811kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:40, 910kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:51, 1.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:04, 1.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:12, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:52, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:22, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:43, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:31, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:07, 2.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:31, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:20, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:00, 3.10MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<00:44, 4.14MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<15:47, 195kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<11:21, 270kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<07:57, 382kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<06:12, 484kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:38, 646kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<03:17, 902kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:58, 986kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:40, 1.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:01, 1.45MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:25, 2.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<07:25, 386kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<05:28, 522kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:52, 732kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<03:19, 841kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:53, 967kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:08, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:30, 1.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:32, 1.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:19, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:45, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:14, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<20:00, 133kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<14:14, 186kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<09:56, 265kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<07:28, 347kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<05:44, 451kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<04:07, 626kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<02:51, 885kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<04:07, 612kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<03:08, 802kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<02:14, 1.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:07, 1.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:59, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:29, 1.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:05, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:25, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:14, 1.92MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:55, 2.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:10, 1.97MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:17, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:59, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:43, 3.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:32, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:18, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:57, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:10, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:02, 2.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:46, 2.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:01, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:08, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:54, 2.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:57, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:52, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:38, 3.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<00:54, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:02, 1.91MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:48, 2.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:34, 3.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<16:07, 118kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<11:26, 166kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<07:56, 236kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<05:53, 313kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<04:17, 427kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<03:00, 601kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:28, 714kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:05, 846kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:32, 1.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:19, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:16, 1.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:58, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:40, 2.40MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<12:08, 135kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<08:37, 189kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<05:58, 268kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<04:27, 351kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<03:15, 477kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<02:16, 672kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:55, 781kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:38, 910kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:12, 1.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:50, 1.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:30, 945kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:11, 1.19MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:51, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:54, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:33, 2.39MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:41, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:44, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:34, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:52<00:24, 3.04MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<02:30, 488kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:52, 650kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<01:18, 907kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:09, 993kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:02, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:46, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:32, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:49, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:40, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:29, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:34, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:21, 2.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:15, 3.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<07:15, 131kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<05:15, 180kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<03:40, 254kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<02:27, 362kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<02:32, 346kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:55, 455kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:22, 628kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:55, 889kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<01:29, 543kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:07, 719kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:46, 1.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:41, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:33, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:23, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:25, 1.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:21, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:15, 2.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:18, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:16, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:12, 2.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:15, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:13, 2.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:09, 3.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:13, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:14, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:11, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:11, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:07, 3.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:09, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:05, 3.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:07, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:04, 3.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:19, 393kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:13, 531kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:07, 744kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 851kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.49MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<12:11:30,  9.11it/s]  0%|          | 718/400000 [00:00<8:31:25, 13.01it/s]  0%|          | 1495/400000 [00:00<5:57:33, 18.58it/s]  1%|          | 2276/400000 [00:00<4:10:03, 26.51it/s]  1%|          | 3083/400000 [00:00<2:54:55, 37.82it/s]  1%|          | 3896/400000 [00:00<2:02:26, 53.92it/s]  1%|          | 4665/400000 [00:00<1:25:48, 76.79it/s]  1%|▏         | 5473/400000 [00:00<1:00:10, 109.26it/s]  2%|▏         | 6289/400000 [00:00<42:16, 155.19it/s]    2%|▏         | 7140/400000 [00:01<29:45, 219.99it/s]  2%|▏         | 8011/400000 [00:01<21:00, 310.90it/s]  2%|▏         | 8820/400000 [00:01<14:56, 436.42it/s]  2%|▏         | 9661/400000 [00:01<10:40, 609.89it/s]  3%|▎         | 10474/400000 [00:01<07:41, 844.11it/s]  3%|▎         | 11283/400000 [00:01<05:38, 1149.45it/s]  3%|▎         | 12134/400000 [00:01<04:09, 1552.17it/s]  3%|▎         | 12941/400000 [00:01<03:09, 2045.30it/s]  3%|▎         | 13784/400000 [00:01<02:25, 2646.66it/s]  4%|▎         | 14598/400000 [00:01<01:56, 3308.34it/s]  4%|▍         | 15427/400000 [00:02<01:35, 4035.27it/s]  4%|▍         | 16247/400000 [00:02<01:20, 4760.13it/s]  4%|▍         | 17071/400000 [00:02<01:10, 5450.47it/s]  4%|▍         | 17906/400000 [00:02<01:02, 6083.45it/s]  5%|▍         | 18730/400000 [00:02<00:58, 6529.29it/s]  5%|▍         | 19543/400000 [00:02<00:55, 6912.62it/s]  5%|▌         | 20352/400000 [00:02<00:52, 7165.11it/s]  5%|▌         | 21154/400000 [00:02<00:51, 7348.46it/s]  6%|▌         | 22012/400000 [00:02<00:49, 7677.36it/s]  6%|▌         | 22826/400000 [00:02<00:48, 7807.65it/s]  6%|▌         | 23662/400000 [00:03<00:47, 7965.45it/s]  6%|▌         | 24508/400000 [00:03<00:46, 8107.33it/s]  6%|▋         | 25336/400000 [00:03<00:46, 8136.65it/s]  7%|▋         | 26202/400000 [00:03<00:45, 8284.45it/s]  7%|▋         | 27040/400000 [00:03<00:45, 8231.89it/s]  7%|▋         | 27877/400000 [00:03<00:44, 8271.66it/s]  7%|▋         | 28734/400000 [00:03<00:44, 8357.90it/s]  7%|▋         | 29574/400000 [00:03<00:44, 8287.08it/s]  8%|▊         | 30430/400000 [00:03<00:44, 8366.77it/s]  8%|▊         | 31269/400000 [00:03<00:44, 8261.63it/s]  8%|▊         | 32114/400000 [00:04<00:44, 8316.92it/s]  8%|▊         | 32978/400000 [00:04<00:43, 8409.64it/s]  8%|▊         | 33821/400000 [00:04<00:43, 8343.40it/s]  9%|▊         | 34683/400000 [00:04<00:43, 8422.51it/s]  9%|▉         | 35527/400000 [00:04<00:43, 8299.08it/s]  9%|▉         | 36358/400000 [00:04<00:44, 8263.18it/s]  9%|▉         | 37216/400000 [00:04<00:43, 8355.09it/s] 10%|▉         | 38053/400000 [00:04<00:43, 8342.50it/s] 10%|▉         | 38925/400000 [00:04<00:42, 8451.25it/s] 10%|▉         | 39788/400000 [00:04<00:42, 8503.82it/s] 10%|█         | 40639/400000 [00:05<00:42, 8503.52it/s] 10%|█         | 41519/400000 [00:05<00:41, 8586.89it/s] 11%|█         | 42379/400000 [00:05<00:43, 8311.59it/s] 11%|█         | 43213/400000 [00:05<00:43, 8276.64it/s] 11%|█         | 44043/400000 [00:05<00:42, 8279.94it/s] 11%|█         | 44873/400000 [00:05<00:42, 8263.05it/s] 11%|█▏        | 45711/400000 [00:05<00:42, 8296.44it/s] 12%|█▏        | 46542/400000 [00:05<00:42, 8223.41it/s] 12%|█▏        | 47393/400000 [00:05<00:42, 8305.03it/s] 12%|█▏        | 48233/400000 [00:05<00:42, 8332.14it/s] 12%|█▏        | 49079/400000 [00:06<00:41, 8368.18it/s] 12%|█▏        | 49917/400000 [00:06<00:41, 8352.89it/s] 13%|█▎        | 50753/400000 [00:06<00:42, 8274.08it/s] 13%|█▎        | 51606/400000 [00:06<00:41, 8346.16it/s] 13%|█▎        | 52442/400000 [00:06<00:41, 8322.34it/s] 13%|█▎        | 53309/400000 [00:06<00:41, 8423.64it/s] 14%|█▎        | 54175/400000 [00:06<00:40, 8492.21it/s] 14%|█▍        | 55025/400000 [00:06<00:40, 8463.21it/s] 14%|█▍        | 55875/400000 [00:06<00:40, 8472.56it/s] 14%|█▍        | 56733/400000 [00:06<00:40, 8503.67it/s] 14%|█▍        | 57584/400000 [00:07<00:40, 8500.74it/s] 15%|█▍        | 58435/400000 [00:07<00:40, 8369.85it/s] 15%|█▍        | 59273/400000 [00:07<00:41, 8131.71it/s] 15%|█▌        | 60093/400000 [00:07<00:41, 8149.78it/s] 15%|█▌        | 60927/400000 [00:07<00:41, 8203.85it/s] 15%|█▌        | 61749/400000 [00:07<00:41, 8163.17it/s] 16%|█▌        | 62573/400000 [00:07<00:41, 8185.99it/s] 16%|█▌        | 63393/400000 [00:07<00:41, 8138.56it/s] 16%|█▌        | 64236/400000 [00:07<00:40, 8220.94it/s] 16%|█▋        | 65104/400000 [00:08<00:40, 8352.09it/s] 16%|█▋        | 65955/400000 [00:08<00:39, 8398.59it/s] 17%|█▋        | 66798/400000 [00:08<00:39, 8405.12it/s] 17%|█▋        | 67673/400000 [00:08<00:39, 8503.29it/s] 17%|█▋        | 68524/400000 [00:08<00:40, 8235.18it/s] 17%|█▋        | 69350/400000 [00:08<00:40, 8212.32it/s] 18%|█▊        | 70202/400000 [00:08<00:39, 8301.53it/s] 18%|█▊        | 71034/400000 [00:08<00:40, 8098.08it/s] 18%|█▊        | 71865/400000 [00:08<00:40, 8156.55it/s] 18%|█▊        | 72685/400000 [00:08<00:40, 8168.34it/s] 18%|█▊        | 73541/400000 [00:09<00:39, 8280.29it/s] 19%|█▊        | 74396/400000 [00:09<00:38, 8358.68it/s] 19%|█▉        | 75267/400000 [00:09<00:38, 8460.89it/s] 19%|█▉        | 76115/400000 [00:09<00:38, 8385.57it/s] 19%|█▉        | 76955/400000 [00:09<00:39, 8228.83it/s] 19%|█▉        | 77840/400000 [00:09<00:38, 8404.28it/s] 20%|█▉        | 78697/400000 [00:09<00:38, 8450.70it/s] 20%|█▉        | 79544/400000 [00:09<00:38, 8320.66it/s] 20%|██        | 80412/400000 [00:09<00:37, 8423.66it/s] 20%|██        | 81256/400000 [00:09<00:38, 8326.15it/s] 21%|██        | 82123/400000 [00:10<00:37, 8424.96it/s] 21%|██        | 82999/400000 [00:10<00:37, 8522.11it/s] 21%|██        | 83853/400000 [00:10<00:37, 8496.02it/s] 21%|██        | 84719/400000 [00:10<00:36, 8543.10it/s] 21%|██▏       | 85574/400000 [00:10<00:37, 8361.14it/s] 22%|██▏       | 86428/400000 [00:10<00:37, 8413.39it/s] 22%|██▏       | 87297/400000 [00:10<00:36, 8490.69it/s] 22%|██▏       | 88147/400000 [00:10<00:36, 8490.56it/s] 22%|██▏       | 89014/400000 [00:10<00:36, 8543.02it/s] 22%|██▏       | 89869/400000 [00:10<00:37, 8324.90it/s] 23%|██▎       | 90709/400000 [00:11<00:37, 8345.83it/s] 23%|██▎       | 91585/400000 [00:11<00:36, 8463.47it/s] 23%|██▎       | 92440/400000 [00:11<00:36, 8488.92it/s] 23%|██▎       | 93291/400000 [00:11<00:36, 8494.68it/s] 24%|██▎       | 94142/400000 [00:11<00:36, 8399.16it/s] 24%|██▍       | 95005/400000 [00:11<00:36, 8465.11it/s] 24%|██▍       | 95861/400000 [00:11<00:35, 8491.92it/s] 24%|██▍       | 96711/400000 [00:11<00:35, 8484.28it/s] 24%|██▍       | 97568/400000 [00:11<00:35, 8506.77it/s] 25%|██▍       | 98419/400000 [00:11<00:35, 8402.94it/s] 25%|██▍       | 99271/400000 [00:12<00:35, 8436.80it/s] 25%|██▌       | 100116/400000 [00:12<00:35, 8400.19it/s] 25%|██▌       | 100957/400000 [00:12<00:36, 8243.68it/s] 25%|██▌       | 101783/400000 [00:12<00:36, 8171.79it/s] 26%|██▌       | 102601/400000 [00:12<00:36, 8074.61it/s] 26%|██▌       | 103416/400000 [00:12<00:36, 8095.55it/s] 26%|██▌       | 104273/400000 [00:12<00:35, 8229.12it/s] 26%|██▋       | 105118/400000 [00:12<00:35, 8292.94it/s] 26%|██▋       | 105982/400000 [00:12<00:35, 8391.97it/s] 27%|██▋       | 106823/400000 [00:12<00:35, 8321.19it/s] 27%|██▋       | 107686/400000 [00:13<00:34, 8410.40it/s] 27%|██▋       | 108550/400000 [00:13<00:34, 8477.67it/s] 27%|██▋       | 109400/400000 [00:13<00:34, 8483.49it/s] 28%|██▊       | 110249/400000 [00:13<00:34, 8467.02it/s] 28%|██▊       | 111097/400000 [00:13<00:34, 8379.09it/s] 28%|██▊       | 111973/400000 [00:13<00:33, 8487.86it/s] 28%|██▊       | 112829/400000 [00:13<00:33, 8507.57it/s] 28%|██▊       | 113681/400000 [00:13<00:34, 8410.67it/s] 29%|██▊       | 114532/400000 [00:13<00:33, 8437.79it/s] 29%|██▉       | 115377/400000 [00:14<00:34, 8359.44it/s] 29%|██▉       | 116220/400000 [00:14<00:33, 8379.11it/s] 29%|██▉       | 117059/400000 [00:14<00:33, 8351.66it/s] 29%|██▉       | 117909/400000 [00:14<00:33, 8393.81it/s] 30%|██▉       | 118764/400000 [00:14<00:33, 8438.86it/s] 30%|██▉       | 119609/400000 [00:14<00:33, 8337.23it/s] 30%|███       | 120444/400000 [00:14<00:33, 8291.89it/s] 30%|███       | 121311/400000 [00:14<00:33, 8400.73it/s] 31%|███       | 122152/400000 [00:14<00:33, 8363.72it/s] 31%|███       | 122991/400000 [00:14<00:33, 8371.11it/s] 31%|███       | 123829/400000 [00:15<00:33, 8296.16it/s] 31%|███       | 124696/400000 [00:15<00:32, 8404.27it/s] 31%|███▏      | 125571/400000 [00:15<00:32, 8504.21it/s] 32%|███▏      | 126423/400000 [00:15<00:32, 8498.91it/s] 32%|███▏      | 127274/400000 [00:15<00:32, 8383.82it/s] 32%|███▏      | 128114/400000 [00:15<00:32, 8261.74it/s] 32%|███▏      | 128942/400000 [00:15<00:33, 8191.79it/s] 32%|███▏      | 129762/400000 [00:15<00:33, 8174.67it/s] 33%|███▎      | 130606/400000 [00:15<00:32, 8252.39it/s] 33%|███▎      | 131455/400000 [00:15<00:32, 8321.11it/s] 33%|███▎      | 132288/400000 [00:16<00:32, 8232.95it/s] 33%|███▎      | 133141/400000 [00:16<00:32, 8316.87it/s] 33%|███▎      | 133984/400000 [00:16<00:31, 8349.75it/s] 34%|███▎      | 134846/400000 [00:16<00:31, 8426.24it/s] 34%|███▍      | 135690/400000 [00:16<00:31, 8386.68it/s] 34%|███▍      | 136530/400000 [00:16<00:31, 8314.96it/s] 34%|███▍      | 137382/400000 [00:16<00:31, 8374.18it/s] 35%|███▍      | 138238/400000 [00:16<00:31, 8429.02it/s] 35%|███▍      | 139101/400000 [00:16<00:30, 8486.80it/s] 35%|███▍      | 139951/400000 [00:16<00:31, 8317.82it/s] 35%|███▌      | 140784/400000 [00:17<00:31, 8241.94it/s] 35%|███▌      | 141633/400000 [00:17<00:31, 8314.29it/s] 36%|███▌      | 142505/400000 [00:17<00:30, 8429.42it/s] 36%|███▌      | 143349/400000 [00:17<00:30, 8369.33it/s] 36%|███▌      | 144187/400000 [00:17<00:30, 8322.52it/s] 36%|███▋      | 145020/400000 [00:17<00:31, 8108.80it/s] 36%|███▋      | 145833/400000 [00:17<00:31, 8058.83it/s] 37%|███▋      | 146685/400000 [00:17<00:30, 8191.38it/s] 37%|███▋      | 147547/400000 [00:17<00:30, 8313.27it/s] 37%|███▋      | 148380/400000 [00:17<00:30, 8304.97it/s] 37%|███▋      | 149212/400000 [00:18<00:30, 8276.19it/s] 38%|███▊      | 150073/400000 [00:18<00:29, 8372.83it/s] 38%|███▊      | 150942/400000 [00:18<00:29, 8462.76it/s] 38%|███▊      | 151802/400000 [00:18<00:29, 8500.89it/s] 38%|███▊      | 152653/400000 [00:18<00:29, 8417.45it/s] 38%|███▊      | 153496/400000 [00:18<00:29, 8293.72it/s] 39%|███▊      | 154359/400000 [00:18<00:29, 8389.31it/s] 39%|███▉      | 155227/400000 [00:18<00:28, 8471.81it/s] 39%|███▉      | 156100/400000 [00:18<00:28, 8546.05it/s] 39%|███▉      | 156956/400000 [00:18<00:28, 8423.23it/s] 39%|███▉      | 157800/400000 [00:19<00:29, 8333.18it/s] 40%|███▉      | 158662/400000 [00:19<00:28, 8417.16it/s] 40%|███▉      | 159524/400000 [00:19<00:28, 8475.22it/s] 40%|████      | 160380/400000 [00:19<00:28, 8499.65it/s] 40%|████      | 161231/400000 [00:19<00:28, 8500.09it/s] 41%|████      | 162082/400000 [00:19<00:28, 8446.59it/s] 41%|████      | 162927/400000 [00:19<00:28, 8411.31it/s] 41%|████      | 163785/400000 [00:19<00:27, 8458.29it/s] 41%|████      | 164632/400000 [00:19<00:27, 8451.78it/s] 41%|████▏     | 165478/400000 [00:19<00:27, 8438.06it/s] 42%|████▏     | 166322/400000 [00:20<00:28, 8331.76it/s] 42%|████▏     | 167185/400000 [00:20<00:27, 8417.57it/s] 42%|████▏     | 168043/400000 [00:20<00:27, 8462.48it/s] 42%|████▏     | 168890/400000 [00:20<00:27, 8428.83it/s] 42%|████▏     | 169734/400000 [00:20<00:27, 8390.85it/s] 43%|████▎     | 170574/400000 [00:20<00:27, 8257.38it/s] 43%|████▎     | 171413/400000 [00:20<00:27, 8295.49it/s] 43%|████▎     | 172266/400000 [00:20<00:27, 8364.38it/s] 43%|████▎     | 173118/400000 [00:20<00:26, 8410.23it/s] 43%|████▎     | 173970/400000 [00:21<00:26, 8441.01it/s] 44%|████▎     | 174815/400000 [00:21<00:27, 8332.32it/s] 44%|████▍     | 175671/400000 [00:21<00:26, 8398.47it/s] 44%|████▍     | 176523/400000 [00:21<00:26, 8433.58it/s] 44%|████▍     | 177374/400000 [00:21<00:26, 8454.19it/s] 45%|████▍     | 178225/400000 [00:21<00:26, 8468.51it/s] 45%|████▍     | 179073/400000 [00:21<00:26, 8366.65it/s] 45%|████▍     | 179919/400000 [00:21<00:26, 8393.48it/s] 45%|████▌     | 180779/400000 [00:21<00:25, 8452.77it/s] 45%|████▌     | 181637/400000 [00:21<00:25, 8487.39it/s] 46%|████▌     | 182486/400000 [00:22<00:26, 8347.51it/s] 46%|████▌     | 183322/400000 [00:22<00:26, 8213.83it/s] 46%|████▌     | 184165/400000 [00:22<00:26, 8277.01it/s] 46%|████▋     | 185021/400000 [00:22<00:25, 8357.67it/s] 46%|████▋     | 185858/400000 [00:22<00:25, 8346.09it/s] 47%|████▋     | 186694/400000 [00:22<00:25, 8320.48it/s] 47%|████▋     | 187527/400000 [00:22<00:25, 8230.70it/s] 47%|████▋     | 188351/400000 [00:22<00:26, 8137.65it/s] 47%|████▋     | 189187/400000 [00:22<00:25, 8201.76it/s] 48%|████▊     | 190057/400000 [00:22<00:25, 8344.44it/s] 48%|████▊     | 190900/400000 [00:23<00:24, 8368.10it/s] 48%|████▊     | 191738/400000 [00:23<00:25, 8325.07it/s] 48%|████▊     | 192596/400000 [00:23<00:24, 8399.30it/s] 48%|████▊     | 193465/400000 [00:23<00:24, 8484.07it/s] 49%|████▊     | 194314/400000 [00:23<00:24, 8431.99it/s] 49%|████▉     | 195159/400000 [00:23<00:24, 8437.06it/s] 49%|████▉     | 196004/400000 [00:23<00:24, 8416.30it/s] 49%|████▉     | 196846/400000 [00:23<00:24, 8415.34it/s] 49%|████▉     | 197688/400000 [00:23<00:24, 8399.20it/s] 50%|████▉     | 198557/400000 [00:23<00:23, 8483.85it/s] 50%|████▉     | 199421/400000 [00:24<00:23, 8527.89it/s] 50%|█████     | 200275/400000 [00:24<00:23, 8490.51it/s] 50%|█████     | 201125/400000 [00:24<00:23, 8336.48it/s] 50%|█████     | 201960/400000 [00:24<00:23, 8272.18it/s] 51%|█████     | 202813/400000 [00:24<00:23, 8345.77it/s] 51%|█████     | 203668/400000 [00:24<00:23, 8405.84it/s] 51%|█████     | 204510/400000 [00:24<00:23, 8386.36it/s] 51%|█████▏    | 205355/400000 [00:24<00:23, 8404.91it/s] 52%|█████▏    | 206235/400000 [00:24<00:22, 8519.18it/s] 52%|█████▏    | 207097/400000 [00:24<00:22, 8548.29it/s] 52%|█████▏    | 207953/400000 [00:25<00:22, 8483.84it/s] 52%|█████▏    | 208802/400000 [00:25<00:22, 8406.62it/s] 52%|█████▏    | 209657/400000 [00:25<00:22, 8448.87it/s] 53%|█████▎    | 210503/400000 [00:25<00:22, 8446.92it/s] 53%|█████▎    | 211353/400000 [00:25<00:22, 8462.26it/s] 53%|█████▎    | 212200/400000 [00:25<00:22, 8403.94it/s] 53%|█████▎    | 213041/400000 [00:25<00:22, 8362.04it/s] 53%|█████▎    | 213879/400000 [00:25<00:22, 8365.96it/s] 54%|█████▎    | 214735/400000 [00:25<00:21, 8421.39it/s] 54%|█████▍    | 215601/400000 [00:25<00:21, 8491.12it/s] 54%|█████▍    | 216459/400000 [00:26<00:21, 8515.21it/s] 54%|█████▍    | 217311/400000 [00:26<00:21, 8418.33it/s] 55%|█████▍    | 218173/400000 [00:26<00:21, 8477.22it/s] 55%|█████▍    | 219039/400000 [00:26<00:21, 8530.64it/s] 55%|█████▍    | 219893/400000 [00:26<00:21, 8505.31it/s] 55%|█████▌    | 220744/400000 [00:26<00:21, 8505.85it/s] 55%|█████▌    | 221595/400000 [00:26<00:21, 8409.97it/s] 56%|█████▌    | 222442/400000 [00:26<00:21, 8427.22it/s] 56%|█████▌    | 223293/400000 [00:26<00:20, 8450.70it/s] 56%|█████▌    | 224139/400000 [00:26<00:20, 8397.40it/s] 56%|█████▌    | 224984/400000 [00:27<00:20, 8412.23it/s] 56%|█████▋    | 225826/400000 [00:27<00:20, 8407.27it/s] 57%|█████▋    | 226667/400000 [00:27<00:20, 8402.75it/s] 57%|█████▋    | 227516/400000 [00:27<00:20, 8427.12it/s] 57%|█████▋    | 228382/400000 [00:27<00:20, 8495.39it/s] 57%|█████▋    | 229248/400000 [00:27<00:19, 8542.87it/s] 58%|█████▊    | 230103/400000 [00:27<00:20, 8360.05it/s] 58%|█████▊    | 230959/400000 [00:27<00:20, 8416.62it/s] 58%|█████▊    | 231802/400000 [00:27<00:20, 8337.13it/s] 58%|█████▊    | 232637/400000 [00:27<00:20, 8320.09it/s] 58%|█████▊    | 233483/400000 [00:28<00:19, 8360.08it/s] 59%|█████▊    | 234328/400000 [00:28<00:19, 8385.48it/s] 59%|█████▉    | 235167/400000 [00:28<00:19, 8371.44it/s] 59%|█████▉    | 236029/400000 [00:28<00:19, 8442.71it/s] 59%|█████▉    | 236899/400000 [00:28<00:19, 8517.43it/s] 59%|█████▉    | 237752/400000 [00:28<00:19, 8394.81it/s] 60%|█████▉    | 238593/400000 [00:28<00:19, 8288.53it/s] 60%|█████▉    | 239425/400000 [00:28<00:19, 8297.41it/s] 60%|██████    | 240295/400000 [00:28<00:18, 8412.23it/s] 60%|██████    | 241137/400000 [00:29<00:18, 8373.64it/s] 61%|██████    | 242053/400000 [00:29<00:18, 8593.88it/s] 61%|██████    | 242920/400000 [00:29<00:18, 8613.88it/s] 61%|██████    | 243913/400000 [00:29<00:17, 8968.38it/s] 61%|██████    | 244886/400000 [00:29<00:16, 9183.00it/s] 61%|██████▏   | 245905/400000 [00:29<00:16, 9462.87it/s] 62%|██████▏   | 246864/400000 [00:29<00:16, 9497.78it/s] 62%|██████▏   | 247846/400000 [00:29<00:15, 9589.20it/s] 62%|██████▏   | 248808/400000 [00:29<00:15, 9506.29it/s] 62%|██████▏   | 249795/400000 [00:29<00:15, 9610.58it/s] 63%|██████▎   | 250758/400000 [00:30<00:15, 9408.37it/s] 63%|██████▎   | 251702/400000 [00:30<00:15, 9311.39it/s] 63%|██████▎   | 252638/400000 [00:30<00:15, 9321.66it/s] 63%|██████▎   | 253572/400000 [00:30<00:15, 9247.53it/s] 64%|██████▎   | 254554/400000 [00:30<00:15, 9410.02it/s] 64%|██████▍   | 255522/400000 [00:30<00:15, 9488.48it/s] 64%|██████▍   | 256508/400000 [00:30<00:14, 9596.66it/s] 64%|██████▍   | 257504/400000 [00:30<00:14, 9702.23it/s] 65%|██████▍   | 258476/400000 [00:30<00:14, 9484.07it/s] 65%|██████▍   | 259434/400000 [00:30<00:14, 9511.79it/s] 65%|██████▌   | 260387/400000 [00:31<00:14, 9427.39it/s] 65%|██████▌   | 261339/400000 [00:31<00:14, 9454.93it/s] 66%|██████▌   | 262286/400000 [00:31<00:14, 9385.21it/s] 66%|██████▌   | 263226/400000 [00:31<00:14, 9268.81it/s] 66%|██████▌   | 264154/400000 [00:31<00:14, 9267.02it/s] 66%|██████▋   | 265119/400000 [00:31<00:14, 9377.43it/s] 67%|██████▋   | 266078/400000 [00:31<00:14, 9438.41it/s] 67%|██████▋   | 267023/400000 [00:31<00:14, 9418.64it/s] 67%|██████▋   | 267966/400000 [00:31<00:14, 9155.77it/s] 67%|██████▋   | 268922/400000 [00:31<00:14, 9269.10it/s] 67%|██████▋   | 269863/400000 [00:32<00:13, 9308.56it/s] 68%|██████▊   | 270818/400000 [00:32<00:13, 9376.37it/s] 68%|██████▊   | 271757/400000 [00:32<00:13, 9257.52it/s] 68%|██████▊   | 272684/400000 [00:32<00:14, 8958.90it/s] 68%|██████▊   | 273583/400000 [00:32<00:14, 8920.37it/s] 69%|██████▊   | 274478/400000 [00:32<00:14, 8832.18it/s] 69%|██████▉   | 275403/400000 [00:32<00:13, 8951.78it/s] 69%|██████▉   | 276348/400000 [00:32<00:13, 9093.04it/s] 69%|██████▉   | 277259/400000 [00:32<00:13, 9036.78it/s] 70%|██████▉   | 278189/400000 [00:32<00:13, 9111.64it/s] 70%|██████▉   | 279102/400000 [00:33<00:13, 9116.91it/s] 70%|███████   | 280015/400000 [00:33<00:13, 8860.04it/s] 70%|███████   | 280930/400000 [00:33<00:13, 8942.21it/s] 70%|███████   | 281826/400000 [00:33<00:13, 8888.58it/s] 71%|███████   | 282751/400000 [00:33<00:13, 8993.77it/s] 71%|███████   | 283680/400000 [00:33<00:12, 9078.37it/s] 71%|███████   | 284643/400000 [00:33<00:12, 9234.60it/s] 71%|███████▏  | 285579/400000 [00:33<00:12, 9269.86it/s] 72%|███████▏  | 286507/400000 [00:33<00:12, 9088.44it/s] 72%|███████▏  | 287418/400000 [00:33<00:12, 9072.31it/s] 72%|███████▏  | 288327/400000 [00:34<00:12, 8951.82it/s] 72%|███████▏  | 289303/400000 [00:34<00:12, 9179.28it/s] 73%|███████▎  | 290240/400000 [00:34<00:11, 9234.16it/s] 73%|███████▎  | 291177/400000 [00:34<00:11, 9273.08it/s] 73%|███████▎  | 292163/400000 [00:34<00:11, 9439.77it/s] 73%|███████▎  | 293140/400000 [00:34<00:11, 9534.86it/s] 74%|███████▎  | 294095/400000 [00:34<00:11, 9526.19it/s] 74%|███████▍  | 295049/400000 [00:34<00:11, 9460.28it/s] 74%|███████▍  | 296013/400000 [00:34<00:10, 9512.15it/s] 74%|███████▍  | 296965/400000 [00:35<00:10, 9407.23it/s] 74%|███████▍  | 297936/400000 [00:35<00:10, 9494.53it/s] 75%|███████▍  | 298937/400000 [00:35<00:10, 9642.13it/s] 75%|███████▍  | 299903/400000 [00:35<00:10, 9620.14it/s] 75%|███████▌  | 300866/400000 [00:35<00:10, 9457.69it/s] 75%|███████▌  | 301867/400000 [00:35<00:10, 9615.27it/s] 76%|███████▌  | 302837/400000 [00:35<00:10, 9638.95it/s] 76%|███████▌  | 303835/400000 [00:35<00:09, 9737.73it/s] 76%|███████▌  | 304810/400000 [00:35<00:09, 9606.13it/s] 76%|███████▋  | 305807/400000 [00:35<00:09, 9711.44it/s] 77%|███████▋  | 306781/400000 [00:36<00:09, 9718.56it/s] 77%|███████▋  | 307779/400000 [00:36<00:09, 9793.34it/s] 77%|███████▋  | 308766/400000 [00:36<00:09, 9815.07it/s] 77%|███████▋  | 309748/400000 [00:36<00:09, 9629.38it/s] 78%|███████▊  | 310713/400000 [00:36<00:09, 9587.55it/s] 78%|███████▊  | 311720/400000 [00:36<00:09, 9726.87it/s] 78%|███████▊  | 312694/400000 [00:36<00:09, 9593.21it/s] 78%|███████▊  | 313692/400000 [00:36<00:08, 9704.16it/s] 79%|███████▊  | 314664/400000 [00:36<00:08, 9542.05it/s] 79%|███████▉  | 315641/400000 [00:36<00:08, 9607.49it/s] 79%|███████▉  | 316628/400000 [00:37<00:08, 9682.32it/s] 79%|███████▉  | 317617/400000 [00:37<00:08, 9743.05it/s] 80%|███████▉  | 318605/400000 [00:37<00:08, 9783.19it/s] 80%|███████▉  | 319584/400000 [00:37<00:08, 9499.78it/s] 80%|████████  | 320537/400000 [00:37<00:08, 9439.62it/s] 80%|████████  | 321537/400000 [00:37<00:08, 9600.40it/s] 81%|████████  | 322499/400000 [00:37<00:08, 9505.63it/s] 81%|████████  | 323458/400000 [00:37<00:08, 9529.38it/s] 81%|████████  | 324412/400000 [00:37<00:08, 9411.13it/s] 81%|████████▏ | 325396/400000 [00:37<00:07, 9534.69it/s] 82%|████████▏ | 326395/400000 [00:38<00:07, 9666.81it/s] 82%|████████▏ | 327378/400000 [00:38<00:07, 9711.88it/s] 82%|████████▏ | 328351/400000 [00:38<00:07, 9491.25it/s] 82%|████████▏ | 329302/400000 [00:38<00:07, 9252.46it/s] 83%|████████▎ | 330231/400000 [00:38<00:07, 9260.17it/s] 83%|████████▎ | 331223/400000 [00:38<00:07, 9448.59it/s] 83%|████████▎ | 332208/400000 [00:38<00:07, 9562.39it/s] 83%|████████▎ | 333180/400000 [00:38<00:06, 9608.48it/s] 84%|████████▎ | 334143/400000 [00:38<00:06, 9411.11it/s] 84%|████████▍ | 335086/400000 [00:38<00:06, 9382.64it/s] 84%|████████▍ | 336026/400000 [00:39<00:06, 9369.56it/s] 84%|████████▍ | 336964/400000 [00:39<00:06, 9323.78it/s] 84%|████████▍ | 337898/400000 [00:39<00:06, 9293.84it/s] 85%|████████▍ | 338828/400000 [00:39<00:06, 8982.46it/s] 85%|████████▍ | 339833/400000 [00:39<00:06, 9276.64it/s] 85%|████████▌ | 340828/400000 [00:39<00:06, 9468.11it/s] 85%|████████▌ | 341838/400000 [00:39<00:06, 9647.96it/s] 86%|████████▌ | 342834/400000 [00:39<00:05, 9736.48it/s] 86%|████████▌ | 343811/400000 [00:39<00:05, 9436.07it/s] 86%|████████▌ | 344759/400000 [00:40<00:05, 9420.37it/s] 86%|████████▋ | 345704/400000 [00:40<00:05, 9355.92it/s] 87%|████████▋ | 346642/400000 [00:40<00:05, 9356.07it/s] 87%|████████▋ | 347580/400000 [00:40<00:05, 9216.47it/s] 87%|████████▋ | 348504/400000 [00:40<00:05, 9112.45it/s] 87%|████████▋ | 349488/400000 [00:40<00:05, 9316.70it/s] 88%|████████▊ | 350472/400000 [00:40<00:05, 9465.04it/s] 88%|████████▊ | 351421/400000 [00:40<00:05, 9442.48it/s] 88%|████████▊ | 352418/400000 [00:40<00:04, 9594.01it/s] 88%|████████▊ | 353379/400000 [00:40<00:04, 9422.79it/s] 89%|████████▊ | 354379/400000 [00:41<00:04, 9585.37it/s] 89%|████████▉ | 355359/400000 [00:41<00:04, 9647.01it/s] 89%|████████▉ | 356326/400000 [00:41<00:04, 9574.41it/s] 89%|████████▉ | 357315/400000 [00:41<00:04, 9666.25it/s] 90%|████████▉ | 358283/400000 [00:41<00:04, 9378.91it/s] 90%|████████▉ | 359233/400000 [00:41<00:04, 9412.90it/s] 90%|█████████ | 360177/400000 [00:41<00:04, 9415.50it/s] 90%|█████████ | 361132/400000 [00:41<00:04, 9452.78it/s] 91%|█████████ | 362079/400000 [00:41<00:04, 9189.59it/s] 91%|█████████ | 363001/400000 [00:41<00:04, 8694.51it/s] 91%|█████████ | 363878/400000 [00:42<00:04, 8444.50it/s] 91%|█████████ | 364729/400000 [00:42<00:04, 8245.88it/s] 91%|█████████▏| 365560/400000 [00:42<00:04, 8128.88it/s] 92%|█████████▏| 366399/400000 [00:42<00:04, 8202.41it/s] 92%|█████████▏| 367244/400000 [00:42<00:03, 8274.48it/s] 92%|█████████▏| 368074/400000 [00:42<00:03, 8149.79it/s] 92%|█████████▏| 368938/400000 [00:42<00:03, 8290.27it/s] 92%|█████████▏| 369942/400000 [00:42<00:03, 8745.52it/s] 93%|█████████▎| 370856/400000 [00:42<00:03, 8859.13it/s] 93%|█████████▎| 371748/400000 [00:43<00:03, 8744.66it/s] 93%|█████████▎| 372665/400000 [00:43<00:03, 8866.43it/s] 93%|█████████▎| 373560/400000 [00:43<00:02, 8890.58it/s] 94%|█████████▎| 374458/400000 [00:43<00:02, 8915.86it/s] 94%|█████████▍| 375352/400000 [00:43<00:02, 8513.57it/s] 94%|█████████▍| 376209/400000 [00:43<00:02, 8345.18it/s] 94%|█████████▍| 377094/400000 [00:43<00:02, 8489.86it/s] 95%|█████████▍| 378022/400000 [00:43<00:02, 8709.51it/s] 95%|█████████▍| 378936/400000 [00:43<00:02, 8833.98it/s] 95%|█████████▍| 379895/400000 [00:43<00:02, 9046.27it/s] 95%|█████████▌| 380875/400000 [00:44<00:02, 9257.63it/s] 95%|█████████▌| 381820/400000 [00:44<00:01, 9314.09it/s] 96%|█████████▌| 382755/400000 [00:44<00:02, 8426.85it/s] 96%|█████████▌| 383616/400000 [00:44<00:01, 8260.16it/s] 96%|█████████▌| 384456/400000 [00:44<00:01, 7922.08it/s] 96%|█████████▋| 385350/400000 [00:44<00:01, 8201.18it/s] 97%|█████████▋| 386287/400000 [00:44<00:01, 8518.28it/s] 97%|█████████▋| 387272/400000 [00:44<00:01, 8877.39it/s] 97%|█████████▋| 388294/400000 [00:44<00:01, 9240.57it/s] 97%|█████████▋| 389238/400000 [00:44<00:01, 9298.32it/s] 98%|█████████▊| 390247/400000 [00:45<00:01, 9522.26it/s] 98%|█████████▊| 391253/400000 [00:45<00:00, 9675.12it/s] 98%|█████████▊| 392275/400000 [00:45<00:00, 9832.00it/s] 98%|█████████▊| 393279/400000 [00:45<00:00, 9891.84it/s] 99%|█████████▊| 394272/400000 [00:45<00:00, 9732.95it/s] 99%|█████████▉| 395290/400000 [00:45<00:00, 9861.62it/s] 99%|█████████▉| 396279/400000 [00:45<00:00, 9778.58it/s] 99%|█████████▉| 397259/400000 [00:45<00:00, 9713.86it/s]100%|█████████▉| 398241/400000 [00:45<00:00, 9744.40it/s]100%|█████████▉| 399217/400000 [00:46<00:00, 9560.09it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8677.89it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f8a0d8f2be0>, <torchtext.data.dataset.TabularDataset object at 0x7f8a0d8f2d30>, <torchtext.vocab.Vocab object at 0x7f8a0d8f2c50>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.73 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.73 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.73 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.22 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.22 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.75 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.75 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.20 file/s]2020-07-19 18:16:51.438868: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-19 18:16:51.441868: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-19 18:16:51.442080: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b1016680e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-19 18:16:51.442148: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 26%|██▌       | 2564096/9912422 [00:00<00:00, 25446238.19it/s]9920512it [00:00, 29176469.84it/s]                             
0it [00:00, ?it/s]32768it [00:00, 582398.06it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475786.82it/s]1654784it [00:00, 11755832.72it/s]                         
0it [00:00, ?it/s]8192it [00:00, 213529.91it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
