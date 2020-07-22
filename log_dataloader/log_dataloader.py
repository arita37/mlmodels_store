
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f300c891a60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f300c891a60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f30774fc510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f30774fc510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f309672aea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f309672aea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3024830950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3024830950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3024830950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 455319.01it/s] 94%|█████████▍| 9314304/9912422 [00:00<00:00, 649067.32it/s]9920512it [00:00, 46711522.35it/s]                           
0it [00:00, ?it/s]32768it [00:00, 710264.15it/s]
0it [00:00, ?it/s]  4%|▍         | 73728/1648877 [00:00<00:02, 737238.52it/s]1654784it [00:00, 11970320.51it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186640.33it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f300c74f588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f300c760828>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f3024830598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f3024830598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f3024830598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:44,  3.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:44,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:43,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:43,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:43,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:31,  4.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:31,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:31,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:31,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:31,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:22,  6.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:22,  6.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:22,  6.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:22,  6.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:22,  6.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:22,  6.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:22,  6.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:21,  6.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:16,  9.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:16,  9.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:15,  9.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:15,  9.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:15,  9.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:15,  9.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:15,  9.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:15,  9.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:15,  9.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:11, 12.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:11, 12.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:11, 12.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:10, 12.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:10, 12.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:10, 12.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:10, 12.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:10, 12.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:10, 12.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:07, 16.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:07, 16.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:07, 16.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:07, 16.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:07, 16.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:07, 16.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:07, 16.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:07, 16.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:07, 16.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:07, 16.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:00<00:05, 21.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 21.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:03, 28.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:02, 35.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 43.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 43.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 43.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 43.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 43.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 43.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 43.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 43.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 43.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 43.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 50.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 58.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 58.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 58.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 58.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 58.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 58.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 58.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 58.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 58.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 58.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 62.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 62.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 62.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:00, 62.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:00, 62.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:00, 62.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 62.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 62.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 62.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 62.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 67.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 67.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 67.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 67.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 67.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 67.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 67.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 67.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 67.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 67.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 72.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 72.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 72.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 72.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 72.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 72.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 72.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 72.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 72.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 72.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 76.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 76.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 81.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 81.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 81.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 81.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 81.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 81.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 81.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 81.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 81.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 81.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 79.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 79.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 84.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.35s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.35s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.35s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.47 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.70s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.35s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 84.47 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.70s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.70s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.46 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.70s/ url]
0 examples [00:00, ? examples/s]2020-07-22 12:09:03.586619: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-22 12:09:03.635629: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-22 12:09:03.635865: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b270936d60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-22 12:09:03.635887: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.23 examples/s]102 examples [00:00,  3.18 examples/s]197 examples [00:00,  4.54 examples/s]305 examples [00:00,  6.47 examples/s]394 examples [00:00,  9.22 examples/s]500 examples [00:00, 13.12 examples/s]606 examples [00:01, 18.64 examples/s]714 examples [00:01, 26.43 examples/s]820 examples [00:01, 37.36 examples/s]927 examples [00:01, 52.58 examples/s]1028 examples [00:01, 73.47 examples/s]1135 examples [00:01, 101.95 examples/s]1240 examples [00:01, 139.81 examples/s]1347 examples [00:01, 189.04 examples/s]1451 examples [00:01, 250.05 examples/s]1558 examples [00:01, 324.66 examples/s]1663 examples [00:02, 408.80 examples/s]1768 examples [00:02, 500.34 examples/s]1877 examples [00:02, 597.20 examples/s]1987 examples [00:02, 691.57 examples/s]2096 examples [00:02, 776.44 examples/s]2204 examples [00:02, 847.36 examples/s]2312 examples [00:02, 894.13 examples/s]2422 examples [00:02, 945.91 examples/s]2530 examples [00:02, 945.74 examples/s]2637 examples [00:02, 977.79 examples/s]2742 examples [00:03, 909.40 examples/s]2848 examples [00:03, 949.59 examples/s]2957 examples [00:03, 986.64 examples/s]3064 examples [00:03, 1008.79 examples/s]3168 examples [00:03, 1009.82 examples/s]3273 examples [00:03, 1019.10 examples/s]3381 examples [00:03, 1033.81 examples/s]3489 examples [00:03, 1044.52 examples/s]3595 examples [00:03, 1046.38 examples/s]3701 examples [00:04, 1020.95 examples/s]3808 examples [00:04, 1034.92 examples/s]3915 examples [00:04, 1043.28 examples/s]4020 examples [00:04, 1040.82 examples/s]4125 examples [00:04, 1036.40 examples/s]4232 examples [00:04, 1044.30 examples/s]4337 examples [00:04, 1036.21 examples/s]4447 examples [00:04, 1053.84 examples/s]4555 examples [00:04, 1059.73 examples/s]4664 examples [00:04, 1066.73 examples/s]4771 examples [00:05, 1051.87 examples/s]4877 examples [00:05, 1047.42 examples/s]4982 examples [00:05, 1037.78 examples/s]5086 examples [00:05, 1007.61 examples/s]5192 examples [00:05, 1021.91 examples/s]5295 examples [00:05, 1010.28 examples/s]5397 examples [00:05, 1006.40 examples/s]5498 examples [00:05, 984.12 examples/s] 5601 examples [00:05, 995.96 examples/s]5706 examples [00:05, 1008.94 examples/s]5811 examples [00:06, 1020.62 examples/s]5914 examples [00:06, 1023.26 examples/s]6019 examples [00:06, 1029.83 examples/s]6123 examples [00:06, 1000.31 examples/s]6224 examples [00:06, 996.06 examples/s] 6330 examples [00:06, 1013.47 examples/s]6435 examples [00:06, 1023.74 examples/s]6543 examples [00:06, 1037.87 examples/s]6647 examples [00:06, 1036.55 examples/s]6755 examples [00:06, 1047.45 examples/s]6864 examples [00:07, 1059.13 examples/s]6971 examples [00:07, 1030.08 examples/s]7075 examples [00:07, 1019.38 examples/s]7178 examples [00:07, 1006.87 examples/s]7285 examples [00:07, 1023.04 examples/s]7388 examples [00:07, 1000.99 examples/s]7493 examples [00:07, 1014.64 examples/s]7598 examples [00:07, 1022.51 examples/s]7705 examples [00:07, 1034.88 examples/s]7809 examples [00:08, 1030.10 examples/s]7915 examples [00:08, 1038.10 examples/s]8022 examples [00:08, 1046.01 examples/s]8129 examples [00:08, 1052.13 examples/s]8238 examples [00:08, 1060.43 examples/s]8345 examples [00:08, 1063.14 examples/s]8452 examples [00:08, 1042.13 examples/s]8557 examples [00:08, 992.49 examples/s] 8658 examples [00:08, 984.00 examples/s]8760 examples [00:08, 993.07 examples/s]8865 examples [00:09, 1007.46 examples/s]8970 examples [00:09, 1017.47 examples/s]9076 examples [00:09, 1027.81 examples/s]9179 examples [00:09, 1026.91 examples/s]9286 examples [00:09, 1037.97 examples/s]9392 examples [00:09, 1042.90 examples/s]9497 examples [00:09, 1042.28 examples/s]9602 examples [00:09, 1040.41 examples/s]9707 examples [00:09, 1016.05 examples/s]9812 examples [00:09, 1024.82 examples/s]9918 examples [00:10, 1034.58 examples/s]10022 examples [00:10, 985.66 examples/s]10128 examples [00:10, 1004.67 examples/s]10235 examples [00:10, 1021.33 examples/s]10338 examples [00:10, 1023.70 examples/s]10446 examples [00:10, 1039.91 examples/s]10551 examples [00:10, 1042.46 examples/s]10656 examples [00:10, 1044.58 examples/s]10761 examples [00:10, 1040.42 examples/s]10866 examples [00:11, 1036.51 examples/s]10972 examples [00:11, 1043.17 examples/s]11077 examples [00:11, 1043.00 examples/s]11185 examples [00:11, 1052.11 examples/s]11291 examples [00:11, 1037.23 examples/s]11397 examples [00:11, 1042.56 examples/s]11502 examples [00:11, 1041.90 examples/s]11608 examples [00:11, 1046.28 examples/s]11713 examples [00:11, 1043.47 examples/s]11818 examples [00:11, 1037.83 examples/s]11924 examples [00:12, 1043.77 examples/s]12029 examples [00:12, 1044.26 examples/s]12134 examples [00:12, 1035.75 examples/s]12240 examples [00:12, 1040.49 examples/s]12347 examples [00:12, 1048.57 examples/s]12455 examples [00:12, 1057.17 examples/s]12561 examples [00:12, 1053.34 examples/s]12667 examples [00:12, 1029.70 examples/s]12771 examples [00:12, 1030.71 examples/s]12879 examples [00:12, 1043.40 examples/s]12984 examples [00:13, 1021.76 examples/s]13088 examples [00:13, 1026.58 examples/s]13197 examples [00:13, 1042.97 examples/s]13302 examples [00:13, 1044.26 examples/s]13407 examples [00:13, 992.58 examples/s] 13514 examples [00:13, 1014.25 examples/s]13622 examples [00:13, 1031.11 examples/s]13726 examples [00:13, 1024.26 examples/s]13832 examples [00:13, 1032.67 examples/s]13939 examples [00:13, 1043.56 examples/s]14044 examples [00:14, 1043.28 examples/s]14151 examples [00:14, 1047.59 examples/s]14256 examples [00:14, 1013.71 examples/s]14360 examples [00:14, 1018.99 examples/s]14471 examples [00:14, 1043.99 examples/s]14580 examples [00:14, 1055.33 examples/s]14692 examples [00:14, 1071.54 examples/s]14800 examples [00:14, 1071.83 examples/s]14908 examples [00:14, 1073.61 examples/s]15019 examples [00:14, 1081.03 examples/s]15128 examples [00:15, 1083.10 examples/s]15240 examples [00:15, 1091.86 examples/s]15350 examples [00:15, 1089.80 examples/s]15460 examples [00:15, 1066.38 examples/s]15567 examples [00:15, 1033.37 examples/s]15673 examples [00:15, 1039.16 examples/s]15780 examples [00:15, 1046.24 examples/s]15885 examples [00:15, 1037.77 examples/s]15989 examples [00:15, 1029.70 examples/s]16093 examples [00:16, 1023.00 examples/s]16202 examples [00:16, 1040.01 examples/s]16307 examples [00:16, 1020.07 examples/s]16416 examples [00:16, 1038.83 examples/s]16521 examples [00:16, 1033.96 examples/s]16629 examples [00:16, 1044.72 examples/s]16737 examples [00:16, 1054.59 examples/s]16845 examples [00:16, 1059.85 examples/s]16952 examples [00:16, 1039.50 examples/s]17058 examples [00:16, 1043.87 examples/s]17163 examples [00:17, 1039.27 examples/s]17272 examples [00:17, 1051.35 examples/s]17378 examples [00:17, 1020.12 examples/s]17481 examples [00:17, 1012.90 examples/s]17588 examples [00:17, 1027.85 examples/s]17697 examples [00:17, 1045.38 examples/s]17806 examples [00:17, 1055.74 examples/s]17915 examples [00:17, 1063.76 examples/s]18024 examples [00:17, 1071.26 examples/s]18132 examples [00:17, 1050.09 examples/s]18240 examples [00:18, 1057.11 examples/s]18346 examples [00:18, 1032.01 examples/s]18451 examples [00:18, 1035.22 examples/s]18557 examples [00:18, 1039.86 examples/s]18662 examples [00:18, 1035.42 examples/s]18767 examples [00:18, 1037.59 examples/s]18871 examples [00:18, 1003.06 examples/s]18980 examples [00:18, 1026.76 examples/s]19086 examples [00:18, 1035.29 examples/s]19194 examples [00:18, 1047.31 examples/s]19301 examples [00:19, 1053.85 examples/s]19409 examples [00:19, 1060.66 examples/s]19516 examples [00:19, 1059.15 examples/s]19623 examples [00:19, 1049.08 examples/s]19728 examples [00:19, 1046.37 examples/s]19833 examples [00:19, 1046.14 examples/s]19942 examples [00:19, 1056.24 examples/s]20048 examples [00:19, 1007.73 examples/s]20156 examples [00:19, 1026.32 examples/s]20260 examples [00:20, 1018.90 examples/s]20368 examples [00:20, 1035.11 examples/s]20478 examples [00:20, 1051.85 examples/s]20587 examples [00:20, 1060.42 examples/s]20694 examples [00:20, 1053.71 examples/s]20800 examples [00:20, 1055.54 examples/s]20909 examples [00:20, 1063.12 examples/s]21016 examples [00:20, 1053.10 examples/s]21122 examples [00:20, 997.17 examples/s] 21223 examples [00:20, 978.58 examples/s]21333 examples [00:21, 1009.70 examples/s]21442 examples [00:21, 1032.00 examples/s]21552 examples [00:21, 1050.42 examples/s]21658 examples [00:21, 1034.94 examples/s]21762 examples [00:21, 1032.14 examples/s]21870 examples [00:21, 1044.67 examples/s]21975 examples [00:21, 1039.10 examples/s]22080 examples [00:21, 1037.22 examples/s]22187 examples [00:21, 1044.55 examples/s]22292 examples [00:21, 1044.09 examples/s]22398 examples [00:22, 1048.23 examples/s]22503 examples [00:22, 1047.99 examples/s]22611 examples [00:22, 1054.94 examples/s]22717 examples [00:22, 1051.43 examples/s]22823 examples [00:22, 1043.31 examples/s]22930 examples [00:22, 1050.02 examples/s]23036 examples [00:22, 1049.28 examples/s]23144 examples [00:22, 1057.74 examples/s]23250 examples [00:22, 1042.03 examples/s]23355 examples [00:22, 1015.96 examples/s]23457 examples [00:23, 1008.80 examples/s]23561 examples [00:23, 1017.23 examples/s]23670 examples [00:23, 1035.85 examples/s]23775 examples [00:23, 1038.48 examples/s]23881 examples [00:23, 1044.62 examples/s]23989 examples [00:23, 1053.51 examples/s]24096 examples [00:23, 1058.09 examples/s]24204 examples [00:23, 1063.16 examples/s]24311 examples [00:23, 1064.87 examples/s]24418 examples [00:24, 1032.16 examples/s]24522 examples [00:24, 1009.94 examples/s]24624 examples [00:24, 1002.52 examples/s]24726 examples [00:24, 1007.61 examples/s]24827 examples [00:24, 977.10 examples/s] 24934 examples [00:24, 1000.79 examples/s]25035 examples [00:24, 994.32 examples/s] 25135 examples [00:24, 986.07 examples/s]25235 examples [00:24, 988.40 examples/s]25341 examples [00:24, 1008.67 examples/s]25448 examples [00:25, 1023.64 examples/s]25556 examples [00:25, 1038.23 examples/s]25663 examples [00:25, 1045.92 examples/s]25768 examples [00:25, 1028.69 examples/s]25872 examples [00:25, 1013.74 examples/s]25978 examples [00:25, 1026.20 examples/s]26084 examples [00:25, 1034.12 examples/s]26189 examples [00:25, 1035.94 examples/s]26295 examples [00:25, 1041.19 examples/s]26403 examples [00:25, 1051.85 examples/s]26509 examples [00:26, 1034.48 examples/s]26616 examples [00:26, 1043.96 examples/s]26724 examples [00:26, 1053.20 examples/s]26832 examples [00:26, 1058.52 examples/s]26940 examples [00:26, 1062.32 examples/s]27047 examples [00:26, 1056.57 examples/s]27153 examples [00:26, 1045.43 examples/s]27261 examples [00:26, 1054.06 examples/s]27368 examples [00:26, 1058.67 examples/s]27475 examples [00:26, 1059.81 examples/s]27583 examples [00:27, 1063.91 examples/s]27691 examples [00:27, 1067.43 examples/s]27798 examples [00:27, 1053.19 examples/s]27904 examples [00:27, 1040.66 examples/s]28012 examples [00:27, 1050.08 examples/s]28118 examples [00:27, 1034.41 examples/s]28224 examples [00:27, 1040.13 examples/s]28333 examples [00:27, 1052.65 examples/s]28439 examples [00:27, 1018.64 examples/s]28547 examples [00:27, 1033.84 examples/s]28651 examples [00:28, 1029.26 examples/s]28759 examples [00:28, 1042.35 examples/s]28868 examples [00:28, 1053.98 examples/s]28976 examples [00:28, 1058.96 examples/s]29083 examples [00:28, 1042.43 examples/s]29190 examples [00:28, 1049.18 examples/s]29298 examples [00:28, 1055.85 examples/s]29406 examples [00:28, 1061.44 examples/s]29513 examples [00:28, 1059.80 examples/s]29620 examples [00:29, 1053.41 examples/s]29726 examples [00:29, 1055.33 examples/s]29832 examples [00:29, 1052.43 examples/s]29940 examples [00:29, 1059.50 examples/s]30046 examples [00:29, 1003.93 examples/s]30152 examples [00:29, 1018.38 examples/s]30257 examples [00:29, 1027.07 examples/s]30361 examples [00:29, 1020.03 examples/s]30464 examples [00:29, 1004.21 examples/s]30572 examples [00:29, 1023.62 examples/s]30675 examples [00:30, 1009.95 examples/s]30780 examples [00:30, 1019.69 examples/s]30883 examples [00:30, 1020.40 examples/s]30991 examples [00:30, 1036.60 examples/s]31098 examples [00:30, 1043.76 examples/s]31205 examples [00:30, 1050.70 examples/s]31311 examples [00:30, 1044.70 examples/s]31418 examples [00:30, 1051.72 examples/s]31524 examples [00:30, 1041.58 examples/s]31629 examples [00:30, 1031.81 examples/s]31737 examples [00:31, 1044.88 examples/s]31842 examples [00:31, 1037.92 examples/s]31946 examples [00:31, 1014.56 examples/s]32052 examples [00:31, 1027.10 examples/s]32155 examples [00:31, 1016.67 examples/s]32262 examples [00:31, 1030.18 examples/s]32367 examples [00:31, 1033.72 examples/s]32474 examples [00:31, 1043.83 examples/s]32580 examples [00:31, 1045.57 examples/s]32687 examples [00:31, 1050.67 examples/s]32795 examples [00:32, 1059.23 examples/s]32901 examples [00:32, 1049.78 examples/s]33008 examples [00:32, 1055.16 examples/s]33114 examples [00:32, 1052.59 examples/s]33221 examples [00:32, 1057.64 examples/s]33329 examples [00:32, 1060.97 examples/s]33436 examples [00:32, 1056.00 examples/s]33543 examples [00:32, 1057.37 examples/s]33649 examples [00:32, 1056.24 examples/s]33755 examples [00:32, 1055.68 examples/s]33861 examples [00:33, 1018.55 examples/s]33967 examples [00:33, 1028.01 examples/s]34074 examples [00:33, 1038.37 examples/s]34182 examples [00:33, 1049.39 examples/s]34288 examples [00:33, 1034.85 examples/s]34392 examples [00:33, 1006.14 examples/s]34493 examples [00:33, 992.37 examples/s] 34600 examples [00:33, 1013.53 examples/s]34702 examples [00:33, 1013.24 examples/s]34811 examples [00:34, 1034.72 examples/s]34918 examples [00:34, 1042.99 examples/s]35023 examples [00:34, 979.04 examples/s] 35129 examples [00:34, 1001.68 examples/s]35230 examples [00:34, 1003.81 examples/s]35338 examples [00:34, 1023.24 examples/s]35442 examples [00:34, 1027.80 examples/s]35549 examples [00:34, 1039.01 examples/s]35658 examples [00:34, 1051.62 examples/s]35764 examples [00:34, 973.19 examples/s] 35869 examples [00:35, 992.95 examples/s]35974 examples [00:35, 1007.42 examples/s]36080 examples [00:35, 1022.56 examples/s]36187 examples [00:35, 1035.89 examples/s]36295 examples [00:35, 1047.40 examples/s]36403 examples [00:35, 1055.29 examples/s]36509 examples [00:35, 1037.04 examples/s]36613 examples [00:35, 1030.43 examples/s]36717 examples [00:35, 1032.08 examples/s]36821 examples [00:36, 1021.19 examples/s]36924 examples [00:36, 1014.96 examples/s]37030 examples [00:36, 1026.91 examples/s]37137 examples [00:36, 1036.56 examples/s]37245 examples [00:36, 1048.78 examples/s]37350 examples [00:36, 1035.04 examples/s]37458 examples [00:36, 1045.49 examples/s]37565 examples [00:36, 1050.27 examples/s]37671 examples [00:36, 1046.86 examples/s]37779 examples [00:36, 1054.27 examples/s]37885 examples [00:37, 1048.64 examples/s]37990 examples [00:37, 1046.33 examples/s]38095 examples [00:37, 1043.16 examples/s]38200 examples [00:37, 1042.72 examples/s]38305 examples [00:37, 1014.84 examples/s]38407 examples [00:37, 1008.41 examples/s]38508 examples [00:37, 1006.80 examples/s]38609 examples [00:37, 1007.24 examples/s]38710 examples [00:37, 991.84 examples/s] 38816 examples [00:37, 1010.92 examples/s]38918 examples [00:38, 954.51 examples/s] 39018 examples [00:38, 965.20 examples/s]39120 examples [00:38, 979.52 examples/s]39221 examples [00:38, 987.80 examples/s]39326 examples [00:38, 1003.58 examples/s]39429 examples [00:38, 1010.56 examples/s]39537 examples [00:38, 1029.33 examples/s]39641 examples [00:38, 973.47 examples/s] 39742 examples [00:38, 980.99 examples/s]39842 examples [00:38, 984.32 examples/s]39949 examples [00:39, 1008.27 examples/s]40051 examples [00:39, 970.00 examples/s] 40157 examples [00:39, 993.74 examples/s]40263 examples [00:39, 1010.14 examples/s]40369 examples [00:39, 1022.11 examples/s]40476 examples [00:39, 1034.74 examples/s]40580 examples [00:39, 1020.98 examples/s]40683 examples [00:39, 949.50 examples/s] 40786 examples [00:39, 970.11 examples/s]40894 examples [00:40, 998.56 examples/s]41000 examples [00:40, 1014.82 examples/s]41107 examples [00:40, 1028.52 examples/s]41212 examples [00:40, 1033.15 examples/s]41316 examples [00:40, 1028.10 examples/s]41424 examples [00:40, 1042.54 examples/s]41530 examples [00:40, 1045.09 examples/s]41635 examples [00:40, 1044.38 examples/s]41740 examples [00:40, 996.84 examples/s] 41841 examples [00:40, 988.45 examples/s]41941 examples [00:41, 865.57 examples/s]42046 examples [00:41, 911.76 examples/s]42150 examples [00:41, 945.81 examples/s]42255 examples [00:41, 973.55 examples/s]42364 examples [00:41, 1004.78 examples/s]42466 examples [00:41, 995.49 examples/s] 42575 examples [00:41, 1019.62 examples/s]42678 examples [00:41, 1011.53 examples/s]42786 examples [00:41, 1030.03 examples/s]42892 examples [00:42, 1038.78 examples/s]42997 examples [00:42, 1030.94 examples/s]43101 examples [00:42, 1032.42 examples/s]43206 examples [00:42, 1037.20 examples/s]43310 examples [00:42, 1019.79 examples/s]43413 examples [00:42, 1021.33 examples/s]43517 examples [00:42, 1025.27 examples/s]43625 examples [00:42, 1040.29 examples/s]43730 examples [00:42, 978.06 examples/s] 43835 examples [00:42, 996.43 examples/s]43936 examples [00:43, 981.82 examples/s]44037 examples [00:43, 987.75 examples/s]44139 examples [00:43, 996.14 examples/s]44239 examples [00:43, 991.92 examples/s]44339 examples [00:43, 972.40 examples/s]44437 examples [00:43, 946.35 examples/s]44542 examples [00:43, 973.60 examples/s]44644 examples [00:43, 986.92 examples/s]44746 examples [00:43, 994.76 examples/s]44848 examples [00:43, 1002.01 examples/s]44950 examples [00:44, 1006.49 examples/s]45051 examples [00:44, 989.18 examples/s] 45156 examples [00:44, 1004.13 examples/s]45260 examples [00:44, 1014.19 examples/s]45362 examples [00:44, 1009.93 examples/s]45464 examples [00:44, 996.48 examples/s] 45571 examples [00:44, 1016.20 examples/s]45676 examples [00:44, 1024.48 examples/s]45780 examples [00:44, 1028.98 examples/s]45883 examples [00:45, 1008.26 examples/s]45989 examples [00:45, 1022.98 examples/s]46097 examples [00:45, 1037.73 examples/s]46201 examples [00:45, 1032.35 examples/s]46306 examples [00:45, 1034.92 examples/s]46410 examples [00:45, 1011.36 examples/s]46512 examples [00:45, 976.89 examples/s] 46615 examples [00:45, 989.51 examples/s]46719 examples [00:45, 1002.54 examples/s]46822 examples [00:45, 1008.23 examples/s]46929 examples [00:46, 1025.54 examples/s]47034 examples [00:46, 1030.48 examples/s]47138 examples [00:46, 1019.69 examples/s]47243 examples [00:46, 1027.59 examples/s]47346 examples [00:46, 1020.31 examples/s]47453 examples [00:46, 1033.40 examples/s]47561 examples [00:46, 1044.73 examples/s]47668 examples [00:46, 1050.56 examples/s]47774 examples [00:46, 1041.21 examples/s]47880 examples [00:46, 1044.82 examples/s]47985 examples [00:47, 986.15 examples/s] 48092 examples [00:47, 1009.16 examples/s]48200 examples [00:47, 1027.41 examples/s]48305 examples [00:47, 1033.85 examples/s]48412 examples [00:47, 1043.89 examples/s]48517 examples [00:47, 1026.50 examples/s]48623 examples [00:47, 1034.02 examples/s]48729 examples [00:47, 1041.21 examples/s]48834 examples [00:47, 1037.29 examples/s]48942 examples [00:47, 1047.12 examples/s]49050 examples [00:48, 1055.25 examples/s]49156 examples [00:48, 1033.94 examples/s]49260 examples [00:48, 1013.87 examples/s]49362 examples [00:48, 1014.22 examples/s]49470 examples [00:48, 1033.09 examples/s]49580 examples [00:48, 1050.29 examples/s]49687 examples [00:48, 1053.75 examples/s]49793 examples [00:48, 1037.26 examples/s]49897 examples [00:48, 1021.47 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6298/50000 [00:00<00:00, 62977.06 examples/s] 38%|███▊      | 18965/50000 [00:00<00:00, 74164.20 examples/s] 64%|██████▎   | 31760/50000 [00:00<00:00, 84865.65 examples/s] 89%|████████▉ | 44599/50000 [00:00<00:00, 94473.22 examples/s]                                                               0 examples [00:00, ? examples/s]86 examples [00:00, 855.60 examples/s]195 examples [00:00, 914.19 examples/s]305 examples [00:00, 961.57 examples/s]412 examples [00:00, 990.83 examples/s]521 examples [00:00, 1017.77 examples/s]631 examples [00:00, 1039.19 examples/s]741 examples [00:00, 1055.35 examples/s]851 examples [00:00, 1067.41 examples/s]958 examples [00:00, 1067.31 examples/s]1066 examples [00:01, 1070.48 examples/s]1176 examples [00:01, 1076.59 examples/s]1283 examples [00:01, 1071.19 examples/s]1391 examples [00:01, 1071.63 examples/s]1498 examples [00:01, 1064.54 examples/s]1604 examples [00:01, 1047.89 examples/s]1714 examples [00:01, 1061.07 examples/s]1820 examples [00:01, 1045.58 examples/s]1926 examples [00:01, 1047.66 examples/s]2035 examples [00:01, 1057.51 examples/s]2141 examples [00:02, 1049.97 examples/s]2247 examples [00:02, 1031.74 examples/s]2353 examples [00:02, 1038.71 examples/s]2457 examples [00:02, 999.94 examples/s] 2564 examples [00:02, 1017.00 examples/s]2667 examples [00:02, 1015.52 examples/s]2772 examples [00:02, 1023.40 examples/s]2878 examples [00:02, 1033.81 examples/s]2986 examples [00:02, 1045.39 examples/s]3091 examples [00:02, 1043.02 examples/s]3196 examples [00:03, 1029.57 examples/s]3300 examples [00:03, 1032.26 examples/s]3405 examples [00:03, 1037.27 examples/s]3512 examples [00:03, 1044.96 examples/s]3617 examples [00:03, 1040.92 examples/s]3726 examples [00:03, 1052.89 examples/s]3832 examples [00:03, 1016.34 examples/s]3940 examples [00:03, 1034.30 examples/s]4049 examples [00:03, 1047.82 examples/s]4156 examples [00:03, 1053.95 examples/s]4264 examples [00:04, 1060.62 examples/s]4373 examples [00:04, 1068.93 examples/s]4483 examples [00:04, 1075.83 examples/s]4592 examples [00:04, 1078.92 examples/s]4701 examples [00:04, 1080.61 examples/s]4810 examples [00:04, 1076.04 examples/s]4918 examples [00:04, 1070.31 examples/s]5026 examples [00:04, 1037.82 examples/s]5131 examples [00:04, 1035.12 examples/s]5239 examples [00:04, 1046.58 examples/s]5344 examples [00:05, 1046.39 examples/s]5449 examples [00:05, 1046.29 examples/s]5558 examples [00:05, 1057.81 examples/s]5666 examples [00:05, 1064.03 examples/s]5773 examples [00:05, 1065.54 examples/s]5880 examples [00:05, 1065.95 examples/s]5987 examples [00:05, 1061.80 examples/s]6094 examples [00:05, 1046.20 examples/s]6200 examples [00:05, 1048.91 examples/s]6305 examples [00:06, 1034.66 examples/s]6412 examples [00:06, 1023.22 examples/s]6515 examples [00:06, 1023.58 examples/s]6625 examples [00:06, 1044.18 examples/s]6734 examples [00:06, 1055.11 examples/s]6842 examples [00:06, 1061.93 examples/s]6949 examples [00:06, 1028.24 examples/s]7054 examples [00:06, 1034.64 examples/s]7158 examples [00:06, 1012.12 examples/s]7260 examples [00:06, 989.42 examples/s] 7365 examples [00:07, 1004.64 examples/s]7472 examples [00:07, 1022.78 examples/s]7580 examples [00:07, 1036.56 examples/s]7687 examples [00:07, 1045.77 examples/s]7794 examples [00:07, 1050.67 examples/s]7900 examples [00:07, 1048.79 examples/s]8007 examples [00:07, 1053.87 examples/s]8113 examples [00:07, 970.17 examples/s] 8221 examples [00:07, 999.68 examples/s]8327 examples [00:07, 1016.22 examples/s]8430 examples [00:08, 1017.57 examples/s]8535 examples [00:08, 1025.49 examples/s]8643 examples [00:08, 1040.56 examples/s]8748 examples [00:08, 1033.41 examples/s]8855 examples [00:08, 1042.80 examples/s]8960 examples [00:08, 1020.65 examples/s]9063 examples [00:08, 974.75 examples/s] 9169 examples [00:08, 998.62 examples/s]9277 examples [00:08, 1021.59 examples/s]9380 examples [00:09, 1009.91 examples/s]9488 examples [00:09, 1028.54 examples/s]9592 examples [00:09, 1023.51 examples/s]9697 examples [00:09, 1030.12 examples/s]9801 examples [00:09, 1023.25 examples/s]9905 examples [00:09, 1010.88 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePATC57/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePATC57/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3024830950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3024830950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3024830950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2faeb1f668>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2faebde668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3024830950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3024830950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3024830950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2faebdee80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f300c7603c8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f2f9d22af28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f2f9d22af28> 

  function with postional parmater data_info <function split_train_valid at 0x7f2f9d22af28> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f2fac0450d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f2fac0450d0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f2fac0450d0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=1ddb44f353667d39ef60233b2c9d80026b14402716bb8eb0e1346423a51087e0
  Stored in directory: /tmp/pip-ephem-wheel-cache-h1eeadzt/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<17:21:30, 13.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:23:03, 19.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:43:23, 27.4kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:00<6:06:56, 39.1kB/s].vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<4:16:14, 55.8kB/s].vector_cache/glove.6B.zip:   1%|          | 8.42M/862M [00:01<2:58:27, 79.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<2:04:23, 114kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.1M/862M [00:01<1:26:37, 162kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.8M/862M [00:01<1:00:28, 232kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<42:11, 330kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:01<29:30, 470kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:01<20:36, 669kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.1M/862M [00:01<14:29, 947kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.5M/862M [00:02<10:09, 1.34MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.4M/862M [00:02<07:10, 1.89MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<05:31, 2.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<05:45, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<07:41, 1.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<06:09, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.5M/862M [00:04<04:33, 2.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<06:40, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<05:53, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:06<04:26, 3.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:06<03:17, 4.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<15:15, 871kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<13:24, 991kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<10:03, 1.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:08<07:10, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<1:53:47, 116kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<1:20:58, 163kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<56:51, 232kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<42:47, 308kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<31:16, 420kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<22:11, 591kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<18:35, 704kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<15:41, 834kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<11:33, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:14<08:14, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<13:31, 963kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<10:47, 1.21MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:16<07:52, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<08:32, 1.52MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<07:16, 1.78MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:18<05:21, 2.41MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<06:49, 1.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<07:25, 1.74MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<05:45, 2.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.3M/862M [00:20<04:11, 3.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<09:05, 1.41MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<07:40, 1.67MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:22<05:41, 2.25MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<06:57, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<05:57, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<04:34, 2.78MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<03:20, 3.80MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<53:26, 237kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<39:59, 317kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<28:31, 444kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<20:02, 630kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<31:17, 403kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<23:12, 543kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<16:32, 760kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<14:29, 865kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:27, 1.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<08:19, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:44, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:23, 1.69MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:28, 2.27MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:46, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:59, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:26, 2.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:03, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:46, 1.82MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:21, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<03:52, 3.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<11:49:33, 17.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<8:17:43, 24.6kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<5:47:59, 35.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<4:05:42, 49.6kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<2:54:26, 69.9kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<2:02:29, 99.4kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<1:25:40, 142kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<1:05:24, 185kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<47:00, 258kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<33:08, 365kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<25:57, 464kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<20:38, 584kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<15:03, 800kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<10:37, 1.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<11:38:14, 17.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<8:09:46, 24.5kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<5:42:24, 34.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<4:01:45, 49.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<2:50:22, 69.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<1:59:18, 99.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<1:26:02, 138kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<1:02:38, 189kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<44:19, 267kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<31:06, 379kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<26:52, 439kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<20:01, 588kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<14:17, 822kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<12:41, 923kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<10:07, 1.16MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<07:20, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:46, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<07:54, 1.47MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:09, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:26, 2.60MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<39:48, 291kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<28:53, 401kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<20:25, 565kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<14:25, 798kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<25:10, 457kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<20:04, 573kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<14:33, 789kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<10:19, 1.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<12:12, 937kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:34, 1.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<06:55, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:01<05:00, 2.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<18:22, 619kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<15:17, 744kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<11:12, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<08:00, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<09:28, 1.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:50, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<05:46, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:33, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:59, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:23, 2.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<03:58, 2.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:19, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:36, 1.99MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<04:13, 2.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:26, 2.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:59, 2.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:44, 2.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:06, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:56, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:39, 2.36MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<03:22, 3.24MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<10:20, 1.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<08:24, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<06:09, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:45, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:03, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:25, 2.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:56, 2.75MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:15, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:14, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:36, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:39, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:06, 2.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:48, 2.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:05, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:50, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:39, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<03:22, 3.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<26:53, 395kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<19:56, 532kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<14:10, 746kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<12:16, 858kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<10:50, 972kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<08:03, 1.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<05:46, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:29, 1.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:03, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<05:12, 2.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:58, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:17, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<03:56, 2.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:04, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:45, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:34, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<03:19, 3.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<33:09, 310kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<24:05, 426kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<17:09, 597kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<12:05, 844kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<19:52, 513kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<14:59, 680kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<10:44, 947kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<09:46, 1.04MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:59, 1.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<06:46, 1.49MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<04:49, 2.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<21:19, 472kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<15:59, 629kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<11:24, 879kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<10:12, 979kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<09:17, 1.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:01, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<05:00, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<27:19, 363kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<20:10, 491kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<14:18, 691kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<12:12, 806kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<10:39, 925kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:58, 1.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<05:40, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<27:25, 357kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<20:13, 483kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<14:22, 678kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<12:13, 795kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<10:38, 913kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<07:57, 1.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<05:39, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<27:07, 356kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<19:59, 482kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<14:10, 678kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<12:02, 795kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<10:28, 914kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<07:46, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<05:31, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<15:17, 622kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:42, 811kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<08:22, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<08:00, 1.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:37, 1.42MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:50, 1.95MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:28, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:49, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:34, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:17, 2.82MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<23:45, 391kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<17:36, 528kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<12:29, 742kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<10:49, 853kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<09:32, 967kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:09, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<05:06, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<31:04, 295kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<22:43, 403kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<16:04, 568kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<13:15, 686kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<11:08, 816kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<08:15, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<05:51, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<51:20, 176kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<36:52, 245kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<25:57, 346kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<20:06, 445kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<16:03, 557kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<11:36, 769kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<08:16, 1.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<08:35, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:57, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<05:03, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:31, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:40, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<04:25, 1.99MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<03:11, 2.74MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<1:04:35, 135kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<46:06, 190kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<32:25, 269kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<24:32, 353kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<18:05, 479kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<12:49, 674kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<10:54, 790kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<09:27, 909kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<07:04, 1.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<05:02, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<23:47, 359kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<17:33, 486kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<12:27, 683kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<10:35, 800kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<09:09, 925kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<06:46, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<04:51, 1.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<06:50, 1.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:40, 1.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<04:09, 2.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:47, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:03, 1.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:55, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<02:51, 2.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<05:33, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:44, 1.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:52, 2.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<02:49, 2.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<08:17, 988kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<10:16, 797kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<08:20, 982kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<06:03, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<04:25, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<06:49, 1.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<08:46, 927kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<07:07, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<05:12, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<03:50, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<07:38, 1.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<08:57, 899kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<07:12, 1.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<05:15, 1.53MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<03:50, 2.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<08:29, 940kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<09:30, 840kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:32, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<05:30, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<04:00, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<10:07, 783kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<10:36, 746kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<08:18, 952kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<06:02, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<04:21, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<10:15, 765kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<10:23, 755kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<08:06, 967kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<06:00, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<04:34, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<03:33, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:51, 2.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<17:07, 454kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<15:47, 492kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<12:00, 647kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<08:45, 886kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<06:25, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<04:49, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<03:42, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<26:24, 292kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<22:15, 346kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<16:28, 468kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<11:48, 651kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<08:34, 895kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<06:14, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<04:48, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<10:06, 757kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<10:28, 729kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<08:03, 947kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:53, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:40, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:31, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:44, 2.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<07:25, 1.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<08:34, 883kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<06:47, 1.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:04, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:52, 1.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<03:00, 2.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:52, 1.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<07:11, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:44, 1.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:20, 1.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<03:19, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:37, 2.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:49, 1.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<07:25, 1.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:58, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:29, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<03:26, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:41, 2.74MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:52, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<07:24, 995kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:57, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:28, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<03:23, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:39, 2.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:50, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<07:21, 991kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:56, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:25, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<03:19, 2.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:41, 2.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<02:06, 3.42MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<10:27, 691kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<10:33, 685kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<08:08, 886kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:00, 1.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<04:28, 1.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<03:24, 2.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:16, 1.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<07:34, 945kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:04, 1.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:32, 1.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<03:27, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:41, 2.64MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:56, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:03, 1.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:42, 1.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:17, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:16, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:31, 2.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<06:00, 1.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:04, 992kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:34, 1.26MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:11, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:11, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<02:25, 2.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<02:04, 3.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<1:21:16, 85.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<59:43, 116kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<42:28, 164kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<29:58, 231kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<21:07, 327kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<14:58, 460kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<14:39, 470kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<12:51, 535kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<09:38, 714kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<06:58, 983kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<05:06, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<03:47, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<08:31, 800kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<08:31, 799kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:34, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:51, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:34, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:42, 2.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<42:45, 158kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<32:18, 209kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<23:10, 291kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<16:25, 409kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<11:38, 575kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:16<08:21, 800kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<20:06, 332kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<16:15, 411kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<11:49, 564kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<08:25, 789kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<06:09, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<04:27, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<11:57, 553kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<12:24, 532kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<09:29, 695kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:55, 952kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<05:00, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<05:28, 1.19MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<05:43, 1.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:26, 1.47MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:16, 1.98MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<02:26, 2.65MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<10:28, 617kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<10:30, 616kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<08:12, 788kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:58, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<04:17, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:44, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:35, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<04:13, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:06, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:55, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:36, 1.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<04:31, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:21, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:28, 2.54MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<06:52, 911kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<07:10, 873kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<05:33, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:03, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:55, 2.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<08:35, 721kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<08:21, 740kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<06:26, 961kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:37, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:36, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<05:14, 1.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:11, 1.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:03, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:38, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:23, 1.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:30, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:34, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:26, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<04:08, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<03:20, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:26, 2.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:36, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:14, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<03:19, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:25, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:48, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:10, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:17, 1.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:24, 2.41MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:57, 1.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:08, 1.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:11, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:17, 2.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:57, 1.44MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:06, 1.39MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:25, 1.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:45<02:31, 2.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:00, 2.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:38, 3.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<07:16, 774kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<07:37, 739kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<05:57, 945kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<04:23, 1.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:16, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:30, 2.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<05:04, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<05:49, 956kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<04:33, 1.22MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:26, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:33, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:56, 2.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<06:49, 805kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<06:51, 802kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<05:14, 1.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<03:52, 1.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:53, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:12, 2.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<09:31, 571kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<08:42, 623kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<06:35, 823kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<04:46, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<03:28, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:41, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:05, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:26, 986kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:16, 1.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<03:10, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:23, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:28, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:17, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:24, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:34, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<01:57, 2.68MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:12, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<05:26, 960kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:41, 1.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:29, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:36, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:58, 2.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<05:20, 965kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<05:34, 924kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:18, 1.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<03:11, 1.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<02:22, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:39, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<05:43, 889kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<04:51, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:36, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:40, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<02:01, 2.49MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<06:20, 792kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<06:11, 810kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:44, 1.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<03:29, 1.43MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<02:34, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:45, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<06:05, 812kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:58, 994kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:43, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<02:45, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<02:04, 2.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:12, 936kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:15, 928kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<04:05, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<03:00, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<02:14, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<03:26, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<06:05, 789kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<05:04, 946kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<03:44, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<02:45, 1.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<02:04, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<06:04, 781kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:48, 817kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:23, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:13, 1.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<02:23, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:30, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:54, 1.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:07, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:18, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:43, 2.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:22, 3.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<05:52, 785kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<05:37, 819kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<04:17, 1.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<03:08, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<02:19, 1.96MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:38, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:02, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:11, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:21, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:48, 2.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<01:22, 3.25MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<06:08, 728kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<05:39, 788kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:15, 1.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<03:07, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<02:18, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<01:43, 2.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:48, 915kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<06:05, 722kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<04:57, 885kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:39, 1.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<02:40, 1.63MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<01:58, 2.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<52:50, 81.9kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<38:12, 113kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<27:00, 160kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<18:55, 227kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<13:16, 321kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<12:30, 340kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<11:05, 384kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<08:21, 509kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<05:57, 710kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<04:18, 981kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<03:06, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<06:26, 650kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<06:47, 616kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<05:17, 790kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:51, 1.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<02:46, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<03:30, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:30, 915kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:39, 1.12MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:41, 1.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:03, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:29, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<05:05, 796kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:12, 963kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:07, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:17, 1.76MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:44, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:42, 1.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:12, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:33, 1.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:54, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:26, 2.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<03:00, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:30, 868kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:47, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:52, 1.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:06, 1.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:35, 2.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:27, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:50, 1.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:15, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:39<01:40, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:15, 2.99MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<04:12, 899kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<05:01, 752kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<04:02, 933kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:56, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<02:08, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:53, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<04:03, 912kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:21, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:28, 1.49MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:48, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:41, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<03:51, 941kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:10, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:21, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:43, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:42, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:38, 978kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:01, 1.18MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:13, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:37, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:13, 2.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<08:16, 423kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<07:30, 466kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<05:40, 616kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<04:03, 856kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<02:54, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:36, 950kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:12, 814kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<03:22, 1.01MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:27, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:47, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:17, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:15, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:37, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:00, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:27, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:05, 3.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<04:40, 705kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<04:53, 674kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<03:49, 860kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:45, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:59, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:56, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<03:38, 885kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:57, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:09, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:34, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:44, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<03:28, 908kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:49, 1.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<02:03, 1.52MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:29, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:07, 2.73MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<18:14, 169kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<14:16, 216kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<10:21, 297kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<07:19, 418kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<05:08, 589kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<04:59, 604kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<04:58, 606kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<03:51, 780kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<02:47, 1.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:00, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:45, 1.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:22, 874kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<02:40, 1.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:57, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:26, 2.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:05, 2.65MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:25, 1.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<03:06, 924kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<02:32, 1.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:51, 1.54MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:21, 2.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:26, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<03:04, 912kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:30, 1.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:49, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<01:19, 2.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:14, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:54, 940kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:22, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:45, 1.55MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:16, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:15, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:46, 963kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:14, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:38, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:11, 2.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:29, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:01, 862kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:24, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:45, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:16, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:21, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:46, 912kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:11, 1.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:35, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<01:10, 2.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<00:52, 2.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:45, 518kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:25, 557kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<03:21, 730kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<02:24, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<01:42, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:30, 683kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:31, 680kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:42, 880kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:57, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<01:24, 1.67MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:05, 753kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:06, 747kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<02:25, 957kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:45, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<01:15, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:27, 652kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:20, 674kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:33, 877kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:24<01:50, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<01:18, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:31, 621kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:40, 814kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:56, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:23, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:38, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:56, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:31, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:08, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:49, 2.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:25, 1.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:42, 1.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:21, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:01, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:44, 2.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:38, 1.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:50, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:27, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:02, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:45, 2.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:10, 877kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:06, 905kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:36, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:10, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:50, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:56, 948kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:55, 957kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:27, 1.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<01:02, 1.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:45, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:47, 636kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:27, 721kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:49, 968kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:17, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:55, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:12, 769kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:11, 777kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:40, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:11, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:51, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:19, 705kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:04, 786kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:32, 1.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:06, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:05, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:08, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:52, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:37, 2.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:00, 1.49MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:03, 1.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:48, 1.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:34, 2.53MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:55, 1.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:57, 1.48MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:44, 1.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:31, 2.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:06, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:06, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:50, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:36, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:41, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:49, 1.57MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:39, 1.95MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:28, 2.60MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:22, 3.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:53, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:28, 822kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:15, 966kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:56, 1.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:41, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:30, 2.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:23, 2.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:17, 885kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:18, 876kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:59, 1.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:43, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:32, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:23, 2.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:48, 598kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:37, 665kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:12, 882kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:52, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:37, 1.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:54, 1.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:15, 806kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:01, 978kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:45, 1.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:32, 1.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:00<00:23, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<05:51, 161kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<04:22, 215kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<03:06, 300kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<02:09, 423kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<01:30, 595kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<01:03, 828kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<09:31, 91.7kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<06:54, 126kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<04:50, 178kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<03:21, 253kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<02:20, 356kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<01:37, 502kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<01:37, 493kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:42, 472kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:17, 620kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:54, 861kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:40, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:29, 1.57MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:20, 2.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<02:47, 263kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<02:10, 337kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<01:33, 465kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<01:04, 653kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:45, 904kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:32, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<06:02, 110kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<04:38, 144kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<03:19, 199kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<02:19, 280kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<01:36, 396kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<01:06, 556kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:46, 778kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<29:12, 20.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<20:32, 29.0kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<14:12, 41.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<09:35, 58.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<06:26, 84.0kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<04:44, 111kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<03:36, 146kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<02:35, 202kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<01:46, 286kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<01:11, 404kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:56, 489kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:56, 489kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:42, 634kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:29, 878kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:21, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:14, 1.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:25, 910kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:24, 946kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:18, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:12, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:09, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:06, 2.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:52, 367kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:47, 408kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:34, 545kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:23, 761kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:16, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:11, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:13, 1.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:13, 1.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:10, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:04, 2.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:11, 930kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:13, 802kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:10, 994kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:07, 1.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:04, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:22, 310kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:19, 357kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:13, 480kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:08, 668kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:03, 934kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 813kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:03, 739kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:02, 940kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.29MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<78:20:44,  1.42it/s]  0%|          | 781/400000 [00:00<54:44:21,  2.03it/s]  0%|          | 1590/400000 [00:00<38:14:38,  2.89it/s]  1%|          | 2427/400000 [00:01<26:43:06,  4.13it/s]  1%|          | 3276/400000 [00:01<18:40:00,  5.90it/s]  1%|          | 4080/400000 [00:01<13:02:40,  8.43it/s]  1%|          | 4849/400000 [00:01<9:07:03, 12.04it/s]   1%|▏         | 5615/400000 [00:01<6:22:27, 17.19it/s]  2%|▏         | 6437/400000 [00:01<4:27:24, 24.53it/s]  2%|▏         | 7279/400000 [00:01<3:07:00, 35.00it/s]  2%|▏         | 8109/400000 [00:01<2:10:52, 49.91it/s]  2%|▏         | 8906/400000 [00:01<1:31:40, 71.11it/s]  2%|▏         | 9731/400000 [00:01<1:04:16, 101.21it/s]  3%|▎         | 10566/400000 [00:02<45:07, 143.83it/s]   3%|▎         | 11415/400000 [00:02<31:44, 204.00it/s]  3%|▎         | 12252/400000 [00:02<22:24, 288.41it/s]  3%|▎         | 13096/400000 [00:02<15:52, 406.06it/s]  3%|▎         | 13929/400000 [00:02<11:20, 567.73it/s]  4%|▎         | 14778/400000 [00:02<08:08, 788.44it/s]  4%|▍         | 15620/400000 [00:02<05:54, 1082.85it/s]  4%|▍         | 16454/400000 [00:02<04:25, 1442.24it/s]  4%|▍         | 17291/400000 [00:02<03:19, 1918.56it/s]  5%|▍         | 18136/400000 [00:02<02:32, 2497.65it/s]  5%|▍         | 18987/400000 [00:03<02:00, 3169.15it/s]  5%|▍         | 19830/400000 [00:03<01:37, 3898.59it/s]  5%|▌         | 20658/400000 [00:03<01:22, 4611.66it/s]  5%|▌         | 21502/400000 [00:03<01:10, 5337.21it/s]  6%|▌         | 22330/400000 [00:03<01:03, 5945.46it/s]  6%|▌         | 23186/400000 [00:03<00:57, 6543.62it/s]  6%|▌         | 24038/400000 [00:03<00:53, 7032.87it/s]  6%|▌         | 24878/400000 [00:03<00:50, 7392.63it/s]  6%|▋         | 25717/400000 [00:03<00:48, 7655.74it/s]  7%|▋         | 26555/400000 [00:03<00:49, 7572.62it/s]  7%|▋         | 27379/400000 [00:04<00:48, 7758.60it/s]  7%|▋         | 28192/400000 [00:04<00:47, 7840.38it/s]  7%|▋         | 29002/400000 [00:04<00:48, 7691.63it/s]  7%|▋         | 29790/400000 [00:04<00:47, 7722.62it/s]  8%|▊         | 30602/400000 [00:04<00:47, 7836.57it/s]  8%|▊         | 31401/400000 [00:04<00:46, 7881.52it/s]  8%|▊         | 32257/400000 [00:04<00:45, 8071.98it/s]  8%|▊         | 33077/400000 [00:04<00:45, 8109.76it/s]  8%|▊         | 33941/400000 [00:04<00:44, 8260.90it/s]  9%|▊         | 34771/400000 [00:04<00:45, 8054.39it/s]  9%|▉         | 35581/400000 [00:05<00:45, 7999.09it/s]  9%|▉         | 36447/400000 [00:05<00:44, 8185.45it/s]  9%|▉         | 37269/400000 [00:05<00:44, 8167.64it/s] 10%|▉         | 38093/400000 [00:05<00:44, 8186.71it/s] 10%|▉         | 38936/400000 [00:05<00:43, 8209.26it/s] 10%|▉         | 39764/400000 [00:05<00:43, 8230.09it/s] 10%|█         | 40607/400000 [00:05<00:43, 8288.29it/s] 10%|█         | 41437/400000 [00:05<00:43, 8265.62it/s] 11%|█         | 42294/400000 [00:05<00:42, 8354.06it/s] 11%|█         | 43153/400000 [00:05<00:42, 8420.95it/s] 11%|█         | 44001/400000 [00:06<00:42, 8436.35it/s] 11%|█         | 44851/400000 [00:06<00:42, 8454.15it/s] 11%|█▏        | 45697/400000 [00:06<00:42, 8328.27it/s] 12%|█▏        | 46531/400000 [00:06<00:42, 8308.56it/s] 12%|█▏        | 47367/400000 [00:06<00:42, 8322.75it/s] 12%|█▏        | 48200/400000 [00:06<00:42, 8309.27it/s] 12%|█▏        | 49032/400000 [00:06<00:42, 8287.16it/s] 12%|█▏        | 49861/400000 [00:06<00:42, 8271.69it/s] 13%|█▎        | 50719/400000 [00:06<00:41, 8361.11it/s] 13%|█▎        | 51556/400000 [00:07<00:41, 8297.21it/s] 13%|█▎        | 52416/400000 [00:07<00:41, 8383.50it/s] 13%|█▎        | 53262/400000 [00:07<00:41, 8405.44it/s] 14%|█▎        | 54115/400000 [00:07<00:40, 8439.41it/s] 14%|█▎        | 54960/400000 [00:07<00:41, 8358.30it/s] 14%|█▍        | 55815/400000 [00:07<00:40, 8412.41it/s] 14%|█▍        | 56657/400000 [00:07<00:41, 8369.46it/s] 14%|█▍        | 57495/400000 [00:07<00:40, 8371.24it/s] 15%|█▍        | 58341/400000 [00:07<00:40, 8396.14it/s] 15%|█▍        | 59203/400000 [00:07<00:40, 8461.94it/s] 15%|█▌        | 60070/400000 [00:08<00:39, 8522.27it/s] 15%|█▌        | 60923/400000 [00:08<00:40, 8415.22it/s] 15%|█▌        | 61775/400000 [00:08<00:40, 8445.51it/s] 16%|█▌        | 62628/400000 [00:08<00:39, 8469.69it/s] 16%|█▌        | 63476/400000 [00:08<00:39, 8456.92it/s] 16%|█▌        | 64339/400000 [00:08<00:39, 8507.62it/s] 16%|█▋        | 65190/400000 [00:08<00:39, 8486.62it/s] 17%|█▋        | 66039/400000 [00:08<00:39, 8402.75it/s] 17%|█▋        | 66880/400000 [00:08<00:40, 8161.72it/s] 17%|█▋        | 67741/400000 [00:08<00:40, 8290.46it/s] 17%|█▋        | 68599/400000 [00:09<00:39, 8374.27it/s] 17%|█▋        | 69455/400000 [00:09<00:39, 8428.97it/s] 18%|█▊        | 70313/400000 [00:09<00:38, 8471.29it/s] 18%|█▊        | 71161/400000 [00:09<00:38, 8444.00it/s] 18%|█▊        | 72006/400000 [00:09<00:38, 8444.77it/s] 18%|█▊        | 72863/400000 [00:09<00:38, 8480.79it/s] 18%|█▊        | 73732/400000 [00:09<00:38, 8540.05it/s] 19%|█▊        | 74587/400000 [00:09<00:38, 8525.97it/s] 19%|█▉        | 75440/400000 [00:09<00:38, 8434.59it/s] 19%|█▉        | 76292/400000 [00:09<00:38, 8459.49it/s] 19%|█▉        | 77139/400000 [00:10<00:38, 8459.31it/s] 19%|█▉        | 77986/400000 [00:10<00:38, 8435.85it/s] 20%|█▉        | 78830/400000 [00:10<00:38, 8416.64it/s] 20%|█▉        | 79672/400000 [00:10<00:38, 8415.98it/s] 20%|██        | 80532/400000 [00:10<00:37, 8469.59it/s] 20%|██        | 81388/400000 [00:10<00:37, 8495.09it/s] 21%|██        | 82238/400000 [00:10<00:37, 8484.34it/s] 21%|██        | 83092/400000 [00:10<00:37, 8500.35it/s] 21%|██        | 83951/400000 [00:10<00:37, 8526.18it/s] 21%|██        | 84804/400000 [00:10<00:37, 8501.41it/s] 21%|██▏       | 85655/400000 [00:11<00:38, 8117.92it/s] 22%|██▏       | 86506/400000 [00:11<00:38, 8229.92it/s] 22%|██▏       | 87352/400000 [00:11<00:37, 8295.67it/s] 22%|██▏       | 88206/400000 [00:11<00:37, 8367.39it/s] 22%|██▏       | 89052/400000 [00:11<00:37, 8393.74it/s] 22%|██▏       | 89922/400000 [00:11<00:36, 8482.61it/s] 23%|██▎       | 90787/400000 [00:11<00:36, 8531.64it/s] 23%|██▎       | 91641/400000 [00:11<00:36, 8346.61it/s] 23%|██▎       | 92499/400000 [00:11<00:36, 8415.22it/s] 23%|██▎       | 93342/400000 [00:11<00:36, 8335.44it/s] 24%|██▎       | 94177/400000 [00:12<00:37, 8120.27it/s] 24%|██▍       | 95027/400000 [00:12<00:37, 8229.07it/s] 24%|██▍       | 95867/400000 [00:12<00:36, 8277.11it/s] 24%|██▍       | 96708/400000 [00:12<00:36, 8314.62it/s] 24%|██▍       | 97541/400000 [00:12<00:36, 8318.73it/s] 25%|██▍       | 98403/400000 [00:12<00:35, 8403.98it/s] 25%|██▍       | 99245/400000 [00:12<00:35, 8408.31it/s] 25%|██▌       | 100090/400000 [00:12<00:35, 8418.62it/s] 25%|██▌       | 100933/400000 [00:12<00:36, 8254.18it/s] 25%|██▌       | 101760/400000 [00:12<00:36, 8243.68it/s] 26%|██▌       | 102586/400000 [00:13<00:36, 8177.28it/s] 26%|██▌       | 103437/400000 [00:13<00:35, 8274.32it/s] 26%|██▌       | 104291/400000 [00:13<00:35, 8350.77it/s] 26%|██▋       | 105127/400000 [00:13<00:35, 8341.48it/s] 26%|██▋       | 105963/400000 [00:13<00:35, 8344.84it/s] 27%|██▋       | 106817/400000 [00:13<00:34, 8400.68it/s] 27%|██▋       | 107665/400000 [00:13<00:34, 8424.02it/s] 27%|██▋       | 108522/400000 [00:13<00:34, 8466.66it/s] 27%|██▋       | 109369/400000 [00:13<00:34, 8374.92it/s] 28%|██▊       | 110207/400000 [00:13<00:35, 8273.70it/s] 28%|██▊       | 111036/400000 [00:14<00:34, 8278.55it/s] 28%|██▊       | 111887/400000 [00:14<00:34, 8345.70it/s] 28%|██▊       | 112722/400000 [00:14<00:35, 8163.51it/s] 28%|██▊       | 113561/400000 [00:14<00:34, 8228.15it/s] 29%|██▊       | 114396/400000 [00:14<00:34, 8263.14it/s] 29%|██▉       | 115246/400000 [00:14<00:34, 8331.09it/s] 29%|██▉       | 116080/400000 [00:14<00:34, 8115.89it/s] 29%|██▉       | 116931/400000 [00:14<00:34, 8229.11it/s] 29%|██▉       | 117776/400000 [00:14<00:34, 8291.29it/s] 30%|██▉       | 118607/400000 [00:15<00:33, 8290.50it/s] 30%|██▉       | 119460/400000 [00:15<00:33, 8359.75it/s] 30%|███       | 120314/400000 [00:15<00:33, 8411.26it/s] 30%|███       | 121166/400000 [00:15<00:33, 8442.97it/s] 31%|███       | 122021/400000 [00:15<00:32, 8470.69it/s] 31%|███       | 122869/400000 [00:15<00:32, 8404.98it/s] 31%|███       | 123727/400000 [00:15<00:32, 8456.12it/s] 31%|███       | 124579/400000 [00:15<00:32, 8474.96it/s] 31%|███▏      | 125448/400000 [00:15<00:32, 8538.21it/s] 32%|███▏      | 126304/400000 [00:15<00:32, 8543.09it/s] 32%|███▏      | 127159/400000 [00:16<00:32, 8496.62it/s] 32%|███▏      | 128009/400000 [00:16<00:32, 8456.05it/s] 32%|███▏      | 128867/400000 [00:16<00:31, 8492.18it/s] 32%|███▏      | 129723/400000 [00:16<00:31, 8510.95it/s] 33%|███▎      | 130575/400000 [00:16<00:31, 8511.95it/s] 33%|███▎      | 131427/400000 [00:16<00:31, 8457.35it/s] 33%|███▎      | 132283/400000 [00:16<00:31, 8486.95it/s] 33%|███▎      | 133136/400000 [00:16<00:31, 8498.73it/s] 33%|███▎      | 133990/400000 [00:16<00:31, 8502.07it/s] 34%|███▎      | 134841/400000 [00:16<00:31, 8461.34it/s] 34%|███▍      | 135688/400000 [00:17<00:31, 8361.09it/s] 34%|███▍      | 136525/400000 [00:17<00:31, 8345.47it/s] 34%|███▍      | 137372/400000 [00:17<00:31, 8381.71it/s] 35%|███▍      | 138223/400000 [00:17<00:31, 8419.71it/s] 35%|███▍      | 139066/400000 [00:17<00:31, 8370.73it/s] 35%|███▍      | 139904/400000 [00:17<00:31, 8236.29it/s] 35%|███▌      | 140752/400000 [00:17<00:31, 8306.49it/s] 35%|███▌      | 141600/400000 [00:17<00:30, 8355.61it/s] 36%|███▌      | 142437/400000 [00:17<00:30, 8348.34it/s] 36%|███▌      | 143273/400000 [00:17<00:30, 8328.92it/s] 36%|███▌      | 144113/400000 [00:18<00:30, 8342.03it/s] 36%|███▌      | 144948/400000 [00:18<00:30, 8281.64it/s] 36%|███▋      | 145815/400000 [00:18<00:30, 8392.25it/s] 37%|███▋      | 146665/400000 [00:18<00:30, 8422.48it/s] 37%|███▋      | 147520/400000 [00:18<00:29, 8458.62it/s] 37%|███▋      | 148381/400000 [00:18<00:29, 8500.95it/s] 37%|███▋      | 149239/400000 [00:18<00:29, 8524.37it/s] 38%|███▊      | 150101/400000 [00:18<00:29, 8551.62it/s] 38%|███▊      | 150957/400000 [00:18<00:29, 8537.87it/s] 38%|███▊      | 151817/400000 [00:18<00:29, 8553.53it/s] 38%|███▊      | 152673/400000 [00:19<00:29, 8291.14it/s] 38%|███▊      | 153526/400000 [00:19<00:29, 8360.14it/s] 39%|███▊      | 154371/400000 [00:19<00:29, 8386.71it/s] 39%|███▉      | 155220/400000 [00:19<00:29, 8417.22it/s] 39%|███▉      | 156068/400000 [00:19<00:28, 8435.30it/s] 39%|███▉      | 156913/400000 [00:19<00:28, 8429.41it/s] 39%|███▉      | 157773/400000 [00:19<00:28, 8478.61it/s] 40%|███▉      | 158626/400000 [00:19<00:28, 8493.04it/s] 40%|███▉      | 159495/400000 [00:19<00:28, 8549.48it/s] 40%|████      | 160351/400000 [00:19<00:28, 8515.30it/s] 40%|████      | 161203/400000 [00:20<00:28, 8279.22it/s] 41%|████      | 162048/400000 [00:20<00:28, 8329.28it/s] 41%|████      | 162899/400000 [00:20<00:28, 8382.17it/s] 41%|████      | 163740/400000 [00:20<00:28, 8390.31it/s] 41%|████      | 164583/400000 [00:20<00:28, 8401.75it/s] 41%|████▏     | 165424/400000 [00:20<00:27, 8390.07it/s] 42%|████▏     | 166272/400000 [00:20<00:27, 8416.82it/s] 42%|████▏     | 167114/400000 [00:20<00:27, 8416.06it/s] 42%|████▏     | 167965/400000 [00:20<00:27, 8443.71it/s] 42%|████▏     | 168817/400000 [00:20<00:27, 8463.65it/s] 42%|████▏     | 169664/400000 [00:21<00:27, 8396.13it/s] 43%|████▎     | 170504/400000 [00:21<00:27, 8372.34it/s] 43%|████▎     | 171352/400000 [00:21<00:27, 8403.42it/s] 43%|████▎     | 172195/400000 [00:21<00:27, 8361.98it/s] 43%|████▎     | 173032/400000 [00:21<00:27, 8309.19it/s] 43%|████▎     | 173864/400000 [00:21<00:27, 8214.91it/s] 44%|████▎     | 174712/400000 [00:21<00:27, 8290.51it/s] 44%|████▍     | 175565/400000 [00:21<00:26, 8359.00it/s] 44%|████▍     | 176402/400000 [00:21<00:26, 8356.88it/s] 44%|████▍     | 177246/400000 [00:21<00:26, 8379.04it/s] 45%|████▍     | 178085/400000 [00:22<00:26, 8352.73it/s] 45%|████▍     | 178933/400000 [00:22<00:26, 8390.07it/s] 45%|████▍     | 179776/400000 [00:22<00:26, 8400.99it/s] 45%|████▌     | 180617/400000 [00:22<00:26, 8370.39it/s] 45%|████▌     | 181463/400000 [00:22<00:26, 8395.58it/s] 46%|████▌     | 182303/400000 [00:22<00:26, 8362.28it/s] 46%|████▌     | 183152/400000 [00:22<00:25, 8397.32it/s] 46%|████▌     | 184006/400000 [00:22<00:25, 8439.30it/s] 46%|████▌     | 184851/400000 [00:22<00:25, 8281.46it/s] 46%|████▋     | 185694/400000 [00:22<00:25, 8324.83it/s] 47%|████▋     | 186528/400000 [00:23<00:26, 8060.87it/s] 47%|████▋     | 187376/400000 [00:23<00:25, 8181.92it/s] 47%|████▋     | 188226/400000 [00:23<00:25, 8272.83it/s] 47%|████▋     | 189088/400000 [00:23<00:25, 8372.16it/s] 47%|████▋     | 189944/400000 [00:23<00:24, 8426.38it/s] 48%|████▊     | 190788/400000 [00:23<00:24, 8402.77it/s] 48%|████▊     | 191638/400000 [00:23<00:24, 8429.32it/s] 48%|████▊     | 192502/400000 [00:23<00:24, 8490.77it/s] 48%|████▊     | 193375/400000 [00:23<00:24, 8559.59it/s] 49%|████▊     | 194232/400000 [00:23<00:24, 8474.04it/s] 49%|████▉     | 195080/400000 [00:24<00:24, 8321.90it/s] 49%|████▉     | 195922/400000 [00:24<00:24, 8349.46it/s] 49%|████▉     | 196758/400000 [00:24<00:24, 8144.70it/s] 49%|████▉     | 197602/400000 [00:24<00:24, 8226.59it/s] 50%|████▉     | 198462/400000 [00:24<00:24, 8332.85it/s] 50%|████▉     | 199307/400000 [00:24<00:23, 8366.89it/s] 50%|█████     | 200162/400000 [00:24<00:23, 8418.35it/s] 50%|█████     | 201005/400000 [00:24<00:23, 8377.31it/s] 50%|█████     | 201844/400000 [00:24<00:23, 8306.98it/s] 51%|█████     | 202689/400000 [00:25<00:23, 8346.75it/s] 51%|█████     | 203529/400000 [00:25<00:23, 8360.95it/s] 51%|█████     | 204382/400000 [00:25<00:23, 8409.54it/s] 51%|█████▏    | 205231/400000 [00:25<00:23, 8431.31it/s] 52%|█████▏    | 206075/400000 [00:25<00:23, 8413.48it/s] 52%|█████▏    | 206928/400000 [00:25<00:22, 8446.19it/s] 52%|█████▏    | 207773/400000 [00:25<00:22, 8434.21it/s] 52%|█████▏    | 208627/400000 [00:25<00:22, 8465.50it/s] 52%|█████▏    | 209479/400000 [00:25<00:22, 8480.33it/s] 53%|█████▎    | 210336/400000 [00:25<00:22, 8504.67it/s] 53%|█████▎    | 211187/400000 [00:26<00:22, 8261.84it/s] 53%|█████▎    | 212015/400000 [00:26<00:22, 8220.01it/s] 53%|█████▎    | 212866/400000 [00:26<00:22, 8302.46it/s] 53%|█████▎    | 213721/400000 [00:26<00:22, 8373.79it/s] 54%|█████▎    | 214576/400000 [00:26<00:22, 8423.88it/s] 54%|█████▍    | 215430/400000 [00:26<00:21, 8418.44it/s] 54%|█████▍    | 216273/400000 [00:26<00:22, 8300.13it/s] 54%|█████▍    | 217104/400000 [00:26<00:22, 8252.23it/s] 54%|█████▍    | 217948/400000 [00:26<00:21, 8306.78it/s] 55%|█████▍    | 218804/400000 [00:26<00:21, 8378.30it/s] 55%|█████▍    | 219643/400000 [00:27<00:22, 8138.01it/s] 55%|█████▌    | 220489/400000 [00:27<00:21, 8230.86it/s] 55%|█████▌    | 221334/400000 [00:27<00:21, 8295.11it/s] 56%|█████▌    | 222165/400000 [00:27<00:21, 8217.37it/s] 56%|█████▌    | 223011/400000 [00:27<00:21, 8287.69it/s] 56%|█████▌    | 223841/400000 [00:27<00:21, 8278.18it/s] 56%|█████▌    | 224670/400000 [00:27<00:21, 8269.51it/s] 56%|█████▋    | 225516/400000 [00:27<00:20, 8324.06it/s] 57%|█████▋    | 226360/400000 [00:27<00:20, 8356.58it/s] 57%|█████▋    | 227220/400000 [00:27<00:20, 8427.20it/s] 57%|█████▋    | 228064/400000 [00:28<00:20, 8409.79it/s] 57%|█████▋    | 228906/400000 [00:28<00:20, 8388.11it/s] 57%|█████▋    | 229746/400000 [00:28<00:20, 8338.50it/s] 58%|█████▊    | 230581/400000 [00:28<00:20, 8305.35it/s] 58%|█████▊    | 231433/400000 [00:28<00:20, 8367.50it/s] 58%|█████▊    | 232286/400000 [00:28<00:19, 8415.31it/s] 58%|█████▊    | 233128/400000 [00:28<00:19, 8405.80it/s] 58%|█████▊    | 233969/400000 [00:28<00:19, 8324.00it/s] 59%|█████▊    | 234807/400000 [00:28<00:19, 8340.14it/s] 59%|█████▉    | 235663/400000 [00:28<00:19, 8404.57it/s] 59%|█████▉    | 236504/400000 [00:29<00:19, 8191.48it/s] 59%|█████▉    | 237329/400000 [00:29<00:19, 8208.09it/s] 60%|█████▉    | 238154/400000 [00:29<00:19, 8217.77it/s] 60%|█████▉    | 239008/400000 [00:29<00:19, 8310.88it/s] 60%|█████▉    | 239846/400000 [00:29<00:19, 8329.16it/s] 60%|██████    | 240706/400000 [00:29<00:18, 8408.18it/s] 60%|██████    | 241548/400000 [00:29<00:18, 8385.62it/s] 61%|██████    | 242387/400000 [00:29<00:18, 8332.12it/s] 61%|██████    | 243243/400000 [00:29<00:18, 8398.79it/s] 61%|██████    | 244084/400000 [00:29<00:18, 8330.44it/s] 61%|██████    | 244918/400000 [00:30<00:19, 8022.63it/s] 61%|██████▏   | 245748/400000 [00:30<00:19, 8102.58it/s] 62%|██████▏   | 246588/400000 [00:30<00:18, 8186.98it/s] 62%|██████▏   | 247440/400000 [00:30<00:18, 8281.52it/s] 62%|██████▏   | 248301/400000 [00:30<00:18, 8376.00it/s] 62%|██████▏   | 249140/400000 [00:30<00:18, 8327.33it/s] 63%|██████▎   | 250005/400000 [00:30<00:17, 8421.31it/s] 63%|██████▎   | 250864/400000 [00:30<00:17, 8469.25it/s] 63%|██████▎   | 251734/400000 [00:30<00:17, 8535.25it/s] 63%|██████▎   | 252589/400000 [00:30<00:17, 8523.52it/s] 63%|██████▎   | 253452/400000 [00:31<00:17, 8555.03it/s] 64%|██████▎   | 254308/400000 [00:31<00:17, 8476.82it/s] 64%|██████▍   | 255157/400000 [00:31<00:17, 8448.37it/s] 64%|██████▍   | 256034/400000 [00:31<00:16, 8540.56it/s] 64%|██████▍   | 256905/400000 [00:31<00:16, 8588.40it/s] 64%|██████▍   | 257765/400000 [00:31<00:16, 8544.33it/s] 65%|██████▍   | 258620/400000 [00:31<00:16, 8458.43it/s] 65%|██████▍   | 259467/400000 [00:31<00:16, 8367.87it/s] 65%|██████▌   | 260306/400000 [00:31<00:16, 8371.99it/s] 65%|██████▌   | 261171/400000 [00:32<00:16, 8451.50it/s] 66%|██████▌   | 262017/400000 [00:32<00:16, 8431.37it/s] 66%|██████▌   | 262870/400000 [00:32<00:16, 8460.31it/s] 66%|██████▌   | 263717/400000 [00:32<00:16, 8461.79it/s] 66%|██████▌   | 264568/400000 [00:32<00:15, 8475.51it/s] 66%|██████▋   | 265416/400000 [00:32<00:16, 8366.23it/s] 67%|██████▋   | 266276/400000 [00:32<00:15, 8433.89it/s] 67%|██████▋   | 267120/400000 [00:32<00:16, 8022.38it/s] 67%|██████▋   | 267927/400000 [00:32<00:16, 7972.40it/s] 67%|██████▋   | 268780/400000 [00:32<00:16, 8130.90it/s] 67%|██████▋   | 269645/400000 [00:33<00:15, 8278.51it/s] 68%|██████▊   | 270498/400000 [00:33<00:15, 8351.24it/s] 68%|██████▊   | 271336/400000 [00:33<00:15, 8254.33it/s] 68%|██████▊   | 272164/400000 [00:33<00:15, 8193.35it/s] 68%|██████▊   | 272985/400000 [00:33<00:15, 8148.58it/s] 68%|██████▊   | 273816/400000 [00:33<00:15, 8194.57it/s] 69%|██████▊   | 274666/400000 [00:33<00:15, 8281.14it/s] 69%|██████▉   | 275518/400000 [00:33<00:14, 8348.48it/s] 69%|██████▉   | 276354/400000 [00:33<00:14, 8316.97it/s] 69%|██████▉   | 277227/400000 [00:33<00:14, 8435.28it/s] 70%|██████▉   | 278090/400000 [00:34<00:14, 8490.54it/s] 70%|██████▉   | 278942/400000 [00:34<00:14, 8497.02it/s] 70%|██████▉   | 279796/400000 [00:34<00:14, 8507.87it/s] 70%|███████   | 280648/400000 [00:34<00:14, 8448.22it/s] 70%|███████   | 281522/400000 [00:34<00:13, 8532.44it/s] 71%|███████   | 282381/400000 [00:34<00:13, 8548.82it/s] 71%|███████   | 283252/400000 [00:34<00:13, 8593.89it/s] 71%|███████   | 284112/400000 [00:34<00:13, 8533.85it/s] 71%|███████   | 284966/400000 [00:34<00:13, 8466.27it/s] 71%|███████▏  | 285817/400000 [00:34<00:13, 8476.47it/s] 72%|███████▏  | 286665/400000 [00:35<00:13, 8335.62it/s] 72%|███████▏  | 287511/400000 [00:35<00:13, 8372.11it/s] 72%|███████▏  | 288361/400000 [00:35<00:13, 8407.73it/s] 72%|███████▏  | 289203/400000 [00:35<00:13, 8258.15it/s] 73%|███████▎  | 290030/400000 [00:35<00:13, 8240.45it/s] 73%|███████▎  | 290884/400000 [00:35<00:13, 8326.08it/s] 73%|███████▎  | 291738/400000 [00:35<00:12, 8386.72it/s] 73%|███████▎  | 292603/400000 [00:35<00:12, 8461.44it/s] 73%|███████▎  | 293450/400000 [00:35<00:12, 8380.77it/s] 74%|███████▎  | 294303/400000 [00:35<00:12, 8422.57it/s] 74%|███████▍  | 295146/400000 [00:36<00:12, 8384.05it/s] 74%|███████▍  | 295985/400000 [00:36<00:12, 8353.88it/s] 74%|███████▍  | 296822/400000 [00:36<00:12, 8358.59it/s] 74%|███████▍  | 297662/400000 [00:36<00:12, 8368.23it/s] 75%|███████▍  | 298519/400000 [00:36<00:12, 8425.25it/s] 75%|███████▍  | 299364/400000 [00:36<00:11, 8431.71it/s] 75%|███████▌  | 300209/400000 [00:36<00:11, 8434.22it/s] 75%|███████▌  | 301079/400000 [00:36<00:11, 8512.01it/s] 75%|███████▌  | 301931/400000 [00:36<00:11, 8493.85it/s] 76%|███████▌  | 302781/400000 [00:36<00:11, 8471.87it/s] 76%|███████▌  | 303632/400000 [00:37<00:11, 8481.32it/s] 76%|███████▌  | 304481/400000 [00:37<00:11, 8464.32it/s] 76%|███████▋  | 305339/400000 [00:37<00:11, 8497.20it/s] 77%|███████▋  | 306194/400000 [00:37<00:11, 8511.54it/s] 77%|███████▋  | 307053/400000 [00:37<00:10, 8532.70it/s] 77%|███████▋  | 307923/400000 [00:37<00:10, 8580.86it/s] 77%|███████▋  | 308782/400000 [00:37<00:10, 8583.12it/s] 77%|███████▋  | 309645/400000 [00:37<00:10, 8596.17it/s] 78%|███████▊  | 310505/400000 [00:37<00:10, 8539.71it/s] 78%|███████▊  | 311360/400000 [00:37<00:10, 8537.00it/s] 78%|███████▊  | 312214/400000 [00:38<00:10, 8493.12it/s] 78%|███████▊  | 313072/400000 [00:38<00:10, 8517.73it/s] 78%|███████▊  | 313929/400000 [00:38<00:10, 8531.53it/s] 79%|███████▊  | 314785/400000 [00:38<00:09, 8537.41it/s] 79%|███████▉  | 315639/400000 [00:38<00:09, 8525.08it/s] 79%|███████▉  | 316502/400000 [00:38<00:09, 8556.25it/s] 79%|███████▉  | 317358/400000 [00:38<00:09, 8444.14it/s] 80%|███████▉  | 318205/400000 [00:38<00:09, 8449.83it/s] 80%|███████▉  | 319051/400000 [00:38<00:09, 8424.58it/s] 80%|███████▉  | 319911/400000 [00:38<00:09, 8475.17it/s] 80%|████████  | 320769/400000 [00:39<00:09, 8503.79it/s] 80%|████████  | 321624/400000 [00:39<00:09, 8517.47it/s] 81%|████████  | 322476/400000 [00:39<00:09, 8512.07it/s] 81%|████████  | 323328/400000 [00:39<00:09, 8484.73it/s] 81%|████████  | 324181/400000 [00:39<00:08, 8495.68it/s] 81%|████████▏ | 325035/400000 [00:39<00:08, 8507.94it/s] 81%|████████▏ | 325902/400000 [00:39<00:08, 8555.55it/s] 82%|████████▏ | 326761/400000 [00:39<00:08, 8565.53it/s] 82%|████████▏ | 327618/400000 [00:39<00:08, 8536.63it/s] 82%|████████▏ | 328472/400000 [00:39<00:08, 8532.14it/s] 82%|████████▏ | 329334/400000 [00:40<00:08, 8557.08it/s] 83%|████████▎ | 330196/400000 [00:40<00:08, 8574.98it/s] 83%|████████▎ | 331065/400000 [00:40<00:08, 8606.66it/s] 83%|████████▎ | 331926/400000 [00:40<00:07, 8561.96it/s] 83%|████████▎ | 332783/400000 [00:40<00:07, 8553.19it/s] 83%|████████▎ | 333639/400000 [00:40<00:07, 8525.03it/s] 84%|████████▎ | 334492/400000 [00:40<00:07, 8486.40it/s] 84%|████████▍ | 335358/400000 [00:40<00:07, 8536.25it/s] 84%|████████▍ | 336212/400000 [00:40<00:07, 8531.38it/s] 84%|████████▍ | 337066/400000 [00:40<00:07, 8526.29it/s] 84%|████████▍ | 337919/400000 [00:41<00:07, 8454.23it/s] 85%|████████▍ | 338772/400000 [00:41<00:07, 8476.05it/s] 85%|████████▍ | 339638/400000 [00:41<00:07, 8528.23it/s] 85%|████████▌ | 340502/400000 [00:41<00:06, 8561.04it/s] 85%|████████▌ | 341361/400000 [00:41<00:06, 8567.71it/s] 86%|████████▌ | 342219/400000 [00:41<00:06, 8569.33it/s] 86%|████████▌ | 343077/400000 [00:41<00:06, 8541.68it/s] 86%|████████▌ | 343936/400000 [00:41<00:06, 8554.39it/s] 86%|████████▌ | 344806/400000 [00:41<00:06, 8595.08it/s] 86%|████████▋ | 345666/400000 [00:42<00:06, 8593.07it/s] 87%|████████▋ | 346526/400000 [00:42<00:06, 8450.40it/s] 87%|████████▋ | 347385/400000 [00:42<00:06, 8489.21it/s] 87%|████████▋ | 348242/400000 [00:42<00:06, 8511.21it/s] 87%|████████▋ | 349110/400000 [00:42<00:05, 8554.94it/s] 87%|████████▋ | 349966/400000 [00:42<00:05, 8542.57it/s] 88%|████████▊ | 350840/400000 [00:42<00:05, 8598.63it/s] 88%|████████▊ | 351701/400000 [00:42<00:05, 8569.70it/s] 88%|████████▊ | 352559/400000 [00:42<00:05, 8554.69it/s] 88%|████████▊ | 353415/400000 [00:42<00:05, 8542.23it/s] 89%|████████▊ | 354270/400000 [00:43<00:05, 8301.17it/s] 89%|████████▉ | 355102/400000 [00:43<00:05, 8246.24it/s] 89%|████████▉ | 355941/400000 [00:43<00:05, 8287.87it/s] 89%|████████▉ | 356771/400000 [00:43<00:05, 8272.83it/s] 89%|████████▉ | 357599/400000 [00:43<00:05, 8122.94it/s] 90%|████████▉ | 358437/400000 [00:43<00:05, 8198.15it/s] 90%|████████▉ | 359292/400000 [00:43<00:04, 8300.22it/s] 90%|█████████ | 360123/400000 [00:43<00:04, 8265.66it/s] 90%|█████████ | 360951/400000 [00:43<00:04, 8248.01it/s] 90%|█████████ | 361786/400000 [00:43<00:04, 8276.39it/s] 91%|█████████ | 362615/400000 [00:44<00:04, 8044.11it/s] 91%|█████████ | 363422/400000 [00:44<00:04, 7857.96it/s] 91%|█████████ | 364221/400000 [00:44<00:04, 7897.11it/s] 91%|█████████▏| 365040/400000 [00:44<00:04, 7980.86it/s] 91%|█████████▏| 365887/400000 [00:44<00:04, 8120.12it/s] 92%|█████████▏| 366748/400000 [00:44<00:04, 8260.28it/s] 92%|█████████▏| 367596/400000 [00:44<00:03, 8323.76it/s] 92%|█████████▏| 368430/400000 [00:44<00:03, 8255.78it/s] 92%|█████████▏| 369280/400000 [00:44<00:03, 8326.43it/s] 93%|█████████▎| 370114/400000 [00:44<00:03, 8190.23it/s] 93%|█████████▎| 370970/400000 [00:45<00:03, 8296.57it/s] 93%|█████████▎| 371808/400000 [00:45<00:03, 8319.69it/s] 93%|█████████▎| 372663/400000 [00:45<00:03, 8387.29it/s] 93%|█████████▎| 373507/400000 [00:45<00:03, 8401.37it/s] 94%|█████████▎| 374356/400000 [00:45<00:03, 8427.62it/s] 94%|█████████▍| 375205/400000 [00:45<00:02, 8445.68it/s] 94%|█████████▍| 376061/400000 [00:45<00:02, 8477.26it/s] 94%|█████████▍| 376914/400000 [00:45<00:02, 8490.83it/s] 94%|█████████▍| 377768/400000 [00:45<00:02, 8503.74it/s] 95%|█████████▍| 378619/400000 [00:45<00:02, 8502.99it/s] 95%|█████████▍| 379470/400000 [00:46<00:02, 8481.46it/s] 95%|█████████▌| 380323/400000 [00:46<00:02, 8494.24it/s] 95%|█████████▌| 381173/400000 [00:46<00:02, 8494.74it/s] 96%|█████████▌| 382027/400000 [00:46<00:02, 8504.59it/s] 96%|█████████▌| 382878/400000 [00:46<00:02, 8466.84it/s] 96%|█████████▌| 383725/400000 [00:46<00:01, 8453.30it/s] 96%|█████████▌| 384579/400000 [00:46<00:01, 8477.89it/s] 96%|█████████▋| 385434/400000 [00:46<00:01, 8497.33it/s] 97%|█████████▋| 386286/400000 [00:46<00:01, 8503.32it/s] 97%|█████████▋| 387137/400000 [00:46<00:01, 8485.32it/s] 97%|█████████▋| 387986/400000 [00:47<00:01, 8191.66it/s] 97%|█████████▋| 388823/400000 [00:47<00:01, 8244.26it/s] 97%|█████████▋| 389671/400000 [00:47<00:01, 8311.77it/s] 98%|█████████▊| 390504/400000 [00:47<00:01, 8236.21it/s] 98%|█████████▊| 391340/400000 [00:47<00:01, 8271.05it/s] 98%|█████████▊| 392177/400000 [00:47<00:00, 8299.88it/s] 98%|█████████▊| 393015/400000 [00:47<00:00, 8323.12it/s] 98%|█████████▊| 393857/400000 [00:47<00:00, 8350.04it/s] 99%|█████████▊| 394708/400000 [00:47<00:00, 8397.26it/s] 99%|█████████▉| 395549/400000 [00:47<00:00, 8374.59it/s] 99%|█████████▉| 396387/400000 [00:48<00:00, 8327.76it/s] 99%|█████████▉| 397229/400000 [00:48<00:00, 8353.16it/s]100%|█████████▉| 398083/400000 [00:48<00:00, 8406.28it/s]100%|█████████▉| 398933/400000 [00:48<00:00, 8433.56it/s]100%|█████████▉| 399783/400000 [00:48<00:00, 8452.97it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8245.72it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f2fac063b70>, <torchtext.data.dataset.TabularDataset object at 0x7f2fac063cc0>, <torchtext.vocab.Vocab object at 0x7f2fac063be0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.38 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.81 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.81 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.11 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.11 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.10 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.10 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.24 file/s]2020-07-22 12:18:15.008856: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-22 12:18:15.013349: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-22 12:18:15.013530: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5641740d02a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-22 12:18:15.013545: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 460445.37it/s] 80%|███████▉  | 7888896/9912422 [00:00<00:03, 656041.17it/s]9920512it [00:00, 44423707.33it/s]                           
0it [00:00, ?it/s]32768it [00:00, 670682.56it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1007387.44it/s]1654784it [00:00, 12525227.88it/s]                           
0it [00:00, ?it/s]8192it [00:00, 196436.77it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
