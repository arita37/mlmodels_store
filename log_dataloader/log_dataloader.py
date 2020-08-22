
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/13f78dac13e826a41e3a7922ab3568a9b02adef6', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '13f78dac13e826a41e3a7922ab3568a9b02adef6', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/13f78dac13e826a41e3a7922ab3568a9b02adef6

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/13f78dac13e826a41e3a7922ab3568a9b02adef6

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/13f78dac13e826a41e3a7922ab3568a9b02adef6

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f686e872268> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f686e872268> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f68d9598400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f68d9598400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f68f87b4ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f68f87b4ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f68868d40d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f68868d40d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f68868d40d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 3006464/9912422 [00:00<00:00, 29928238.77it/s]9920512it [00:00, 30896659.15it/s]                             
0it [00:00, ?it/s]32768it [00:00, 975366.93it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160193.48it/s]1654784it [00:00, 11178559.76it/s]                         
0it [00:00, ?it/s]8192it [00:00, 158091.38it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f686dae9b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f686daf7e48>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f68868ccd08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f68868ccd08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f68868ccd08> , (data_info, **args) 

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
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:16,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:53,  2.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:53,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:52,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:51,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:51,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:51,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  4.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:33,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:15,  7.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:14,  7.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  7.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 10.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 10.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 32.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 48.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 61.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 61.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 61.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 61.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 61.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 61.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 61.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 61.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 61.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 61.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 61.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 61.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 61.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 61.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 61.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 61.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 61.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 61.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
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

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 61.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 61.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 61.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 61.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 61.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 61.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 61.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 61.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 61.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 61.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 61.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 61.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.54s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 61.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 61.70 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 61.70 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.24 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ url]
0 examples [00:00, ? examples/s]2020-08-22 12:07:01.190515: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-22 12:07:01.202377: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-22 12:07:01.202717: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d973e218a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-22 12:07:01.202740: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
22 examples [00:00, 219.12 examples/s]132 examples [00:00, 288.24 examples/s]246 examples [00:00, 371.30 examples/s]365 examples [00:00, 467.64 examples/s]484 examples [00:00, 571.13 examples/s]600 examples [00:00, 673.17 examples/s]716 examples [00:00, 769.49 examples/s]831 examples [00:00, 854.09 examples/s]949 examples [00:00, 930.82 examples/s]1066 examples [00:01, 990.01 examples/s]1182 examples [00:01, 1034.38 examples/s]1295 examples [00:01, 1044.96 examples/s]1407 examples [00:01, 1001.10 examples/s]1521 examples [00:01, 1038.75 examples/s]1636 examples [00:01, 1067.22 examples/s]1751 examples [00:01, 1088.60 examples/s]1867 examples [00:01, 1107.36 examples/s]1980 examples [00:01, 1108.52 examples/s]2099 examples [00:01, 1131.48 examples/s]2216 examples [00:02, 1139.95 examples/s]2331 examples [00:02, 1136.05 examples/s]2446 examples [00:02, 1131.62 examples/s]2561 examples [00:02, 1134.46 examples/s]2677 examples [00:02, 1141.92 examples/s]2796 examples [00:02, 1153.88 examples/s]2912 examples [00:02, 1153.10 examples/s]3028 examples [00:02, 1137.76 examples/s]3142 examples [00:02, 1137.46 examples/s]3262 examples [00:02, 1154.42 examples/s]3378 examples [00:03, 1149.06 examples/s]3493 examples [00:03, 1112.07 examples/s]3608 examples [00:03, 1122.60 examples/s]3723 examples [00:03, 1128.53 examples/s]3845 examples [00:03, 1153.15 examples/s]3961 examples [00:03, 1147.80 examples/s]4079 examples [00:03, 1156.57 examples/s]4198 examples [00:03, 1165.32 examples/s]4315 examples [00:03, 1136.32 examples/s]4429 examples [00:03, 1137.41 examples/s]4547 examples [00:04, 1147.87 examples/s]4662 examples [00:04, 1135.00 examples/s]4776 examples [00:04, 1130.12 examples/s]4890 examples [00:04, 1123.45 examples/s]5006 examples [00:04, 1131.91 examples/s]5121 examples [00:04, 1137.07 examples/s]5235 examples [00:04, 1118.66 examples/s]5353 examples [00:04, 1135.94 examples/s]5467 examples [00:04, 1113.06 examples/s]5580 examples [00:05, 1117.42 examples/s]5696 examples [00:05, 1128.46 examples/s]5809 examples [00:05, 1117.91 examples/s]5924 examples [00:05, 1125.74 examples/s]6040 examples [00:05, 1133.38 examples/s]6159 examples [00:05, 1147.50 examples/s]6274 examples [00:05, 1147.33 examples/s]6389 examples [00:05, 1137.43 examples/s]6503 examples [00:05, 1133.94 examples/s]6617 examples [00:05, 1133.13 examples/s]6737 examples [00:06, 1150.97 examples/s]6853 examples [00:06, 1153.09 examples/s]6969 examples [00:06, 1151.06 examples/s]7089 examples [00:06, 1164.73 examples/s]7208 examples [00:06, 1170.52 examples/s]7329 examples [00:06, 1179.73 examples/s]7452 examples [00:06, 1193.23 examples/s]7572 examples [00:06, 1183.38 examples/s]7691 examples [00:06, 1182.06 examples/s]7810 examples [00:06, 1168.22 examples/s]7932 examples [00:07, 1183.04 examples/s]8052 examples [00:07, 1187.58 examples/s]8171 examples [00:07, 1167.14 examples/s]8289 examples [00:07, 1168.26 examples/s]8406 examples [00:07, 1162.52 examples/s]8527 examples [00:07, 1174.75 examples/s]8645 examples [00:07, 1175.87 examples/s]8765 examples [00:07, 1181.72 examples/s]8889 examples [00:07, 1196.50 examples/s]9009 examples [00:07, 1149.64 examples/s]9126 examples [00:08, 1154.88 examples/s]9243 examples [00:08, 1158.19 examples/s]9360 examples [00:08, 1159.17 examples/s]9477 examples [00:08, 1160.52 examples/s]9594 examples [00:08, 1144.77 examples/s]9709 examples [00:08, 1142.49 examples/s]9830 examples [00:08, 1160.46 examples/s]9947 examples [00:08, 1143.53 examples/s]10062 examples [00:08, 1099.55 examples/s]10173 examples [00:08, 1093.23 examples/s]10292 examples [00:09, 1118.00 examples/s]10411 examples [00:09, 1137.62 examples/s]10526 examples [00:09, 1129.63 examples/s]10645 examples [00:09, 1146.66 examples/s]10760 examples [00:09, 1132.23 examples/s]10882 examples [00:09, 1155.80 examples/s]10998 examples [00:09, 1119.24 examples/s]11116 examples [00:09, 1134.37 examples/s]11236 examples [00:09, 1152.86 examples/s]11352 examples [00:10, 1151.22 examples/s]11470 examples [00:10, 1157.27 examples/s]11586 examples [00:10, 1155.61 examples/s]11702 examples [00:10, 1150.71 examples/s]11819 examples [00:10, 1155.80 examples/s]11935 examples [00:10, 1146.15 examples/s]12050 examples [00:10, 1146.50 examples/s]12170 examples [00:10, 1159.48 examples/s]12287 examples [00:10, 1161.99 examples/s]12404 examples [00:10, 1162.89 examples/s]12521 examples [00:11, 1155.75 examples/s]12643 examples [00:11, 1171.78 examples/s]12761 examples [00:11, 1146.27 examples/s]12876 examples [00:11, 1128.39 examples/s]12995 examples [00:11, 1144.95 examples/s]13110 examples [00:11, 1135.22 examples/s]13230 examples [00:11, 1152.83 examples/s]13347 examples [00:11, 1157.08 examples/s]13463 examples [00:11, 1155.38 examples/s]13584 examples [00:11, 1169.70 examples/s]13702 examples [00:12, 1153.48 examples/s]13818 examples [00:12, 1151.05 examples/s]13935 examples [00:12, 1154.96 examples/s]14051 examples [00:12, 1155.18 examples/s]14167 examples [00:12, 1127.58 examples/s]14280 examples [00:12, 1115.90 examples/s]14396 examples [00:12, 1127.94 examples/s]14513 examples [00:12, 1137.97 examples/s]14633 examples [00:12, 1155.31 examples/s]14755 examples [00:12, 1173.53 examples/s]14873 examples [00:13, 1160.02 examples/s]14995 examples [00:13, 1176.90 examples/s]15113 examples [00:13, 1169.72 examples/s]15231 examples [00:13, 1159.54 examples/s]15348 examples [00:13, 1152.94 examples/s]15469 examples [00:13, 1168.11 examples/s]15591 examples [00:13, 1180.47 examples/s]15710 examples [00:13, 1165.21 examples/s]15827 examples [00:13, 1166.40 examples/s]15944 examples [00:13, 1141.00 examples/s]16059 examples [00:14, 1139.96 examples/s]16174 examples [00:14, 1134.50 examples/s]16291 examples [00:14, 1144.13 examples/s]16409 examples [00:14, 1154.29 examples/s]16525 examples [00:14, 1146.03 examples/s]16640 examples [00:14, 1143.69 examples/s]16760 examples [00:14, 1158.79 examples/s]16876 examples [00:14, 1106.92 examples/s]16991 examples [00:14, 1118.29 examples/s]17104 examples [00:15, 1117.04 examples/s]17220 examples [00:15, 1128.69 examples/s]17339 examples [00:15, 1144.67 examples/s]17454 examples [00:15, 1144.78 examples/s]17574 examples [00:15, 1160.79 examples/s]17691 examples [00:15, 1152.84 examples/s]17807 examples [00:15, 1139.13 examples/s]17927 examples [00:15, 1154.10 examples/s]18048 examples [00:15, 1168.00 examples/s]18168 examples [00:15, 1175.45 examples/s]18286 examples [00:16, 1167.87 examples/s]18403 examples [00:16, 1154.46 examples/s]18523 examples [00:16, 1166.24 examples/s]18641 examples [00:16, 1168.26 examples/s]18758 examples [00:16, 1119.03 examples/s]18871 examples [00:16, 1119.27 examples/s]18984 examples [00:16, 1122.24 examples/s]19097 examples [00:16, 1117.69 examples/s]19209 examples [00:16, 1108.67 examples/s]19321 examples [00:16, 1102.67 examples/s]19437 examples [00:17, 1117.30 examples/s]19549 examples [00:17, 1112.99 examples/s]19661 examples [00:17, 1114.87 examples/s]19774 examples [00:17, 1118.80 examples/s]19888 examples [00:17, 1122.07 examples/s]20001 examples [00:17, 1064.41 examples/s]20115 examples [00:17, 1085.69 examples/s]20233 examples [00:17, 1109.98 examples/s]20345 examples [00:17, 1110.27 examples/s]20461 examples [00:17, 1123.30 examples/s]20581 examples [00:18, 1144.20 examples/s]20696 examples [00:18, 1140.49 examples/s]20811 examples [00:18, 1135.77 examples/s]20931 examples [00:18, 1153.78 examples/s]21047 examples [00:18, 1152.67 examples/s]21163 examples [00:18, 1121.79 examples/s]21276 examples [00:18, 1120.38 examples/s]21389 examples [00:18, 1118.92 examples/s]21503 examples [00:18, 1123.07 examples/s]21616 examples [00:18, 1121.80 examples/s]21731 examples [00:19, 1128.68 examples/s]21844 examples [00:19, 1124.96 examples/s]21957 examples [00:19, 1105.49 examples/s]22071 examples [00:19, 1113.76 examples/s]22183 examples [00:19, 1093.82 examples/s]22295 examples [00:19, 1100.20 examples/s]22410 examples [00:19, 1111.76 examples/s]22522 examples [00:19, 1102.50 examples/s]22637 examples [00:19, 1115.27 examples/s]22750 examples [00:20, 1117.42 examples/s]22862 examples [00:20, 1114.53 examples/s]22976 examples [00:20, 1118.50 examples/s]23091 examples [00:20, 1125.49 examples/s]23209 examples [00:20, 1140.02 examples/s]23324 examples [00:20, 1126.49 examples/s]23438 examples [00:20, 1129.39 examples/s]23552 examples [00:20, 1079.54 examples/s]23663 examples [00:20, 1085.98 examples/s]23777 examples [00:20, 1099.64 examples/s]23888 examples [00:21, 1088.30 examples/s]24002 examples [00:21, 1100.79 examples/s]24113 examples [00:21, 1094.81 examples/s]24224 examples [00:21, 1098.10 examples/s]24340 examples [00:21, 1114.69 examples/s]24452 examples [00:21, 1106.57 examples/s]24565 examples [00:21, 1111.34 examples/s]24677 examples [00:21, 1111.33 examples/s]24792 examples [00:21, 1121.14 examples/s]24905 examples [00:21, 1123.64 examples/s]25020 examples [00:22, 1130.81 examples/s]25134 examples [00:22, 1094.41 examples/s]25244 examples [00:22, 1090.13 examples/s]25358 examples [00:22, 1102.16 examples/s]25469 examples [00:22, 1103.89 examples/s]25582 examples [00:22, 1109.49 examples/s]25695 examples [00:22, 1112.38 examples/s]25807 examples [00:22, 1081.30 examples/s]25919 examples [00:22, 1089.07 examples/s]26029 examples [00:22, 1090.80 examples/s]26143 examples [00:23, 1104.59 examples/s]26254 examples [00:23, 1103.74 examples/s]26365 examples [00:23, 1101.84 examples/s]26476 examples [00:23, 1101.93 examples/s]26590 examples [00:23, 1112.83 examples/s]26703 examples [00:23, 1116.76 examples/s]26816 examples [00:23, 1119.29 examples/s]26928 examples [00:23, 1119.45 examples/s]27040 examples [00:23, 1112.21 examples/s]27152 examples [00:23, 1110.53 examples/s]27273 examples [00:24, 1136.27 examples/s]27387 examples [00:24, 1133.74 examples/s]27501 examples [00:24, 1102.34 examples/s]27613 examples [00:24, 1105.22 examples/s]27727 examples [00:24, 1115.12 examples/s]27843 examples [00:24, 1127.02 examples/s]27962 examples [00:24, 1142.98 examples/s]28077 examples [00:24, 1122.15 examples/s]28190 examples [00:24, 1117.03 examples/s]28306 examples [00:25, 1129.52 examples/s]28420 examples [00:25, 1129.43 examples/s]28535 examples [00:25, 1135.34 examples/s]28650 examples [00:25, 1139.49 examples/s]28765 examples [00:25, 1133.34 examples/s]28880 examples [00:25, 1137.91 examples/s]28999 examples [00:25, 1151.10 examples/s]29119 examples [00:25, 1163.53 examples/s]29237 examples [00:25, 1166.50 examples/s]29356 examples [00:25, 1171.95 examples/s]29474 examples [00:26, 1145.40 examples/s]29589 examples [00:26, 1145.28 examples/s]29709 examples [00:26, 1160.47 examples/s]29826 examples [00:26, 1148.51 examples/s]29941 examples [00:26, 1144.30 examples/s]30056 examples [00:26, 1092.42 examples/s]30170 examples [00:26, 1103.91 examples/s]30287 examples [00:26, 1121.61 examples/s]30400 examples [00:26, 1114.98 examples/s]30515 examples [00:26, 1123.93 examples/s]30632 examples [00:27, 1136.39 examples/s]30750 examples [00:27, 1146.68 examples/s]30865 examples [00:27, 1145.28 examples/s]30981 examples [00:27, 1147.69 examples/s]31102 examples [00:27, 1163.75 examples/s]31220 examples [00:27, 1165.94 examples/s]31341 examples [00:27, 1176.36 examples/s]31459 examples [00:27, 1163.69 examples/s]31576 examples [00:27, 1156.74 examples/s]31692 examples [00:27, 1132.00 examples/s]31808 examples [00:28, 1137.53 examples/s]31923 examples [00:28, 1140.66 examples/s]32038 examples [00:28, 1139.99 examples/s]32153 examples [00:28, 1128.09 examples/s]32268 examples [00:28, 1133.21 examples/s]32382 examples [00:28, 1131.11 examples/s]32496 examples [00:28, 1105.89 examples/s]32610 examples [00:28, 1114.68 examples/s]32722 examples [00:28, 1067.77 examples/s]32835 examples [00:29, 1083.34 examples/s]32947 examples [00:29, 1093.92 examples/s]33057 examples [00:29, 1095.12 examples/s]33167 examples [00:29, 1093.14 examples/s]33278 examples [00:29, 1096.71 examples/s]33391 examples [00:29, 1106.40 examples/s]33502 examples [00:29, 1080.62 examples/s]33613 examples [00:29, 1085.93 examples/s]33722 examples [00:29, 1080.04 examples/s]33831 examples [00:29, 1078.18 examples/s]33946 examples [00:30, 1098.42 examples/s]34056 examples [00:30, 1054.85 examples/s]34167 examples [00:30, 1069.38 examples/s]34277 examples [00:30, 1076.00 examples/s]34393 examples [00:30, 1097.11 examples/s]34510 examples [00:30, 1116.40 examples/s]34622 examples [00:30, 1106.61 examples/s]34736 examples [00:30, 1114.46 examples/s]34848 examples [00:30, 1074.05 examples/s]34963 examples [00:30, 1094.24 examples/s]35087 examples [00:31, 1132.45 examples/s]35205 examples [00:31, 1143.80 examples/s]35320 examples [00:31, 1141.42 examples/s]35435 examples [00:31, 1055.78 examples/s]35551 examples [00:31, 1084.08 examples/s]35666 examples [00:31, 1101.60 examples/s]35778 examples [00:31, 1102.83 examples/s]35892 examples [00:31, 1111.74 examples/s]36004 examples [00:31, 1102.66 examples/s]36115 examples [00:31, 1100.90 examples/s]36233 examples [00:32, 1121.74 examples/s]36346 examples [00:32, 1115.58 examples/s]36461 examples [00:32, 1125.28 examples/s]36575 examples [00:32, 1127.94 examples/s]36691 examples [00:32, 1136.98 examples/s]36805 examples [00:32, 1112.70 examples/s]36918 examples [00:32, 1115.96 examples/s]37037 examples [00:32, 1134.78 examples/s]37153 examples [00:32, 1140.54 examples/s]37269 examples [00:33, 1144.45 examples/s]37384 examples [00:33, 1123.86 examples/s]37503 examples [00:33, 1140.70 examples/s]37620 examples [00:33, 1147.67 examples/s]37738 examples [00:33, 1155.28 examples/s]37854 examples [00:33, 1139.06 examples/s]37969 examples [00:33, 1132.81 examples/s]38085 examples [00:33, 1138.95 examples/s]38199 examples [00:33, 1133.22 examples/s]38313 examples [00:33, 1122.74 examples/s]38426 examples [00:34, 1114.93 examples/s]38538 examples [00:34, 1113.40 examples/s]38650 examples [00:34, 1095.67 examples/s]38762 examples [00:34, 1101.42 examples/s]38873 examples [00:34, 1102.37 examples/s]38984 examples [00:34, 1100.56 examples/s]39101 examples [00:34, 1119.04 examples/s]39220 examples [00:34, 1139.28 examples/s]39335 examples [00:34, 1130.23 examples/s]39449 examples [00:34, 1099.32 examples/s]39572 examples [00:35, 1133.08 examples/s]39687 examples [00:35, 1111.20 examples/s]39802 examples [00:35, 1122.16 examples/s]39924 examples [00:35, 1148.11 examples/s]40040 examples [00:35, 1081.91 examples/s]40161 examples [00:35, 1114.82 examples/s]40280 examples [00:35, 1135.93 examples/s]40397 examples [00:35, 1144.40 examples/s]40513 examples [00:35, 1145.82 examples/s]40629 examples [00:35, 1149.27 examples/s]40746 examples [00:36, 1154.50 examples/s]40862 examples [00:36, 1154.24 examples/s]40978 examples [00:36, 1155.05 examples/s]41094 examples [00:36, 1144.87 examples/s]41211 examples [00:36, 1150.47 examples/s]41327 examples [00:36, 1150.79 examples/s]41446 examples [00:36, 1162.05 examples/s]41565 examples [00:36, 1169.32 examples/s]41682 examples [00:36, 1151.37 examples/s]41798 examples [00:37, 1145.67 examples/s]41920 examples [00:37, 1165.59 examples/s]42040 examples [00:37, 1173.96 examples/s]42158 examples [00:37, 1168.98 examples/s]42275 examples [00:37, 1157.70 examples/s]42395 examples [00:37, 1167.54 examples/s]42518 examples [00:37, 1182.79 examples/s]42637 examples [00:37, 1163.10 examples/s]42754 examples [00:37, 1164.93 examples/s]42871 examples [00:37, 1157.38 examples/s]42987 examples [00:38, 1152.78 examples/s]43107 examples [00:38, 1164.05 examples/s]43224 examples [00:38, 1134.32 examples/s]43346 examples [00:38, 1157.45 examples/s]43463 examples [00:38, 1140.87 examples/s]43588 examples [00:38, 1171.00 examples/s]43710 examples [00:38, 1184.51 examples/s]43829 examples [00:38, 1172.81 examples/s]43947 examples [00:38, 1174.15 examples/s]44065 examples [00:38, 1159.35 examples/s]44182 examples [00:39, 1149.99 examples/s]44299 examples [00:39, 1152.97 examples/s]44415 examples [00:39, 1151.30 examples/s]44531 examples [00:39, 1152.58 examples/s]44647 examples [00:39, 1138.02 examples/s]44761 examples [00:39, 1136.21 examples/s]44875 examples [00:39, 1122.32 examples/s]44988 examples [00:39, 1118.19 examples/s]45101 examples [00:39, 1121.59 examples/s]45214 examples [00:39, 1097.12 examples/s]45331 examples [00:40, 1116.79 examples/s]45443 examples [00:40, 1097.20 examples/s]45554 examples [00:40, 1099.43 examples/s]45667 examples [00:40, 1107.08 examples/s]45778 examples [00:40, 1107.82 examples/s]45894 examples [00:40, 1122.07 examples/s]46013 examples [00:40, 1128.23 examples/s]46130 examples [00:40, 1139.25 examples/s]46249 examples [00:40, 1151.61 examples/s]46367 examples [00:40, 1159.43 examples/s]46484 examples [00:41, 1109.14 examples/s]46602 examples [00:41, 1128.39 examples/s]46719 examples [00:41, 1139.63 examples/s]46834 examples [00:41, 1130.25 examples/s]46948 examples [00:41, 1127.79 examples/s]47070 examples [00:41, 1152.51 examples/s]47194 examples [00:41, 1176.14 examples/s]47314 examples [00:41, 1182.27 examples/s]47433 examples [00:41, 1182.12 examples/s]47552 examples [00:42, 1172.95 examples/s]47670 examples [00:42, 1160.82 examples/s]47787 examples [00:42, 1134.97 examples/s]47902 examples [00:42, 1138.51 examples/s]48016 examples [00:42, 1130.06 examples/s]48130 examples [00:42, 1129.95 examples/s]48244 examples [00:42, 1098.43 examples/s]48362 examples [00:42, 1118.76 examples/s]48475 examples [00:42, 1105.60 examples/s]48593 examples [00:42, 1126.30 examples/s]48706 examples [00:43, 1120.33 examples/s]48820 examples [00:43, 1123.93 examples/s]48935 examples [00:43, 1129.76 examples/s]49049 examples [00:43, 1116.69 examples/s]49170 examples [00:43, 1141.16 examples/s]49285 examples [00:43, 1138.92 examples/s]49404 examples [00:43, 1151.98 examples/s]49522 examples [00:43, 1155.44 examples/s]49640 examples [00:43, 1162.46 examples/s]49757 examples [00:43, 1151.86 examples/s]49873 examples [00:44, 1143.72 examples/s]49988 examples [00:44, 1140.49 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 17%|█▋        | 8621/50000 [00:00<00:00, 86208.03 examples/s] 46%|████▌     | 22817/50000 [00:00<00:00, 97720.44 examples/s] 76%|███████▌  | 37797/50000 [00:00<00:00, 109099.04 examples/s]                                                                0 examples [00:00, ? examples/s]100 examples [00:00, 999.26 examples/s]208 examples [00:00, 1021.44 examples/s]321 examples [00:00, 1051.34 examples/s]439 examples [00:00, 1086.41 examples/s]555 examples [00:00, 1105.67 examples/s]663 examples [00:00, 1097.30 examples/s]774 examples [00:00, 1100.59 examples/s]895 examples [00:00, 1129.75 examples/s]1019 examples [00:00, 1158.44 examples/s]1135 examples [00:01, 1157.25 examples/s]1256 examples [00:01, 1170.52 examples/s]1374 examples [00:01, 1171.67 examples/s]1494 examples [00:01, 1178.61 examples/s]1612 examples [00:01, 1165.88 examples/s]1729 examples [00:01, 1139.08 examples/s]1845 examples [00:01, 1143.18 examples/s]1960 examples [00:01, 1142.95 examples/s]2075 examples [00:01, 1131.86 examples/s]2195 examples [00:01, 1150.30 examples/s]2312 examples [00:02, 1153.81 examples/s]2428 examples [00:02, 1147.19 examples/s]2545 examples [00:02, 1152.45 examples/s]2662 examples [00:02, 1155.91 examples/s]2779 examples [00:02, 1157.11 examples/s]2895 examples [00:02, 1154.60 examples/s]3016 examples [00:02, 1170.25 examples/s]3134 examples [00:02, 1169.80 examples/s]3252 examples [00:02, 1160.27 examples/s]3373 examples [00:02, 1172.40 examples/s]3491 examples [00:03, 1165.09 examples/s]3613 examples [00:03, 1180.00 examples/s]3732 examples [00:03, 1149.05 examples/s]3851 examples [00:03, 1159.54 examples/s]3968 examples [00:03, 1150.67 examples/s]4084 examples [00:03, 1145.51 examples/s]4202 examples [00:03, 1153.14 examples/s]4318 examples [00:03, 1145.19 examples/s]4438 examples [00:03, 1159.82 examples/s]4558 examples [00:03, 1169.02 examples/s]4675 examples [00:04, 1160.01 examples/s]4792 examples [00:04, 1133.87 examples/s]4907 examples [00:04, 1138.43 examples/s]5022 examples [00:04, 1140.18 examples/s]5143 examples [00:04, 1159.77 examples/s]5260 examples [00:04, 1152.22 examples/s]5379 examples [00:04, 1160.47 examples/s]5496 examples [00:04, 918.89 examples/s] 5618 examples [00:04, 990.68 examples/s]5732 examples [00:05, 1029.07 examples/s]5851 examples [00:05, 1071.78 examples/s]5969 examples [00:05, 1101.38 examples/s]6087 examples [00:05, 1121.28 examples/s]6207 examples [00:05, 1142.30 examples/s]6323 examples [00:05, 1139.42 examples/s]6439 examples [00:05, 1123.00 examples/s]6553 examples [00:05, 1121.88 examples/s]6672 examples [00:05, 1139.17 examples/s]6789 examples [00:05, 1147.91 examples/s]6905 examples [00:06, 1149.95 examples/s]7021 examples [00:06, 1123.80 examples/s]7134 examples [00:06, 1099.99 examples/s]7253 examples [00:06, 1123.33 examples/s]7371 examples [00:06, 1137.28 examples/s]7486 examples [00:06, 1140.63 examples/s]7601 examples [00:06, 1070.71 examples/s]7716 examples [00:06, 1092.62 examples/s]7832 examples [00:06, 1109.60 examples/s]7947 examples [00:07, 1119.57 examples/s]8060 examples [00:07, 1121.73 examples/s]8176 examples [00:07, 1132.81 examples/s]8294 examples [00:07, 1146.28 examples/s]8411 examples [00:07, 1153.19 examples/s]8527 examples [00:07, 1154.93 examples/s]8644 examples [00:07, 1156.32 examples/s]8760 examples [00:07, 1148.64 examples/s]8875 examples [00:07, 1139.37 examples/s]8990 examples [00:07, 1120.87 examples/s]9110 examples [00:08, 1142.16 examples/s]9225 examples [00:08, 1142.45 examples/s]9340 examples [00:08, 1137.43 examples/s]9454 examples [00:08, 1137.91 examples/s]9573 examples [00:08, 1152.30 examples/s]9689 examples [00:08, 1144.26 examples/s]9804 examples [00:08, 1140.02 examples/s]9919 examples [00:08, 1140.78 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW94LSD/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW94LSD/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f68868d40d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f68868d40d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f68868d40d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f68d22b2828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f680dc715f8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f68868d40d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f68868d40d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f68868d40d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f686dae9b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f686db18048>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.54 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.54 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.54 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.58 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.58 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.43 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.43 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.13 file/s]2020-08-22 12:08:01.569131: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-22 12:08:01.572859: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-22 12:08:01.572990: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5585edde9ee0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-22 12:08:01.573003: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 159587.09it/s] 70%|██████▉   | 6914048/9912422 [00:00<00:13, 227755.55it/s]9920512it [00:00, 43325461.41it/s]                           
0it [00:00, ?it/s]32768it [00:00, 508823.98it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161875.33it/s]1654784it [00:00, 11177353.62it/s]                         
0it [00:00, ?it/s]8192it [00:00, 196240.44it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
