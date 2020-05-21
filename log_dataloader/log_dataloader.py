
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a79f4a8a2dce0380e0116b4e6ded1ae34a4a07e6', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'a79f4a8a2dce0380e0116b4e6ded1ae34a4a07e6', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a79f4a8a2dce0380e0116b4e6ded1ae34a4a07e6

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a79f4a8a2dce0380e0116b4e6ded1ae34a4a07e6

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/a79f4a8a2dce0380e0116b4e6ded1ae34a4a07e6

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  



###### Test_run_model  #############################################################



 ####################################################################################################
dataset/json/refactor/keras_textcnn.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.dataloader.textcnn.py",
      "maxlen": 40,
      "max_features": 5,
      "embedding_dims": 50
    },
    "data_pars": {
      "data_info": {
        "dataset": "mlmodels/dataset/text/imdb",
        "pass_data_pars": false,
        "train": true,
        "maxlen": 40,
        "max_features": 5
      },
      "preprocessors": [
        {
          "name": "loader",
          "uri": "mlmodels/preprocess/generic.py::NumpyDataset",
          "args": {
            "numpy_loader_args": {
              "allow_pickle": true
            },
            "encoding": "'ISO-8859-1'"
          }
        },
        {
          "name": "imdb_process",
          "uri": "mlmodels/preprocess/text_keras.py::IMDBDataset",
          "args": {
            "num_words": 5
          }
        }
      ]
    },
    "compute_pars": {
      "engine": "adam",
      "loss": "binary_crossentropy",
      "metrics": [
        "accuracy"
      ],
      "batch_size": 1000,
      "epochs": 1
    },
    "out_pars": {
      "path": "mlmodels/output/textcnn_keras//model.h5",
      "model_path": "mlmodels/output/textcnn_keras/model.h5"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.dataloader.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/dataloader/textcnn.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 40, 50)       250         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 38, 128)      19328       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 37, 128)      25728       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 36, 128)      32128       embedding_1[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  <mlmodels.model_keras.dataloader.textcnn.Model object at 0x7fb64779f6a0> 

  #### Fit   ######################################################## 
Loading data...

  URL:  mlmodels/preprocess/generic.py::NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.NumpyDataset'>
cls_name : NumpyDataset
Dataset File path :  mlmodels/dataset/text/imdb.npz

  URL:  mlmodels/preprocess/text_keras.py::IMDBDataset {'num_words': 5} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/text_keras.IMDBDataset'>
cls_name : IMDBDataset
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-21 09:49:38.120135: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-21 09:49:38.124542: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-21 09:49:38.124753: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a5828ad650 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 09:49:38.124769: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

2020-05-21 09:49:38.505375: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 19456000 exceeds 10% of system memory.
2020-05-21 09:49:38.505528: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 18432000 exceeds 10% of system memory.
2020-05-21 09:49:38.599617: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 19456000 exceeds 10% of system memory.
2020-05-21 09:49:38.605393: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 18432000 exceeds 10% of system memory.
2020-05-21 09:49:38.623857: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 18944000 exceeds 10% of system memory.

 Object Creation

 Object Compute

 Object get_data
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.0653 - accuracy: 0.4740
 2000/25000 [=>............................] - ETA: 8s - loss: 8.0270 - accuracy: 0.4765 
 3000/25000 [==>...........................] - ETA: 6s - loss: 8.0040 - accuracy: 0.4780
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.9388 - accuracy: 0.4823
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8844 - accuracy: 0.4858
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7816 - accuracy: 0.4925
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7827 - accuracy: 0.4924
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7586 - accuracy: 0.4940
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7518 - accuracy: 0.4944
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7080 - accuracy: 0.4973
11000/25000 [============>.................] - ETA: 3s - loss: 7.7294 - accuracy: 0.4959
12000/25000 [=============>................] - ETA: 3s - loss: 7.7228 - accuracy: 0.4963
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7350 - accuracy: 0.4955
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7039 - accuracy: 0.4976
15000/25000 [=================>............] - ETA: 2s - loss: 7.6942 - accuracy: 0.4982
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7107 - accuracy: 0.4971
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7207 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7067 - accuracy: 0.4974
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7102 - accuracy: 0.4972
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7073 - accuracy: 0.4974
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7148 - accuracy: 0.4969
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 7s 287us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Predict   #################################################### 
Loading data...

  URL:  mlmodels/preprocess/generic.py::NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.NumpyDataset'>
cls_name : NumpyDataset
Dataset File path :  mlmodels/dataset/text/imdb.npz

  URL:  mlmodels/preprocess/text_keras.py::IMDBDataset {'num_words': 5} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/text_keras.IMDBDataset'>
cls_name : IMDBDataset

 Object Creation

 Object Compute

 Object get_data
(array([[1.],
       [1.],
       [1.],
       ...,
       [1.],
       [1.],
       [1.]], dtype=float32), None)

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 
