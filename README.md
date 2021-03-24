# Document_MultiObject_Detection_FasterRCNN
Detect handwritten Signatures and Dates on documents with FasterRCNN in Keras

suggested directory structure:
'''
-data_folder ("data")
	- train_images
	- test_images
	- train_path ("train_annotation_sig_date.txt")
	- test_path ("test_annotation_sig_date.txt")
	- predict_path
- main_path ("Model")
	- config_filename (default: "model_config.pickle")
	- output_weight_path(default: "./Model/model_frcnn.hdf5')
	- base_weight_path (default: "./Model/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
	- record_path (default: "./Model/record.csv")
- model package
	- results_imgs
'''



Preprocessing for new data:
Parser Argument:
yaml annotation file as in the template "new_data.yaml"

Example:
(base) C:\Users\yesit\PycharmProjects\fsf-signature_detection\Keras-FasterRCNN>
python preprocessing.py --config new_data.yml





Training Process:

Parser Argument:

required argument:

-- main_path
-- train_path

optional argument:

-- num_rois (default:32)
-- network (default: vgg)
-- [Data Augumentation options] (default: False)
horizontal_flips, vertical_flips, rot_90
-- num_epochs (default: 2000) we set to 40 by our training..
-- record_path (default: none)
-- config_filename (default: "./Model/model_config.pickle")
-- output_weight_path (default: './Model/model_frcnn.hdf5')
-- base_weight_path (default: ./Model/vgg16_weights_tf_dim_ordering_tf_kernels.h5")


example:

python train_frcnn.py --main_path "C:/Users/yesit/PycharmProjects/fsf-signature_detection\Keras-FasterRCNN" --train_path "C:\Users\yesit\PycharmProjects\fsf-signature_detection\Keras-FasterRCNN\data\train_2020.txt"

"./Model/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

Testing Process:
example:
python test_frcnn.py --main_path "C:/Users/yesit/PycharmProjects/fsf-signature_detection\Keras-FasterRCNN" --test_path "C:\Users\yesit\PycharmProjects\fsf-signature_detection\Keras-FasterRCNN\data\test_2020.txt" --record_path "./MOdel/record_sig_date.csv" --model_path "C:\Users\yesit\PycharmProjects\fsf-signature_detection\Keras-FasterRCNN\Model\model_frcnn_vgg_sig_date.hdf5"


Prediction Process:

python predict_frcnn.py --model_path "C:\Users\yesit\PycharmProjects\fsf-signature_detection\Keras-FasterRCNN\Model\model_frcnn_vgg_sig_date.hdf5" ----predict_images "C:\Users\yesit\PycharmProjects\fsf-signature_detection\Keras-FasterRCNN\data\predict"
