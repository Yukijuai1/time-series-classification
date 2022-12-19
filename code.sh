#执行SVM、KNN和LearningShapelet分类
python main/classify.py
#执行Shapelet分类
python main/shapelet_classify.py
#执行ShapeNet分类
python main/shapenet_classify.py --dataset BasicMotions --path .. --save_path main/classification/shapenet/model/ --hyper main/classification/shapenet/default_parameters.json --cuda --load
