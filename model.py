import numpy as np


def predict_age(image, model):
    #   im = plt.imread(path)
    # im = image.img_to_array(image)
    im = image * 1./255
    im = np.resize(im, (224, 224, 3))
    im = np.expand_dims(im, axis=0)
    # regr, classif = model.predict(im)
    # regr_pred_age = "%.2f" % regr[0][0]*100
    # classif = list(classif[0])
    # pred_age_dict = {
    #     0: "20-25",
    #     1: "25-32",
    #     2: "33-43",
    #     3: "44-50",
    # }
    # age_index = classif.index(max(classif))
    # class_pred_age = pred_age_dict[age_index]
    # return f"Regression: {classif}"
    # return f"Classification: {class_pred_age} \n Regression: {regr_pred_age}"
    return f"{model.predict(im)[0][0]}"
