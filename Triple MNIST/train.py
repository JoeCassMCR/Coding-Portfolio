"""
Training orchestration for Triple MNIST:
 1) Logistic Regression baseline
 2) Basic CNN
 3) Multi-head CNN (advanced & split)
 4) DCGAN augmentation + final CNN
 5) Visualization
"""

import os
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tensorflow.keras.utils import to_categorical
from dataloader import load_images, split_labels
from models import create_basic_cnn, create_multi_head_cnn, build_generator
from utils import IMAGE_SIZE, BATCH_SIZE, EPOCHS, NOISE_DIM, early_stop, save_figure, ensure_directory
import matplotlib.pyplot as plt


# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset2', 'triple_mnist')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR   = os.path.join(DATA_DIR, 'val')
TEST_DIR  = os.path.join(DATA_DIR, 'test')
VIS_DIR   = os.path.join(BASE_DIR, 'assets', 'visualizations')
ensure_directory(VIS_DIR)


# 1) Logistic Regression baseline
logging.info("==> Logistic Regression Baseline")
x_tr_flat, y_tr = load_images(TRAIN_DIR, flatten=True)
x_te_flat, y_te = load_images(TEST_DIR, flatten=True)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(x_tr_flat, y_tr)
preds = logreg.predict(x_te_flat)
logging.info(f"LogReg Accuracy: {accuracy_score(y_te,preds):.4f}")
logging.info(f"LogReg F1: {f1_score(y_te,preds,average='weighted'):.4f}")
logging.info(f"LogReg Report:\n{classification_report(y_te,preds,zero_division=1)}")

# 2) Basic CNN
logging.info("==> Basic CNN")

x_tr, y_tr = load_images(TRAIN_DIR)
x_vl, y_vl = load_images(VAL_DIR)
x_te, y_te = load_images(TEST_DIR)
x_tr_c = x_tr.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)/255.0
x_vl_c = x_vl.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)/255.0
x_te_c = x_te.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)/255.0

basic = create_basic_cnn()
basic.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

# one-hot labels for 3 digits
y_tr_oh = np.array([to_categorical([int(d) for d in f"{lbl:03d}"],10).flatten() for lbl in y_tr])
y_vl_oh = np.array([to_categorical([int(d) for d in f"{lbl:03d}"],10).flatten() for lbl in y_vl])

history = basic.fit(x_tr_c, y_tr_oh,
                    validation_data=(x_vl_c, y_vl_oh),
                    epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[early_stop])

pred_basic = basic.predict(x_te_c).reshape(-1,3,10)
basic_labels = np.array([int(''.join(map(str,np.argmax(p,axis=1)))) for p in pred_basic])
logging.info(f"Basic CNN F1: {f1_score(y_te,basic_labels,average='weighted'):.4f}")


# 3) Multi-head CNNs (advanced & split use same function)
logging.info("==> Multi-head CNN")
labels_tr_s = split_labels(y_tr)
labels_vl_s = split_labels(y_vl)
labels_te_s = split_labels(y_te)

adv = create_multi_head_cnn()
adv.compile(optimizer='adam',
            loss={name: 'sparse_categorical_crossentropy' for name in adv.output_names},
            metrics={name:'accuracy' for name in adv.output_names})

adv.fit( x_tr_c,
        {adv.output_names[i]: labels_tr_s[:,i] for i in range(len(adv.output_names))},
        validation_data=( x_vl_c,
        {adv.output_names[i]: labels_vl_s[:,i] for i in range(len(adv.output_names))}),
        epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[early_stop])

preds_adv = adv.predict(x_te_c)
for i,p in enumerate(preds_adv):
    logging.info(f"Adv CNN Digit {i+1} F1: {f1_score(labels_te_s[:,i], np.argmax(p,axis=1), average='macro'):.4f}")

# 4) DCGAN augmentation + Final CNN
logging.info("==> DCGAN + Final CNN")
gen = build_generator()
# limit synthetic images to avoid OOM
num_syn = 10000
noise   = np.random.normal(0, 1, (num_syn, NOISE_DIM))
syn     = (gen.predict(noise, batch_size=256) + 1) / 2.0

# augment
x_aug = np.concatenate([x_tr_c, syn], axis=0)
y_aug = np.concatenate([labels_tr_s, labels_tr_s[:num_syn]], axis=0)

final = create_multi_head_cnn()
final.compile(
    optimizer='adam',
    loss={name: 'sparse_categorical_crossentropy'
          for name in final.output_names},
    metrics={name: 'accuracy' for name in final.output_names}
)
final.fit(
    x_aug,
    {name: y_aug[:, i] for i, name in enumerate(final.output_names)},
    validation_data=(
        x_vl_c,
        {name: labels_vl_s[:, i] for i, name in enumerate(final.output_names)}
    ),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

preds_final = final.predict(x_te_c)
for i, pred in enumerate(preds_final):
    score = f1_score(labels_te_s[:, i], np.argmax(pred, axis=1), average='macro')
    logging.info(f"Final CNN Digit {i+1} F1: {score:.4f}")


# 5) Visualization of some test samples
logging.info("==> Visualization")
fig, axs = plt.subplots(1,3,figsize=(12,4))
for i in range(3):
    axs[i].imshow(x_te[i].reshape(IMAGE_SIZE,IMAGE_SIZE), cmap='gray')
    axs[i].set_title(f"Label: {y_te[i]}")
    axs[i].axis('off')
plt.tight_layout()
out_path = os.path.join(VIS_DIR, 'test_samples.png')
save_figure(fig, out_path)
