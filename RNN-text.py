from sklearn import datasets
from sklearn import svm, metrics
import matplotlib.pyplot as plt
%matplotlib inline

# The digits dataset
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images,digits.target))
for index, (image,label) in enumerate(images_and_labels[:4]):
    plt.subplot(2,4, index + 1)
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Training: %i" % label)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))

# Create a classifier
classifier = svm.SVC(gamma=0.001)
# Use the first half of the data to train
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2 ])
# Use the second half of the data to test
expected = digits.target[n_samples // 2 :]
predicted = classifier.predict(data[n_samples // 2 :])

print ("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected,predicted)))
print ("Confusion matrix: \n%s" % metrics.confusion_matrix(expected,predicted))

images_and_predictions = list(zip(digits.images[n_samples //2 :],predicted))
for index, (image,prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2,4,index + 5)
    plt.axis("off")
    plt.imshow(image,cmap=plt.cm.gray_r, interpolation = "nearest")
    plt.title("Prediction: %i" % prediction)
    
#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()


# In[3]:


import matplotlib.pyplot as plt

def plot_visualization(record,memo):
  plt.plot(record.record[memo])
  plt.plot(record.record['val_'+memo], '')
  plt.xlabel("Epochs")
  plt.ylabel(memo)
  plt.legend([memo, 'val_'+memo])


# In[4]:


dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset.element_spec


# In[5]:


for example, label in train_dataset.take(2):
  print('text: ', example.numpy())
  print('label: ', label.numpy())


# In[6]:


BUFFER_SIZE = 5000
BATCH_SIZE = 32


# In[7]:


train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


# In[8]:


for example, label in train_dataset.take(2):
  print('texts: ', example.numpy()[:3])
  print()
  print('labels: ', label.numpy()[:3])


# In[9]:


VOCAB_SIZE=5000
encoder1 = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder1.adapt(train_dataset.map(lambda text, label: text))


# In[10]:


voc = np.array(encoder1.get_vocabulary())
voc[:20]


# In[11]:


encoded_eg = encoder1(example)[:3].numpy()
encoded_eg


# In[12]:


for n in range(3):
  print("Original: ", example[n].numpy())
  print("Round-trip: ", " ".join(voc[encoded_eg[n]]))
  print()


# In[13]:


model_number1 = tf.keras.Sequential([
    encoder1,
    tf.keras.layers.Embedding(
        input_dim=len(encoder1.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(56, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[14]:


print([layer.supports_masking for layer in model_number1.layers])


# In[15]:


sample_text_bad = ('The movie was a shit. The animation is just a joke.'
               'too bad. I should not recommend this movie.')
predictions_time1 = model_number1.predict(np.array([sample_text_bad]))
print(predictions_time1[0])


# In[16]:


padding_mine = "the " * 2000
predictions_time2 = model_number1.predict(np.array([sample_text_bad, padding_mine]))
print(predictions_time2[0])


# In[17]:


model_number1.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[18]:


record = model_number1.fit(train_dataset, epochs=5,
                    validation_data=test_dataset, 
                    validation_steps=100)


# In[19]:


test_loss, test_acc = model_number1.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[20]:


plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plot_visualization(record, 'accuracy')
plt.ylim(None,1)
plt.subplot(1,2,2)
plot_visualization(record, 'loss')
plt.ylim(0,None)


# In[21]:


sample_text_bad = ('The movie was a shit. The animation is just a joke.'
               'too bad. I should not recommend this movie.')
prediction_time3 = model_number1.predict(np.array([sample_text_bad]))


# In[22]:


model_number2 = tf.keras.Sequential([
    encoder1,
    tf.keras.layers.Embedding(len(encoder1.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])


# In[23]:


model_number2.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[24]:


record2 = model_number2.fit(train_dataset, epochs=5,
                    validation_data=test_dataset,
                    validation_steps=15)


# In[25]:


test_loss, test_acc = model_number2.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[26]:


sample_text_bad = ('The movie was a shit. The animation is just a joke.'
               'too bad. I should not recommend this movie.')
prediction_time4 = model_number2.predict(np.array([sample_text_bad]))
print(prediction_time4)


# In[ ]:


plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plot_visualization(record, 'accuracy')
plt.subplot(1,2,2)
plot_visualization(record, 'loss')

