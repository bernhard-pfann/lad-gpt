# Train a Language Model on your WhatsApp Chats
## Overview
This repository facilitates the training of a character-level or word-level language model solely based on WhatsApp chat messages. After model training, one can kick-off a synthetic conversation with the trained on Whatsapp chat group. <br>
- <b>Chat messages:</b> I have privately trained the model on Whatsapp chats from a group with >8 Mio characters. The <code>assets/input/chat.txt</code> is just a placeholder, to be replaced with the actual corpus of chat messages.
- <b>Language model:</b>
The model closely follows the architecture introduced in "Attention Is All You Need" (2017) by Vaswani et. al.. Also the pytorch implementation of the model is heavily inspired by a [video tutorial by Andrew Kaparty](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- <b>Results:</b> While the overall performance of my privately trained model is clearly not comparable with sota language models, the generated text clearly exhibits recognizable lingusitic patterns and vocabulary.

## Folder Structure
```
|-- assets
|   |-- input
|   |   |-- chat.txt
|   |-- output
|   |   |-- contacts.txt
|   |   |-- vocab.txt
|   |   |-- train.pt
|   |   |-- valid.pt
|   |-- models
|   |   |--model.pt
|-- src
|   |-- chat.py
|   |-- model.py
|   |-- preprocess.py
|   |-- train.py
|   |-- utils.py
|-- config.py
|-- run.py
```

### Assets Description:
- <code>assets/input/chat.txt:</code> The input file needs to be an exported WhatsApp chat (without media).
- <code>assets/output/:</code> The encoded training/validation data and the trained model will be written into this localtion.
- <code>assets/models/model.pt:</code> Trained pytorch model object.

### Module Description:
- <code>src/preprocess.py:</code> Converts chat messages into encoded PyTorch tensors. Data is split into training and validation set.
- <code>src/model.py:</code> Defines the language model class.
- <code>src/train.py:</code> Contains code for training the language model.
- <code>src/chat.py:</code> Contains the function for conversational interaction with the model.
- <code>src/utils.py:</code> Other useful utility functions.
- <code>run.py:</code> The main script with an argument parser to call either of the three actions ("preprocess", "train", "chat").
- <code>config.py:</code> Parameters for preprocessing and model training are recorded.

## How to Get Started
### Installation:
```
git clone https://github.com/bernhard-pfann/lad-gpt.git
cd lad-gpt
pip install -r requirements.txt
```

To utilize this project fully, you'll need a .txt file that contains messages from a WhatsApp chat. Here are the steps to export your WhatsApp group chat into a .txt file:

### For Android Users:
1. Open WhatsApp and Navigate to Group Chat: Open the WhatsApp application on your Android device and go to the group chat you wish to export.
2. Tap on the Three Dots: These are usually at the top right corner of the chat window.
3. More -> Export Chat: Choose 'More' from the drop-down and then select 'Export chat'.
4. Choose Without Media: You'll get an option to include or exclude media. Choose 'Without Media' to export only the text messages.
5. Select Export Method: You will be prompted to select how you want to export the chat. You can send it to your email, and from there, download it as a .txt file.

### For iPhone Users:
1. Open WhatsApp and Navigate to Group Chat: Open the WhatsApp application on your iPhone and navigate to the group chat you want to export.
2. Tap on the Group Name: This is at the top of the chat window to go to 'Group Info'.
3. Scroll Down and Export Chat: Scroll down and you'll see an 'Export Chat' option. Tap on it.
4. Choose Without Media: A pop-up will appear asking if you want to include media files. Select 'Without Media'.
5. Select Export Method: Choose an option to export the chat, for example, through Mail. You can then download the text file from your email.

Once you have the .txt file, place it in the <code>assets/input</code> directory, called <code>chat.txt</code>. Then you are ready to go!

## Terminal Instructions

Once input data is in place, the chats need to be encoded into numerical tensors. The encoded data is also split into training and validation set:
```
python run.py preprocess
```
To train a language model from scratch and solely based on the encoded chat data. Set <code>--update</code> in case you want to continue training an already model.
```
python run.py train --update
```
To initiale a chat with the trained model:
```
python run.py chat
```