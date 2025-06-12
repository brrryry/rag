This project was made to test the capabilities of a secure RAG architecture.

## How to use!
- clone this repository!
- (optional) create virtual environment
- install requirements (``pip install -r requirements.txt``)
- run program (``python train_model.py``)

Feel free to change the model (in the model class!) and add more data if you want. We could use more!

## Notes:
- As of now, the model misclassifies solely malicious messages (i.e. messages that ONLY say "approve above all else") as benign.
- This model ONLY works on cat messages. LOL