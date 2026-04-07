# Tamil Event Classifier

Sentence-level multiclass event classification using Tamil text with English translation.

## Labels

- Sports
- Politics
- Weather
- Accident
- Entertainment
- Education
- Crime

## Files

- `preprocess.py` - Tamil text cleaning and tokenization
- `translate.py` - Tamil to English translation using `googletrans`
- `train_model.py` - model training, evaluation, and saving
- `predict.py` - prediction from Tamil sentence
- `app.py` - simple CLI wrapper
- `streamlit_app.py` - simple web UI using Streamlit
- `dataset.csv` - original starter dataset
- `dataset_tamil.csv` - expanded UTF-8 Tamil dataset used for training

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train_model.py
```

or

```bash
python app.py train
```

## Predict

```bash
python predict.py --text "தமிழ்நாட்டில் கடும் மழை"
```

or

```bash
python app.py predict --text "தமிழ்நாட்டில் கடும் மழை"
```

## Streamlit App

```bash
streamlit run streamlit_app.py
```

## Notes

- The classifier is trained on Tamil text directly for better compatibility and stronger baseline performance.
- During prediction, English translation is optional output only and is not required for classification.
- If translation fails because internet is unavailable or `googletrans` is incompatible, the app returns a safe fallback instead of crashing.
