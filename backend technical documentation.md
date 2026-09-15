## Technical design and data flow

1.	A user pastes email text into the extension or explicitly selects Read open Gmail email. The extension attempts to extract the subject and the last visible message body from the active Gmail page. Gmail DOM selectors may change; the extracted text is shown for user review before sending.
2.	Selecting Analyse email sends a JSON body with text to http://127.0.0.1:5000/api/analyse. The popup does not silently send text merely because a Gmail tab is open.
3.	Flask checks that the body is a JSON object and text is a non-empty string. The predictor instance has already been created at startup, so the large checkpoint is not reloaded for each request.
4.	predictor.py applies its email normalisation, extracts twelve NLP features, scales them using saved mean and standard deviation, and tokenises text with the local tokenizer. The model consumes tokens/attention mask plus scaled features.
5.	PyTorch inference produces two class probabilities. The served code displays label 0 as legitimate and label 1 as phishing; the exported checkpoint's training label mapping must still be cross-checked. The response includes the predicted verdict and score for that verdict.
6.	The LIME text explainer perturbs the email, calls the same model-prediction function on nearby examples, and fits a local surrogate. Its signed term weights explain the selected prediction locally, not whether the email is truly fraudulent.
7.	Flask returns JSON; the extension displays a verdict, score, explanatory terms and connection/validation errors. The extension itself does not train or fine-tune the model.
This is a hybrid text-and-feature system, not a plain BERT text-only model. Any changes to normalisation, feature extraction, feature order, tokenizer, max length or model layer dimensions could make serving inconsistent with training.

