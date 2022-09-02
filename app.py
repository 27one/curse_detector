from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

app = Flask(__name__)

model = torch.load('./static/model.pt')

MODEL_NAME = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = torch.device('cuda:0' if torch.cuda. is_available() else 'cpu')


def sentence_predict(sent):
    # 평가모드로 변경
    model.eval()

    # 입력된 문장 토크나이징
    tokenized_sent = tokenizer(
        sent,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        max_length=128
    )

    # 모델이 위치한 GPU로 이동
    tokenized_sent.to(device)

    # 예측
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_sent["input_ids"],
            attention_mask=tokenized_sent["attention_mask"],
            token_type_ids=tokenized_sent["token_type_ids"]
        )

    # 결과 return
    logits = outputs[0]
    logits = logits.detach().cpu()
    result = logits.argmax(-1)

    if result == 1:
        result = "1"
    elif result == 0:
        result = "0"

    return result
    
@app.route('/predict', methods=['POST'])
def prediction():  # put application's code here
    sentence = request.form.get("sentence")
    res = sentence_predict(sentence)
    return jsonify({"result": res})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)