# curse_detector

### Model Download :  [HERE!](https://drive.google.com/file/d/1gXZ-VYq7rjlhvwgVC0wpTXbOiPsbao3E/view?usp=sharing)
## Unzip and insert in root directory!!

# KcELECTRA를 활용한 욕설 탐지 App

## 개발 목표

- 단순히 금지어 등록을 통한 욕설 탐지가 아닌 NLP Deep Learning을 활용하여 입력한 문장이 욕설인지 아닌지 return하는 모델을 생성한다.
- 안드로이드 app에서 문장입력과 동시에  결과값을 확인할 수 있도록 한다.

---

## Dataset

일간베스트(일베), 오늘의 유머와 같은 각종 사이트의 댓글에 있는 5825 문장을 labeling 해놓은 Dataset 

[https://github.com/2runo/Curse-detection-data/blob/master/dataset.txt](https://github.com/2runo/Curse-detection-data/blob/master/dataset.txt)

[https://smilegate.ai/2020/12/21/korean-curse/](https://smilegate.ai/2020/12/21/korean-curse/)

## Model

- Bert model을 base로 진행하면 좋은 성능을 기대할 수 있지만, pretained model이라고 할 지라도 노트북에서 learning을 진행하기에 어렵다고 생각하여, ELECTRA 모델을 base 모델로 제작된 KcELECTRA를 pretained model로 선택.
- Data 전처리 및 Model train
    
    ```python
    MODEL_NAME = "beomi/KcELECTRA-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    tokenized_train_sentences = tokenizer(
        list(train_data["text"]),
        return_tensors="pt",                # pytorch의 tensor 형태로 return
        max_length=128,                     # 최대 토큰길이 설정
        padding=True,                       # 제로패딩 설정
        truncation=True,                    # max_length 초과 토큰 truncate
        add_special_tokens=True,            # special token 추가
        )
    
    tokenized_test_sentences = tokenizer(
        list(test_data["text"]),
        return_tensors="pt",
        max_length=128,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        )
    
    class CurseDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
    
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
    
        def __len__(self):
            return len(self.labels)
    
    train_label = train_data["curse"].values
    test_label = test_data["curse"].values
    
    train_dataset = CurseDataset(tokenized_train_sentences, train_label)
    test_dataset = CurseDataset(tokenized_test_sentences, test_label)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    raining_args = TrainingArguments(
        output_dir='./',                    # 학습결과 저장경로
        num_train_epochs=10,                # 학습 epoch 설정
        per_device_train_batch_size=8,      # train batch_size 설정
        per_device_eval_batch_size=64,      # test batch_size 설정
        logging_dir='./logs',               # 학습log 저장경로
        logging_steps=500,                  # 학습log 기록 단위
        save_total_limit=2,                 # 학습결과 저장 최대갯수 
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    trainer = Trainer(
        model=model,                         # 학습하고자하는 🤗 Transformers model
        args=training_args,                  # 위에서 정의한 Training Arguments
        train_dataset=train_dataset,         # 학습 데이터셋
        eval_dataset=test_dataset,           # 평가 데이터셋
        compute_metrics=compute_metrics,     # 평가지표
    )
    
    torch.save(model, "./model.pt")
    trainer.evaluate(eval_dataset = test_data)
    ```
    
![Untitled](https://user-images.githubusercontent.com/68226421/188041645-ffdd32c1-b1f0-4e5c-b65a-ac2bb65b7bc3.png)


    
- 모델 output 확인
    
    ```python
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
            result = " >> 악성댓글 👿"
        elif result == 0:
            result = " >> 정상댓글 😀"
        return result
    #0 입력시 종료
    while True: 
        sentence = input("댓글을 입력해주세요: ")
        if sentence == "0":
            break
        print(sentence_predict(sentence), sentence)
        print("\n")
    
    ```
    
    ### Output ex.
    
 ![Untitled 1](https://user-images.githubusercontent.com/68226421/188041639-b49c15fa-356f-40d3-8bf9-e4fef1597527.png)
![Untitled 2](https://user-images.githubusercontent.com/68226421/188041642-89d020dc-9ce7-41af-b7f6-602c511f63f9.png)
    

## Model Serving & Deployment on Web

### Flask

```python
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
```

## Android

- MainActivity.java
    
    ```java
    package com.skt.curse_detector;
    
    import android.os.Bundle;
    import android.view.View;
    import android.widget.Button;
    import android.widget.EditText;
    import android.widget.ImageView;
    import android.widget.TextView;
    import android.widget.Toast;
    
    import androidx.appcompat.app.AppCompatActivity;
    
    import com.android.volley.Request;
    import com.android.volley.RequestQueue;
    import com.android.volley.Response;
    import com.android.volley.VolleyError;
    import com.android.volley.toolbox.StringRequest;
    import com.android.volley.toolbox.Volley;
    import com.bumptech.glide.Glide;
    
    import org.json.JSONException;
    import org.json.JSONObject;
    
    import java.util.HashMap;
    import java.util.Map;
    
    public class MainActivity extends AppCompatActivity {
    
        TextView result, showText;
        EditText sentence;
        Button btn;
        String url = "http://172.23.251.152:5000/predict";
    
        @Override
        protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_main);
    
            ImageView imageView = (ImageView) findViewById(R.id.imageView);
            Glide.with(this).load(R.drawable.pic).into(imageView);
            result = findViewById(R.id.result);
            btn = findViewById(R.id.btn);
            sentence = findViewById(R.id.sentence);
            showText = findViewById(R.id.showText);
    
            btn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    // hit the API -> Volley
                    StringRequest stringRequest = new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {
                        @Override
                        public void onResponse(String response) {
                            try {
                                JSONObject jsonObject = new JSONObject(response);
                                String data = jsonObject.getString("result");
                                if(data.equals("1")){
                                    showText.setText(sentence.getText().toString());
                                    result.setText("결과 : 악성 문장 \uD83D\uDC7F");
                                }else{
                                    showText.setText(sentence.getText().toString());
                                    result.setText("결과 : 정상 문장 \uD83D\uDE00");
                                }
                            } catch (JSONException e) {
                                e.printStackTrace();
                            }
                        }
                    },
                            new Response.ErrorListener() {
                                @Override
                                public void onErrorResponse(VolleyError error) {
                                    Toast.makeText(MainActivity.this, error.getMessage(), Toast.LENGTH_SHORT).show();
                                }
                            }){
                        @Override
                        protected Map getParams(){
                            Map params = new HashMap();
                            params.put("sentence", sentence.getText().toString());
                            return params;
                        }
                    };
                    RequestQueue queue = Volley.newRequestQueue(MainActivity.this);
                    queue.add(stringRequest);
                }
            });
        }
    }
    ```
    
- AndroidManifest.xml
    
    ```xml
    <?xml version="1.0" encoding="utf-8"?>
    <manifest xmlns:android="http://schemas.android.com/apk/res/android"
        package="com.skt.curse_detector">
    
        <uses-permission android:name="android.permission.INTERNET" />
    
        <application
            android:allowBackup="true"
            android:dataExtractionRules="@xml/data_extraction_rules"
            android:fullBackupContent="@xml/backup_rules"
            android:icon="@mipmap/ic_launcher"
            android:label="@string/app_name"
            android:roundIcon="@mipmap/ic_launcher_round"
            android:supportsRtl="true"
            android:theme="@style/Theme.Curse_detector"
            android:networkSecurityConfig="@xml/network_security_config"
            android:usesCleartextTraffic="true">
            <activity
                android:name=".MainActivity"
                android:exported="true">
                <intent-filter>
                    <action android:name="android.intent.action.MAIN" />
    
                    <category android:name="android.intent.category.LAUNCHER" />
                </intent-filter>
            </activity>
        </application>
    
    </manifest>
    ```
    
- activity_main.xml
    
    ```xml
    <?xml version="1.0" encoding="utf-8"?>
    <androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
        xmlns:app="http://schemas.android.com/apk/res-auto"
        xmlns:tools="http://schemas.android.com/tools"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        tools:context=".MainActivity">
    
        <ImageView
            android:id="@+id/imageView"
            android:layout_width="match_parent"
            android:layout_height="400dp"
            android:src="@drawable/pic"
            android:scaleType="centerCrop"
            app:layout_constraintBottom_toTopOf="@+id/sentence"
            app:layout_constraintEnd_toEndOf="parent"
            app:layout_constraintHorizontal_bias="0.0"
            app:layout_constraintStart_toStartOf="parent"
            app:layout_constraintTop_toTopOf="parent" />
    
        <EditText
            android:id="@+id/sentence"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="10dp"
            android:fontFamily="@font/d2coding"
            android:gravity="center"
            android:hint="판별할 문장을 입력하세요!"
            android:minHeight="48dp"
            app:layout_constraintBottom_toTopOf="@+id/btn"
            app:layout_constraintEnd_toEndOf="parent"
            app:layout_constraintHorizontal_bias="0.0"
            app:layout_constraintStart_toStartOf="parent"
            app:layout_constraintTop_toBottomOf="@+id/imageView" />
    
        <Button
            android:id="@+id/btn"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_marginStart="162dp"
            android:layout_marginTop="8dp"
            android:layout_marginEnd="162dp"
            android:layout_marginBottom="32dp"
            android:backgroundTint="#00bf00"
            android:fontFamily="@font/d2coding"
            android:gravity="center"
            android:text="판독"
            app:layout_constraintBottom_toTopOf="@+id/showText"
            app:layout_constraintEnd_toEndOf="parent"
            app:layout_constraintHorizontal_bias="1.0"
            app:layout_constraintStart_toStartOf="parent"
            app:layout_constraintTop_toBottomOf="@+id/sentence" />
    
        <TextView
            android:id="@+id/showText"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="8dp"
            android:fontFamily="@font/d2coding"
            android:text="입력받은 문장"
            android:textSize="20dp"
            app:layout_constraintTop_toBottomOf="@+id/btn"
            tools:layout_editor_absoluteX="0dp" />
    
        <TextView
            android:id="@+id/result"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_gravity="bottom"
            android:layout_marginTop="8dp"
            android:fontFamily="@font/d2coding"
            android:text="결과 : "
            android:textSize="20dp"
            app:layout_constraintTop_toBottomOf="@+id/showText"
            tools:layout_editor_absoluteX="0dp" />
    
    </androidx.constraintlayout.widget.ConstraintLayout>
    ```
    

## Check

[https://lostark.game.onstove.com/Community/Free/Views/2267182?page=5505&searchtype=0&searchtext=&ordertype=like&communityNo=541](https://lostark.game.onstove.com/Community/Free/Views/2267182?page=5505&searchtype=0&searchtext=&ordertype=like&communityNo=541)

![KakaoTalk_20220812_111731883](https://user-images.githubusercontent.com/68226421/188041626-712d10a5-331e-41df-a863-f0ecd57654f1.png)
![KakaoTalk_20220812_111731883_01](https://user-images.githubusercontent.com/68226421/188041630-5c7000ea-4058-426e-8be1-e20de8572e36.png)
![KakaoTalk_20220812_111731883_02](https://user-images.githubusercontent.com/68226421/188041638-e6227240-176b-4cf2-b3fd-f3c2de78d4a1.png)


