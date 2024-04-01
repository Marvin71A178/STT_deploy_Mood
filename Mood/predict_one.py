from simpletransformers.classification import ClassificationModel,  ClassificationArgs
import time
import numpy as np

# 預測情緒
def prediction(listTestData):
    # 輸出模型存在的目錄名稱
    # dir_name = './StoryTeller_docker/mood_analized/checkpoint-20000' 
    dir_name = './Mood/checkpoint-20000'
    # 自訂參數
    model_args = ClassificationArgs()
    model_args.train_batch_size = 64
    model_args.num_train_epochs = 3

    # 讀取 ClassificationModel
    model = ClassificationModel(
        'bert', 
        f"{dir_name}", # 這裡要改成訓練完成的模型資料夾路徑
        use_cuda=True, 
        cuda_device=0, 
        num_labels=6, 
        args=model_args
    )

    # 預測
    predictions, raw_outputs = model.predict([listTestData])
    result = predictions
    # 回傳預測結果，會是一個 list
    return result

def mood_ana_api(input_string):
    Data = [str(input_string)]
    ans = int(prediction(Data)[0])
    return ans
    

# 主程式
if __name__ == "__main__":
    # 計時開始
    tStart = time.time()

    # 準備預測情緒類別。語料可以不只一句！
    listTestData = "我喜歡這個欣欣向榮的春天"

    # 進行預測
    test = mood_ana_api(listTestData)
    print(test)
    print(type(test))
    
    # 計時結束
    tEnd = time.time()
    # 輸出程式執行的時間
    print(f"執行花費 {tEnd - tStart} 秒。")