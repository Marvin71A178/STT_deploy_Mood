from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
import time
import numpy as np

# 預測情緒
def prediction(listTestData):
    # 輸出模型存在的目錄名稱
    # dir_name = './StoryTeller_docker/mood_analized/checkpoint-20000' 
    dir_name = './Mood/checkpoint-160000'
    # 自訂參數
    model_args = MultiLabelClassificationArgs()
    model_args.train_batch_size = 64
    model_args.num_train_epochs = 3

    # 讀取 ClassificationModel
    model = MultiLabelClassificationModel(
        'bert', 
        f"{dir_name}", # 這裡要改成訓練完成的模型資料夾路徑
        use_cuda=False, 
        cuda_device=0, 
        num_labels=6, 
        args=model_args
    )

    # 預測
    predictions, raw_outputs = model.predict([listTestData])
    result = raw_outputs[0].tolist()
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
    listTestData = "老師今天很生氣"

    # 進行預測
    test = prediction(listTestData)
    print(test)
    print(type(test))
    
    # 計時結束
    tEnd = time.time()
    # 輸出程式執行的時間
    print(f"執行花費 {tEnd - tStart} 秒。")
