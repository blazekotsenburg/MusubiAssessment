import pandas as pd
from llm_classifier import classifier

df = pd.read_csv("synthetic_dataset.csv")

df.insert(0, "id", range(1, len(df) + 1))

# This is the llm dataframe to capture the output and later join with synthetic data.
llm_df = pd.DataFrame({
    "id": df["id"],
    "ModelLabel": "",
    "Reason": ""
})

correct=0
llm_output=[]
for idx, row in df.iterrows():
    content = row["Content"]
    expected = row["Expected"]

    result = classifier(content)
    
    llm_output.append({
        "id": idx + 1,
        "ModelLabel": result["label"],
        "Reason": result["reason"]
    })
    
    print("----------------------------")
    print("ID:", idx)
    print("Expected:", expected)
    print("Label:", result["label"])
    print("Accuracy:", correct / df.shape[0])
    print("----------------------------")

    if result["label"] == expected:
        correct +=1
accuracy = correct / df.shape[0]
print("Accuracy:", accuracy)

llm_df = pd.DataFrame(llm_output)
final_df = pd.merge(df, llm_df, on="id")
final_df.to_csv("final_output_hotfix.csv", index=False)


with open("llm_eval.txt", 'w') as f:
    f.write("Accuracy of the llm was: " + str(accuracy))