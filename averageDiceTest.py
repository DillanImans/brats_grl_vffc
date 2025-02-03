import pandas as pd

def calculate_average_dice_scores(file_path):
    try:
        data = pd.read_csv(file_path)

        required_columns = ['et_dice', 'tc_dice', 'wt_dice']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' is missing in the input CSV.")

        average_dice_scores = data[required_columns].mean()

        mean_dice_score = average_dice_scores.mean()

        print("Average Dice Scores:")
        print(average_dice_scores)
        print("Mean Dice Score:")
        print(f"{mean_dice_score:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

modality = 'flair'
model = 'SEGRESNET_t1cet2bezier'

file_path = f"/home/monetai/Desktop/dillan/code/brrr/Brain-Tumors-Segmentation/archiveModels/{model}/csv/{modality}_test.csv"

calculate_average_dice_scores(file_path)