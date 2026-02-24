from src.models import models

def choose_model():
    model_options = {
        1: ("XGBoost", models.get_xgboost),
        2: ("Random Forest", models.get_random_forest),
        3: ("Random Forest Search", models.get_random_forest_search)
    }
    print("\nChoose the model to execute:\n")
    for key, (name, _) in model_options.items():
        print(f"{key}. {name}")
    
    while True:
        try:
            choice = int(input("Enter your choice: "))
            if choice in model_options:
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid number.")
            
    return model_options[choice]