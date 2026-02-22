from src import process_manager
from src.visualization import maps
import sys

def main():
    try:
        ans = process_manager.choose_option()
        process_manager.choose_sub_option(ans)
        # main_model = train_model(
        #     model=xgb, X_train=X_train, y_train=y_train
        # )
        
        # probs = evaluate_model(
        #     model=main_model, X_test=X_train, y_test=y_train, features=features1
        # )
        
        # future = generate_forecast(
        #     model=main_model, df=future, features=features1
        # )

        # maps.plot_month_map(
        #     future,
        #     year=2026,
        #     month=1,
        #     title="Wildfire Forecast – January 2026",
        #     save_path="wildfire_risk_jan_2026.jpg"
        # )
        # print("=== Random Forest ===")
    finally:
        print("Cleaning processes")
if __name__ == "__main__":
    main()
    sys.exit(1)    